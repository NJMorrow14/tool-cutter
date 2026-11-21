import Foundation
import CoreML
import CoreGraphics
import CoreVideo

// SAM 2.1 Small (apple/coreml-sam2.1-small, Apache-2.0) through the GENERIC MLModel API, so the same file runs in the
// app and in `ios/tests/run_sam2_test.sh` on the Mac. Tensor handling mirrors sam2-studio's SAM2.swift:
//   image encoder   in  image            1024 x 1024 BGRA pixel buffer (the plain-scaled photo, no letterbox)
//                   out image_embedding  [1,256,64,64]  feats_s0 [1,32,256,256]  feats_s1 [1,64,128,128]   (float16)
//   prompt encoder  in  points [1,N,2] (1024 space, x right / y down)  labels [1,N] (1 fg, 0 bg, 2 box origin, 3 box end)
//                   out sparse_embeddings [1,N+1,256]  dense_embeddings [1,256,64,64]
//   mask decoder    in  image_embedding, sparse_embedding, dense_embedding, feats_s0, feats_s1
//                   out low_res_masks [1,3,256,256] (logits)  scores [1,3]
// The decoder accepts 2...15 sparse tokens, so at most 14 prompts per call.
// Compute units .cpuAndGPU as the reference does: the Neural Engine path made the demo fail (and on the Mac the ANE
// compile simply hangs).

public enum SAM2Error: Error, CustomStringConvertible {
    case modelMissing(String)
    case notEncoded
    case pixelBufferCreationFailed
    case badPrompts(String)
    case missingOutput(String)

    public var description: String {
        switch self {
        case .modelMissing(let s): return "SAM2 model missing: \(s)"
        case .notEncoded: return "SAM2: encode(_:) an image before predicting"
        case .pixelBufferCreationFailed: return "SAM2: could not create the 1024x1024 input buffer"
        case .badPrompts(let s): return "SAM2 prompts: \(s)"
        case .missingOutput(let s): return "SAM2: model output '\(s)' missing"
        }
    }
}

public final class SAM2Engine: @unchecked Sendable {
    /// A prompt in ORIGINAL image pixels (x right, y down). label: 1 foreground, 0 background, 2 box origin, 3 box end.
    public struct Prompt: Sendable, Equatable {
        public var x: Double
        public var y: Double
        public var label: Int
        public init(x: Double, y: Double, label: Int) { self.x = x; self.y = y; self.label = label }
    }

    public static let inputSide = 1024
    public static let maskSide = 256
    public static let maxPrompts = 14

    private let imageEncoder: MLModel
    private let promptEncoder: MLModel
    private let maskDecoder: MLModel
    private let lock = NSLock()
    private var embedding: (image: MLMultiArray, s0: MLMultiArray, s1: MLMultiArray)?
    private var originalSize: (width: Int, height: Int) = (0, 0)

    public init(modelsDirectory: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU
        func load(_ name: String) throws -> MLModel {
            let url = modelsDirectory.appendingPathComponent("SAM2_1Small\(name)FLOAT16.mlmodelc")
            guard FileManager.default.fileExists(atPath: url.path) else { throw SAM2Error.modelMissing(url.path) }
            return try MLModel(contentsOf: url, configuration: cfg)
        }
        imageEncoder = try load("ImageEncoder")
        promptEncoder = try load("PromptEncoder")
        maskDecoder = try load("MaskDecoder")
    }

    /// Loads the three models off the calling thread (~1-2 s the first time, longer while Core ML specialises them).
    public static func load(modelsDirectory: URL) async throws -> SAM2Engine {
        try await Task.detached(priority: .userInitiated) { try SAM2Engine(modelsDirectory: modelsDirectory) }.value
    }

    /// True once an image has been encoded and prompts can be answered.
    public var isReady: Bool { lock.lock(); defer { lock.unlock() }; return embedding != nil }

    /// Scale the image to 1024 x 1024 (plain scale, as the models were converted) and run the image encoder.
    public func encode(_ image: CGImage) throws {
        let pb = try Self.pixelBuffer(from: image, side: Self.inputSide)
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)])
        let out = try imageEncoder.prediction(from: input)
        guard let emb = out.featureValue(for: "image_embedding")?.multiArrayValue else { throw SAM2Error.missingOutput("image_embedding") }
        guard let s0 = out.featureValue(for: "feats_s0")?.multiArrayValue else { throw SAM2Error.missingOutput("feats_s0") }
        guard let s1 = out.featureValue(for: "feats_s1")?.multiArrayValue else { throw SAM2Error.missingOutput("feats_s1") }
        lock.lock()
        embedding = (emb, s0, s1)
        originalSize = (image.width, image.height)
        lock.unlock()
    }

    /// Prompt encoder + mask decoder; the best-scoring of the 3 masks, its 256 x 256 logits upsampled (bilinear) to the
    /// ORIGINAL image size and thresholded at 0.
    public func predictMask(prompts: [Prompt]) throws -> BitMask {
        let (logits, side) = try predictLogits(prompts: prompts)
        lock.lock(); let (W, H) = originalSize; lock.unlock()
        return Self.upsampleThreshold(logits, side: side, width: W, height: H, threshold: 0)
    }

    /// The raw best 256 x 256 logits (row-major, row 0 = top of the image), for callers with their own threshold.
    public func predictLogits(prompts: [Prompt]) throws -> (logits: [Float], side: Int) {
        lock.lock()
        let emb = embedding, size = originalSize
        lock.unlock()
        guard let emb else { throw SAM2Error.notEncoded }
        guard !prompts.isEmpty else { throw SAM2Error.badPrompts("no prompts") }
        guard prompts.count <= Self.maxPrompts else { throw SAM2Error.badPrompts("\(prompts.count) prompts, at most \(Self.maxPrompts)") }

        // prompt encoder (float32 points / int32 labels, as the reference sends them; Core ML converts to float16)
        let n = prompts.count
        let points = try MLMultiArray(shape: [1, NSNumber(value: n), 2], dataType: .float32)
        let labels = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
        let sx = Double(Self.inputSide) / Double(max(1, size.width)), sy = Double(Self.inputSide) / Double(max(1, size.height))
        for (i, p) in prompts.enumerated() {
            points[[0, i, 0] as [NSNumber]] = NSNumber(value: Float(p.x * sx))
            points[[0, i, 1] as [NSNumber]] = NSNumber(value: Float(p.y * sy))
            labels[[0, i] as [NSNumber]] = NSNumber(value: Int32(p.label))
        }
        let pin = try MLDictionaryFeatureProvider(dictionary: ["points": MLFeatureValue(multiArray: points),
                                                               "labels": MLFeatureValue(multiArray: labels)])
        let pout = try promptEncoder.prediction(from: pin)
        guard let sparse = pout.featureValue(for: "sparse_embeddings")?.multiArrayValue else { throw SAM2Error.missingOutput("sparse_embeddings") }
        guard let dense = pout.featureValue(for: "dense_embeddings")?.multiArrayValue else { throw SAM2Error.missingOutput("dense_embeddings") }

        // mask decoder
        let din = try MLDictionaryFeatureProvider(dictionary: [
            "image_embedding": MLFeatureValue(multiArray: emb.image),
            "sparse_embedding": MLFeatureValue(multiArray: sparse),
            "dense_embedding": MLFeatureValue(multiArray: dense),
            "feats_s0": MLFeatureValue(multiArray: emb.s0),
            "feats_s1": MLFeatureValue(multiArray: emb.s1),
        ])
        let dout = try maskDecoder.prediction(from: din)
        guard let masks = dout.featureValue(for: "low_res_masks")?.multiArrayValue else { throw SAM2Error.missingOutput("low_res_masks") }
        guard let scores = dout.featureValue(for: "scores")?.multiArrayValue else { throw SAM2Error.missingOutput("scores") }

        let nm = scores.count
        var best = 0, bestScore = -Double.infinity
        for i in 0..<nm {
            let s = scores[i].doubleValue
            if s > bestScore { bestScore = s; best = i }
        }
        let all = (0..<nm).map { scores[$0].floatValue }
        lock.lock(); _lastScores = all; lock.unlock()
        let h = masks.shape[2].intValue, w = masks.shape[3].intValue
        return (Self.slice(masks, index: best, height: h, width: w), w)
    }

    /// Scores of the 3 candidate masks from the last prediction (diagnostics).
    public var lastScores: [Float] { lock.lock(); defer { lock.unlock() }; return _lastScores }
    private var _lastScores: [Float] = []

    // MARK: - Helpers

    /// Copy masks[0, index, :, :] out as Float, honouring strides and the array's data type.
    private static func slice(_ a: MLMultiArray, index: Int, height h: Int, width w: Int) -> [Float] {
        var out = [Float](repeating: 0, count: h * w)
        let st = a.strides.map(\.intValue)
        let base = index * st[1]
        switch a.dataType {
        case .float32:
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<h { for x in 0..<w { out[y * w + x] = p[base + y * st[2] + x * st[3]] } }
        case .double:
            let p = a.dataPointer.assumingMemoryBound(to: Double.self)
            for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[base + y * st[2] + x * st[3]]) } }
        default:
            #if arch(arm64)
            if a.dataType == .float16 {
                let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
                for y in 0..<h { for x in 0..<w { out[y * w + x] = Float(p[base + y * st[2] + x * st[3]]) } }
                return out
            }
            #endif
            for y in 0..<h { for x in 0..<w { out[y * w + x] = a[[0, index, y, x] as [NSNumber]].floatValue } }
        }
        return out
    }

    /// Bilinear upsample of a side x side logit grid (pixel-centre aligned, i.e. align_corners = false) to W x H,
    /// thresholded: logit > threshold -> 1.
    public static func upsampleThreshold(_ logits: [Float], side: Int, width W: Int, height H: Int, threshold: Float) -> BitMask {
        var mask = BitMask(width: W, height: H)
        guard W > 0, H > 0, side > 0, logits.count >= side * side else { return mask }
        let sxs = Double(side) / Double(W), sys = Double(side) / Double(H)
        // per-column source indices/weights once
        var x0s = [Int](repeating: 0, count: W), x1s = [Int](repeating: 0, count: W), wxs = [Float](repeating: 0, count: W)
        for x in 0..<W {
            let fx = min(Double(side - 1), max(0, (Double(x) + 0.5) * sxs - 0.5))
            let i = Int(fx)
            x0s[x] = i; x1s[x] = min(side - 1, i + 1); wxs[x] = Float(fx - Double(i))
        }
        logits.withUnsafeBufferPointer { L in
            mask.data.withUnsafeMutableBufferPointer { m in
                for y in 0..<H {
                    let fy = min(Double(side - 1), max(0, (Double(y) + 0.5) * sys - 0.5))
                    let j = Int(fy), j1 = min(side - 1, j + 1)
                    let wy = Float(fy - Double(j))
                    let r0 = j * side, r1 = j1 * side
                    for x in 0..<W {
                        let a = L[r0 + x0s[x]], b = L[r0 + x1s[x]], c = L[r1 + x0s[x]], d = L[r1 + x1s[x]]
                        let wx = wxs[x]
                        let v = (a + (b - a) * wx) * (1 - wy) + (c + (d - c) * wx) * wy
                        m[y * W + x] = v > threshold ? 1 : 0
                    }
                }
            }
        }
        return mask
    }

    /// side x side 32BGRA pixel buffer with the image drawn plain-scaled into it (row 0 = top of the image).
    private static func pixelBuffer(from image: CGImage, side: Int) throws -> CVPixelBuffer {
        var pbOut: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true]
        guard CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pbOut) == kCVReturnSuccess,
              let pb = pbOut else { throw SAM2Error.pixelBufferCreationFailed }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw SAM2Error.pixelBufferCreationFailed }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return pb
    }
}
