// ToolCutter photogrammetry worker: frames (+ depth + gravity) -> textured mesh via RealityKit Object Capture.
//
//   photogrammetry <input-dir> <output.obj|.usdz> [--detail preview|reduced|medium|full|raw] [--mask]
//
// <input-dir> either holds plain images (HEIC/JPG; RealityKit reads embedded depth/gravity if present)
// or a `manifest.json` describing frames captured by the ToolCutter iOS app:
//   { "frames": [ { "image": "frame_000.jpg", "depth": "depth_000.f32", "depth_width": 256, "depth_height": 192,
//                   "gravity": [gx, gy, gz] }, ... ] }
// Depth is float32 little-endian metres, row-major, same field of view as the image. Gravity is in
// camera coordinates (ARKit convention). Object masking is OFF by default so the drawer floor is kept.
import Foundation
import CoreMotion
import RealityKit
import ModelIO
import CoreImage
import CoreVideo
import ImageIO

struct Manifest: Decodable {
    struct Frame: Decodable {
        var image: String
        var depth: String?
        var depth_width: Int?
        var depth_height: Int?
        var gravity: [Float]?
    }
    var frames: [Frame]
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func loadPixelBuffer(_ url: URL) throws -> CVPixelBuffer {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
        throw NSError(domain: "photogrammetry", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot read \(url.lastPathComponent)"])
    }
    let w = cg.width, h = cg.height
    var pb: CVPixelBuffer?
    let attrs: [CFString: Any] = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true]
    CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    guard let buf = pb else { throw NSError(domain: "photogrammetry", code: 3) }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buf), width: w, height: h, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(buf), space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return buf
}

func loadDepth(_ url: URL, width: Int, height: Int) throws -> CVPixelBuffer {
    let data = try Data(contentsOf: url)
    guard data.count == width * height * 4 else {
        throw NSError(domain: "photogrammetry", code: 4, userInfo: [NSLocalizedDescriptionKey: "depth size mismatch for \(url.lastPathComponent)"])
    }
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_DepthFloat32, nil, &pb)
    guard let buf = pb else { throw NSError(domain: "photogrammetry", code: 5) }
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let rowBytes = CVPixelBufferGetBytesPerRow(buf)
    let base = CVPixelBufferGetBaseAddress(buf)!
    data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        for row in 0..<height {
            memcpy(base.advanced(by: row * rowBytes), src.baseAddress!.advanced(by: row * width * 4), width * 4)
        }
    }
    return buf
}

var args = Array(CommandLine.arguments.dropFirst())
guard args.count >= 2 else { fail("usage: photogrammetry <input-dir> <output.obj|.usdz> [--detail level] [--mask]") }
let inputDir = URL(fileURLWithPath: args.removeFirst())
let outputURL = URL(fileURLWithPath: args.removeFirst())
var detail: PhotogrammetrySession.Request.Detail = .medium
var masking = false
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--detail":
        let v = args.isEmpty ? "medium" : args.removeFirst()
        switch v {
        case "preview": detail = .preview
        case "reduced": detail = .reduced
        case "medium": detail = .medium
        case "full": detail = .full
        case "raw": detail = .raw
        default: fail("unknown detail \(v)")
        }
    case "--mask": masking = true
    default: fail("unknown argument \(a)")
    }
}

guard PhotogrammetrySession.isSupported else { fail("PhotogrammetrySession is not supported on this Mac") }

var config = PhotogrammetrySession.Configuration()
config.isObjectMaskingEnabled = masking
config.sampleOrdering = .sequential
config.featureSensitivity = .normal

let session: PhotogrammetrySession
let manifestURL = inputDir.appendingPathComponent("manifest.json")
if FileManager.default.fileExists(atPath: manifestURL.path) {
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
    var samples: [PhotogrammetrySample] = []
    for (i, f) in manifest.frames.enumerated() {
        do {
            var s = PhotogrammetrySample(id: i, image: try loadPixelBuffer(inputDir.appendingPathComponent(f.image)))
            if let d = f.depth, let dw = f.depth_width, let dh = f.depth_height {
                s.depthDataMap = try loadDepth(inputDir.appendingPathComponent(d), width: dw, height: dh)
            }
            if let g = f.gravity, g.count == 3 {
                s.gravity = CMAcceleration(x: Double(g[0]), y: Double(g[1]), z: Double(g[2]))
            }
            samples.append(s)
        } catch {
            FileHandle.standardError.write("skipping frame \(i): \(error.localizedDescription)\n".data(using: .utf8)!)
        }
    }
    guard samples.count >= 3 else { fail("need at least 3 usable frames, got \(samples.count)") }
    FileHandle.standardError.write("loaded \(samples.count) frames with depth=\(samples.contains { $0.depthDataMap != nil })\n".data(using: .utf8)!)
    session = try PhotogrammetrySession(input: samples, configuration: config)
} else {
    session = try PhotogrammetrySession(input: inputDir, configuration: config)
}

let group = DispatchGroup()
group.enter()
var exitCode: Int32 = 0
Task {
    do {
        for try await output in session.outputs {
            switch output {
            case .processingComplete:
                FileHandle.standardError.write("processing complete\n".data(using: .utf8)!)
                group.leave()
                return
            case .requestError(_, let error):
                FileHandle.standardError.write("request error: \(error)\n".data(using: .utf8)!)
                exitCode = 2
                group.leave()
                return
            case .requestComplete(_, let result):
                if case .modelFile(let url) = result {
                    print(url.path)
                }
            case .requestProgress(_, let fraction):
                FileHandle.standardError.write(String(format: "progress %.0f%%\n", fraction * 100).data(using: .utf8)!)
            case .inputComplete:
                FileHandle.standardError.write("input complete\n".data(using: .utf8)!)
            case .invalidSample(let id, let reason):
                FileHandle.standardError.write("invalid sample \(id): \(reason)\n".data(using: .utf8)!)
            case .skippedSample(let id):
                FileHandle.standardError.write("skipped sample \(id)\n".data(using: .utf8)!)
            case .automaticDownsampling:
                FileHandle.standardError.write("automatic downsampling\n".data(using: .utf8)!)
            case .processingCancelled:
                exitCode = 3
                group.leave()
                return
            default:
                break
            }
        }
    } catch {
        FileHandle.standardError.write("session failed: \(error)\n".data(using: .utf8)!)
        exitCode = 4
        group.leave()
    }
}
// PhotogrammetrySession writes USDZ; convert to OBJ (with MTL + textures) via Model I/O when asked.
let wantsOBJ = outputURL.pathExtension.lowercased() == "obj"
let usdzURL = wantsOBJ ? outputURL.deletingPathExtension().appendingPathExtension("usdz") : outputURL
try session.process(requests: [.modelFile(url: usdzURL, detail: detail)])
group.wait()
if exitCode == 0 && wantsOBJ {
    let asset = MDLAsset(url: usdzURL)
    asset.loadTextures()
    do {
        try asset.export(to: outputURL)
        FileHandle.standardError.write("exported OBJ \(outputURL.lastPathComponent)\n".data(using: .utf8)!)
    } catch {
        FileHandle.standardError.write("OBJ export failed: \(error)\n".data(using: .utf8)!)
        exitCode = 5
    }
}
exit(exitCode)
