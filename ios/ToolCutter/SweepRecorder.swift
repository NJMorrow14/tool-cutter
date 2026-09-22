import Foundation
import ARKit
import CoreImage
import Combine
import simd

struct SweepFrame {
    var jpeg: Data
    var depth: DepthPayload?
    var intrinsics: CaptureIntrinsics
    var gravity: SIMD3<Float>?
    var transform: simd_float4x4? // absent for AVFoundation TrueDepth: never invent a pose
    var sensor: String = "rear_lidar"
    /// What this frame is FOR: "depth" (topography only — its photo is never painted into the mosaic),
    /// "color" (a sharp still, no depth), or "both". Two-pass capture sweeps with the front TrueDepth camera
    /// and then shoots stills with the rear one; each pass registers off the markers, so they share a grid.
    var use: String = "both"
    /// "normal" or "limited": how much the world pose on this frame can be trusted.
    var poseQuality: String = "normal"
    var timestamp: Double = 0
    var lens: LensCalibration? = nil
}

/// Main-thread recorder with one encoder job in flight and an explicit drain on Stop.
final class SweepRecorder: ObservableObject {
    private(set) var frames: [SweepFrame] = []
    private var timer: Timer?
    private let ciContext = CIContext()
    private let queue = DispatchQueue(label: "toolcutter.sweep", qos: .userInitiated)
    private var generation = UUID()
    private var encoding = false
    private var previous: (Double, simd_float4x4)?
    @Published var status = "Move slowly and keep the markers in view."
    var onCount: ((Int) -> Void)?

    func start(session: ARSession, interval: TimeInterval = 0.5) {
        cancel()
        frames.removeAll()
        previous = nil
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let self, !self.encoding, let frame = session.currentFrame else { return }
            guard self.frames.count < 120 else { self.status = "120 frames captured. Stop and build."; return }
            if case .normal = frame.camera.trackingState {} else { self.status = "Pause briefly while tracking recovers."; return }
            let t = frame.camera.transform
            defer { self.previous = (frame.timestamp, t) }
            if let (time, old) = self.previous {
                let dt = Float(frame.timestamp - time)
                guard dt > 0 else { return }
                let speed = simd_distance(simd_make_float3(t.columns.3), simd_make_float3(old.columns.3)) / dt
                let angle = acos(min(1, max(-1, simd_dot(simd_make_float3(t.columns.2), simd_make_float3(old.columns.2))))) / dt
                guard speed < 0.25, angle < 0.45 else { self.status = "Slow down for sharper, overlapping frames."; return }
            }
            let pixelBuffer = frame.capturedImage
            let k = frame.camera.intrinsics
            let intr = CaptureIntrinsics(fx: Double(k.columns.0.x), fy: Double(k.columns.1.y), cx: Double(k.columns.2.x), cy: Double(k.columns.2.y),
                                         width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
            let r = simd_float3x3(simd_make_float3(t.columns.0), simd_make_float3(t.columns.1), simd_make_float3(t.columns.2))
            let g = r.transpose * SIMD3<Float>(0, -1, 0)
            let depth = frame.sceneDepth
            let token = self.generation
            self.encoding = true
            self.status = "Capturing · keep a slow, steady glide."
            self.queue.async {
                let dp = depth.flatMap { Self.depthPayload($0.depthMap, confidence: $0.confidenceMap) }
                let jpeg = self.ciContext.jpegRepresentation(of: CIImage(cvPixelBuffer: pixelBuffer), colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95])
                let f = jpeg.map { SweepFrame(jpeg: $0, depth: dp, intrinsics: intr, gravity: g, transform: t,
                                             sensor: depth == nil ? "rear_rgb" : "rear_lidar", timestamp: frame.timestamp) }
                DispatchQueue.main.async {
                    guard self.generation == token else { return }
                    self.encoding = false
                    if let f { self.frames.append(f); self.onCount?(self.frames.count) }
                }
            }
        }
    }

    func stop() async -> [SweepFrame] {
        timer?.invalidate()
        timer = nil
        return await withCheckedContinuation { continuation in
            queue.async {
                // The encoder's main-queue append is already enqueued before this callback.
                DispatchQueue.main.async { continuation.resume(returning: self.frames) }
            }
        }
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        generation = UUID()
        encoding = false
    }

    static func depthQuality(_ depth: CVPixelBuffer) -> (fraction: Double, distance: Float) {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return (0, 0) }
        var values: [Float] = []
        var samples = 0
        for y in stride(from: 0, to: CVPixelBufferGetHeight(depth), by: 4) {
            let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(depth)).assumingMemoryBound(to: Float.self)
            for x in stride(from: 0, to: CVPixelBufferGetWidth(depth), by: 4) {
                samples += 1
                if row[x].isFinite && row[x] > 0 { values.append(row[x]) }
            }
        }
        values.sort()
        return (Double(values.count) / Double(max(1, samples)), values.isEmpty ? 0 : values[values.count / 2])
    }

    static func depthPayload(_ depth: CVPixelBuffer, confidence: CVPixelBuffer? = nil) -> DepthPayload? {
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let conf = confidence.flatMap { CVPixelBufferGetWidth($0) == w && CVPixelBufferGetHeight($0) == h ? $0 : nil }
        if let conf { CVPixelBufferLockBaseAddress(conf, .readOnly) }
        defer { if let conf { CVPixelBufferUnlockBaseAddress(conf, .readOnly) } }
        var values = [Float](repeating: .nan, count: w * h)
        for y in 0..<h {
            let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(depth)).assumingMemoryBound(to: Float.self)
            let crow = conf.flatMap { CVPixelBufferGetBaseAddress($0)?.advanced(by: y * CVPixelBufferGetBytesPerRow($0)).assumingMemoryBound(to: UInt8.self) }
            for x in 0..<w where row[x].isFinite && row[x] > 0 {
                if crow == nil || crow![x] >= ARConfidenceLevel.medium.rawValue { values[y * w + x] = row[x] }
            }
        }
        return DepthPayload(data: values.withUnsafeBytes { Data($0) }, width: w, height: h)
    }
}
