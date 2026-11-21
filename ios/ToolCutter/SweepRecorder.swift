import Foundation
import ARKit
import CoreImage
import Combine
import simd
import AVFoundation

/// Per-frame lens model the server uses to undistort TrueDepth frames. The rear LiDAR path never fills it (ARKit
/// frames are already rectilinear), but the wire format keeps the field so old captures still replay.
struct LensCalibration {
    var inverseLookup: [Float]
    var center: [Double]
    var reference: [Double]
    init(_ calibration: AVCameraCalibrationData) {
        inverseLookup = calibration.inverseLensDistortionLookupTable?.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } ?? []
        center = [Double(calibration.lensDistortionCenter.x), Double(calibration.lensDistortionCenter.y)]
        reference = [Double(calibration.intrinsicMatrixReferenceDimensions.width), Double(calibration.intrinsicMatrixReferenceDimensions.height)]
    }
    var manifest: [String: Any] {
        ["inverse_lookup": inverseLookup, "center": center, "reference": reference]
    }
}

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
    /// Live top-down coverage/height preview, fed with every recorded LiDAR frame (see LiveMap.swift).
    let liveMap = LiveHeightMap()
    @Published var liveImage: CGImage?
    @Published var liveCells = 0
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
        liveMap.reset()
        liveImage = nil
        liveCells = 0
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
            self.feedLiveMap(frame: frame)
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

    /// Fold one ARKit frame into the live preview map. Called from the recorder's own timer AND from
    /// `CaptureController.onFrame` at up to 10 Hz, so the map fills in continuously rather than in 0.5 s blocks.
    /// Must be called on the frame's thread (the depth buffer is read synchronously); the fusion itself is queued.
    func feedLiveMap(frame: ARFrame) {
        guard let depth = frame.sceneDepth, let live = Self.depthArray(depth.depthMap) else { return }
        let k = frame.camera.intrinsics, t = frame.camera.transform
        let pixelBuffer = frame.capturedImage
        // depth intrinsics: ARKit reports them for the colour image; the depth map is the same field of view
        // at a lower resolution, so scale by the size ratio
        let sx = Float(live.w) / Float(CVPixelBufferGetWidth(pixelBuffer)), sy = Float(live.h) / Float(CVPixelBufferGetHeight(pixelBuffer))
        let token = generation
        queue.async {
            let pts = LiveHeightMap.worldPoints(depth: live.depth, w: live.w, h: live.h,
                                                fx: k.columns.0.x * sx, fy: k.columns.1.y * sy, cx: k.columns.2.x * sx, cy: k.columns.2.y * sy,
                                                transform: t, stride: 2)
            self.liveMap.add(points: pts)
            let pic = self.liveMap.rgba()
            let img = pic.flatMap { LiveMapImage.make($0) }
            let cells = self.liveMap.cellsCovered
            DispatchQueue.main.async { if self.generation == token { self.liveImage = img; self.liveCells = cells } }
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

    /// Depth map as a plain Float array (metres), for the live map. Must be read on the frame's thread.
    static func depthArray(_ depth: CVPixelBuffer) -> (depth: [Float], w: Int, h: Int)? {
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(depth)
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float.self)
            for x in 0..<w { out[y * w + x] = row[x] }
        }
        return (out, w, h)
    }

    /// Depth as uint16 millimetres (0 = no return): half the bytes of float32 and it compresses far better. The
    /// server reads it when the manifest says `depth_dtype: "u2mm"`. 1 mm quantisation is well under the sensor's
    /// ~2 mm edge response; the sweep's upload was 138 MB of float32 for one drawer (2026-10-02).
    static func depthPayloadU16(_ depth: CVPixelBuffer) -> DepthPayload? {
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let rb = CVPixelBufferGetBytesPerRow(depth)
        var out = [UInt16](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = base.advanced(by: y * rb).assumingMemoryBound(to: Float.self)
            for x in 0..<w {
                let v = row[x]
                out[y * w + x] = (v.isFinite && v > 0 && v < 65.0) ? UInt16((v * 1000).rounded()) : 0
            }
        }
        return DepthPayload(data: out.withUnsafeBytes { Data($0) }, width: w, height: h, dtype: "u2mm")
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
