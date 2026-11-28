import Foundation
import ARKit
import CoreImage
import UIKit
import Combine
import Vision
import ImageIO

/// Runs the AR session, reports levelness, and grabs a high-resolution still + aligned LiDAR depth.
final class CaptureController: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()
    @Published var tiltDegrees: Double = 0          // 0 = camera pointing straight down
    /// Which WAY the camera is off straight down, in screen axes (+x right, +y down), in degrees.
    /// Its length is `tiltDegrees`; a bullseye level needs the direction, not just the angle.
    @Published var tiltOffset: CGSize = .zero
    private var smoothedTilt = simd_float2(0, 0)
    @Published var trackingOK = false
    @Published var hasDepth = false
    /// Height of the camera above what it is looking at, from the LiDAR depth at the image centre (metres -> mm,
    /// smoothed). With the phone near level this IS the scan height, and the scan height is the one thing that
    /// decided both of Nolan's first two rear scans: 42 cm registered well but the 256x144 LiDAR could not see a
    /// 3 mm rule (~2 mm/px); 18 cm saw the rule but ARKit drifted 30 mm over featureless liner and no frame held
    /// two markers. ~30 cm is the band where both hold.
    @Published var cameraHeightMm: Double = 0
    private var smoothedHeight: Double = 0
    /// Every ARKit frame (<= 10 Hz) while recording, for the live preview map — the recorder's own timer only
    /// takes a frame every 0.5 s, which made the map fill in as slow, coarse blocks.
    var onFrame: ((ARFrame) -> Void)?
    private var lastPreviewAt: TimeInterval = 0
    @Published var statusText = "Starting camera…"
    @Published var isCapturing = false
    private(set) var lastFrame: ARFrame?
    private let ciContext = CIContext()

    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }
    static var supportsDepth: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    func start() {
        guard Self.isSupported else { statusText = "AR not supported on this device"; return }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.isAutoFocusEnabled = true
        if Self.supportsDepth {
            config.frameSemantics.insert(.sceneDepth)
        }
        // highest-resolution video format available (affects the standard frames; stills use captureHighResolutionFrame)
        if let best = ARWorldTrackingConfiguration.supportedVideoFormats.max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = best
        }
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        statusText = Self.supportsDepth ? "Hold the phone flat above the drawer" : "No LiDAR: outlines only, heights typed later"
    }

    func stop() { session.pause() }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        lastFrame = frame
        // How far off straight down, and WHICH WAY, in the axes the preview is drawn with. The view matrix
        // carries the interface rotation; the app is portrait-only. (Maths in LevelMath, tested on the Mac.)
        let vec = LevelMath.tiltVector(view: frame.camera.viewMatrix(for: .portrait))
        // a handheld phone jitters a degree or two; smooth or the bubble is unreadable
        smoothedTilt += (vec - smoothedTilt) * 0.2
        let shown = smoothedTilt
        let ok: Bool
        if case .normal = frame.camera.trackingState { ok = true } else { ok = false }
        var centreM: Double = 0
        if let d = frame.sceneDepth?.depthMap, CVPixelBufferGetPixelFormatType(d) == kCVPixelFormatType_DepthFloat32 {
            CVPixelBufferLockBaseAddress(d, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(d) {
                let w = CVPixelBufferGetWidth(d), h = CVPixelBufferGetHeight(d), rb = CVPixelBufferGetBytesPerRow(d)
                // median of a 5x5 patch at the centre: one bad return must not swing the readout
                var vals: [Float] = []
                for dy in -2...2 { for dx in -2...2 {
                    let v = base.advanced(by: (h / 2 + dy) * rb).assumingMemoryBound(to: Float.self)[w / 2 + dx]
                    if v.isFinite && v > 0.05 { vals.append(v) }
                } }
                if vals.count >= 8 { vals.sort(); centreM = Double(vals[vals.count / 2]) }
            }
            CVPixelBufferUnlockBaseAddress(d, .readOnly)
        }
        if centreM > 0 { smoothedHeight = smoothedHeight == 0 ? centreM : smoothedHeight + (centreM - smoothedHeight) * 0.25 }
        let heightMm = smoothedHeight * 1000
        if let sink = onFrame, frame.timestamp - lastPreviewAt >= 0.1 {
            lastPreviewAt = frame.timestamp
            sink(frame)
        }
        DispatchQueue.main.async {
            self.tiltDegrees = Double(simd_length(shown))
            self.tiltOffset = CGSize(width: Double(shown.x), height: Double(shown.y))
            self.trackingOK = ok
            self.hasDepth = frame.sceneDepth != nil
            self.cameraHeightMm = heightMm
        }
    }

    /// One sharp still for the photo pass: full-resolution frame, no depth, marked colour-only.
    /// Taken on demand so the phone is held still, which is what the continuous glide could not guarantee.
    /// A small JPEG of the live frame (~1200 px wide) for the marker preflight. ArUco needs a marker ~20 px across;
    /// a 25 mm marker at 60 cm is ~43 px at this size, and the round trip stays well under 200 ms.
    func previewJPEG() -> Data? {
        guard let frame = lastFrame else { return nil }
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
        let scale = min(1.0, 1200.0 / ci.extent.width)     // 25 mm markers at 60 cm are ~43 px here; ArUco wants >= 20
        let small = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return ciContext.jpegRepresentation(of: small, colorSpace: CGColorSpaceCreateDeviceRGB(),
                                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.8])
    }

    func stillFrame() async throws -> SweepFrame {
        let c = try await capture(preferHighResolution: true)
        return SweepFrame(jpeg: c.jpeg, depth: nil, intrinsics: c.intrinsics, gravity: nil,
                          transform: c.transform, sensor: "rear_photo", use: "color",
                          timestamp: c.timestamp)
    }

    struct Capture {
        var jpeg: Data
        var intrinsics: CaptureIntrinsics
        var depth: DepthPayload?
        var previewImage: UIImage?
        var transform: simd_float4x4
        var timestamp: Double
    }

    /// Take the still. Uses the high-resolution frame when the device offers it, else the current frame.
    func capture(preferHighResolution: Bool = false) async throws -> Capture {
        await MainActor.run { isCapturing = true }
        defer { Task { @MainActor in self.isCapturing = false } }
        guard let live = lastFrame else { throw APIError.transport("Camera not ready") }
        var frame: ARFrame = live
        // Keep RGB and depth at the same instant. A later high-resolution image
        // cannot be paired with the earlier live depth while the phone moves.
        if preferHighResolution || live.sceneDepth == nil, #available(iOS 16.0, *) {
            if let hi = try? await session.captureHighResolutionFrame() {
                frame = hi
            }
        }
        let pixelBuffer = frame.capturedImage
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let k = frame.camera.intrinsics
        let intr = CaptureIntrinsics(fx: Double(k.columns.0.x), fy: Double(k.columns.1.y),
                                     cx: Double(k.columns.2.x), cy: Double(k.columns.2.y), width: w, height: h)
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        guard let jpeg = ciContext.jpegRepresentation(of: ci, colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                      options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.93]) else {
            throw APIError.transport("Could not encode the photo")
        }
        let preview = UIImage(data: jpeg)
        let depth = frame.sceneDepth.flatMap { SweepRecorder.depthPayload($0.depthMap, confidence: $0.confidenceMap) }
        return Capture(jpeg: jpeg, intrinsics: intr, depth: depth, previewImage: preview,
                       transform: frame.camera.transform, timestamp: frame.timestamp)
    }

    // MARK: - Per-tool still (Capture tools)

    /// One tool photo, UPRIGHT (portrait), with the live frame's LiDAR depth rotated the same way.
    struct StillCapture {
        /// Upright portrait image (the sensor's landscape pixels rotated once, 90° clockwise).
        var image: CGImage
        /// Intrinsics of `image` (already rotated), integer (u, v) = pixel centre.
        var intrinsics: CameraIntrinsics
        /// Depth in metres, row-major, upright like `image`; NaN = no return or confidence below medium.
        var depth: [Float]?
        var depthWidth: Int = 0
        var depthHeight: Int = 0
        /// Intrinsics of the depth map (the LIVE frame's camera scaled to the depth size, then rotated).
        var depthIntrinsics: CameraIntrinsics?
        var transform: simd_float4x4
        var timestamp: Double
    }

    /// Full-resolution still for one tool + the latest live frame's LiDAR depth, both rotated upright.
    ///
    /// ROTATION. ARKit delivers `capturedImage` in the sensor's native landscape orientation (W x H, W > H); this app
    /// is portrait-only (Info: UISupportedInterfaceOrientations_iPhone = Portrait), and for the rear camera held in
    /// portrait the upright picture is the sensor image rotated 90° CLOCKWISE — EXIF/CGImagePropertyOrientation
    /// `.right`, the same orientation UIKit assigns to rear-camera portrait photos. In integer-pixel-centre
    /// coordinates a sensor pixel (u, v) lands at
    ///     u' = (H - 1) - v,   v' = u           (new size W' = H, H' = W)
    /// i.e. the sensor's top row becomes the right-hand column. The camera axes rotate with it: X' = -Y, Y' = X, Z' = Z.
    /// Projection u = fx X/Z + cx, v = fy Y/Z + cy then gives
    ///     u' = H - 1 - (fy Y/Z + cy) = fy X'/Z + (H - 1 - cy)
    ///     v' = fx X/Z + cx            = fx Y'/Z + cx
    /// so  fx' = fy,  fy' = fx,  cx' = H - 1 - cy,  cy' = cx.   (`rotatedUpright` below; the depth map gets the same.)
    /// Both the photo and the depth are expressed in that one rotated camera frame, so a plane fitted to the depth
    /// can be used directly with the photo's intrinsics.
    func captureStillWithDepth() async throws -> StillCapture {
        await MainActor.run { isCapturing = true }
        defer { Task { @MainActor in self.isCapturing = false } }
        guard let live = lastFrame else { throw APIError.transport("Camera not ready") }
        // depth from the LIVE frame, read now (the high-resolution frame carries none)
        var depth: [Float]?
        var dW = 0, dH = 0
        var depthK: CameraIntrinsics?
        if let sd = live.sceneDepth, let dp = SweepRecorder.depthPayload(sd.depthMap, confidence: sd.confidenceMap) {
            let raw: [Float] = dp.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let r = Self.rotateUpright(raw, width: dp.width, height: dp.height)
            depth = r; dW = dp.height; dH = dp.width
            let lk = live.camera.intrinsics
            let liveK = CameraIntrinsics(fx: Double(lk.columns.0.x), fy: Double(lk.columns.1.y), cx: Double(lk.columns.2.x), cy: Double(lk.columns.2.y),
                                         width: CVPixelBufferGetWidth(live.capturedImage), height: CVPixelBufferGetHeight(live.capturedImage))
            depthK = Self.rotatedUpright(liveK.scaled(toWidth: dp.width, height: dp.height))
        }
        var frame: ARFrame = live
        if let hi = try? await session.captureHighResolutionFrame() { frame = hi }
        let pb = frame.capturedImage
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let k = frame.camera.intrinsics
        let sensorK = CameraIntrinsics(fx: Double(k.columns.0.x), fy: Double(k.columns.1.y), cx: Double(k.columns.2.x), cy: Double(k.columns.2.y),
                                       width: w, height: h)
        let upright = CIImage(cvPixelBuffer: pb).oriented(.right)
        guard let cg = ciContext.createCGImage(upright, from: upright.extent) else {
            throw APIError.transport("Could not read the photo")
        }
        return StillCapture(image: cg, intrinsics: Self.rotatedUpright(sensorK), depth: depth, depthWidth: dW, depthHeight: dH,
                            depthIntrinsics: depthK, transform: frame.camera.transform, timestamp: frame.timestamp)
    }

    /// Sensor (landscape) intrinsics -> upright portrait intrinsics (90° clockwise; derivation above).
    static func rotatedUpright(_ k: CameraIntrinsics) -> CameraIntrinsics {
        CameraIntrinsics(fx: k.fy, fy: k.fx, cx: Double(k.height - 1) - k.cy, cy: k.cx, width: k.height, height: k.width)
    }

    /// Row-major w x h grid rotated 90° clockwise -> h x w: out[v' * h + u'] = in[v * w + u] with u' = h-1-v, v' = u.
    static func rotateUpright(_ a: [Float], width w: Int, height h: Int) -> [Float] {
        var out = [Float](repeating: .nan, count: w * h)
        for v in 0..<h { for u in 0..<w { out[u * h + (h - 1 - v)] = a[v * w + u] } }
        return out
    }

    static func photoThumbnail(_ data: Data) -> UIImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 160,
                kCGImageSourceCreateThumbnailWithTransform: true
              ] as CFDictionary) else { return nil }
        return UIImage(cgImage: image)
    }

    /// Image overlap is an estimate, not a claim that every drawer region is covered.
    /// Work on small images off the main thread, using the same orientation for both shots.
    static func photoOverlap(previous: Data, current: Data) -> Double? {
        let context = CIContext()
        func thumbnail(_ data: Data) -> CGImage? {
            guard let image = CIImage(data: data) else { return nil }
            let scale = 640 / max(image.extent.width, image.extent.height)
            let small = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            return context.createCGImage(small, from: small.extent)
        }
        guard let reference = thumbnail(previous), let floating = thumbnail(current) else { return nil }
        let request = VNTranslationalImageRegistrationRequest(targetedCGImage: floating, options: [:])
        do {
            try VNImageRequestHandler(cgImage: reference, options: [:]).perform([request])
            guard let observation = request.results?.first else { return nil }
            // Level, same-orientation shots should differ mainly by translation.
            // An unconstrained homography can invent perspective on repeating
            // drawer texture and badly overstate overlap.
            let transform = observation.alignmentTransform
            guard transform.tx.isFinite, transform.ty.isFinite else { return nil }
            let bounds = CGRect(x: 0, y: 0, width: floating.width, height: floating.height)
            let target = CGRect(x: 0, y: 0, width: reference.width, height: reference.height)
            let intersection = bounds.applying(transform).intersection(target)
            if intersection.isNull { return 0 }
            return min(1, max(0, intersection.width * intersection.height / (bounds.width * bounds.height)))
        } catch { return nil }
    }

}
