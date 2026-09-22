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
        DispatchQueue.main.async {
            self.tiltDegrees = Double(simd_length(shown))
            self.tiltOffset = CGSize(width: Double(shown.x), height: Double(shown.y))
            self.trackingOK = ok
            self.hasDepth = frame.sceneDepth != nil
        }
    }

    /// One sharp still for the photo pass: full-resolution frame, no depth, marked colour-only.
    /// Taken on demand so the phone is held still, which is what the continuous glide could not guarantee.
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
