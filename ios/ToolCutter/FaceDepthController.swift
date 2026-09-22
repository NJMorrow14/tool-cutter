import Foundation
import ARKit
import CoreImage
import UIKit

/// TrueDepth capture that ALSO carries a world pose.
///
/// The AVFoundation TrueDepth path (`TrueDepthController`) gives depth but no pose, so every frame has to place
/// itself off the corner markers — and at 20-50 cm the view is far too narrow to keep one in shot. On the first
/// two-pass capture 35 of 44 depth frames saw no marker at all and could not be placed, which is how half a
/// drawer went missing. `ARFaceTrackingConfiguration` with `isWorldTrackingEnabled` runs the front camera through
/// ARKit instead: `frame.capturedDepthData` is the same TrueDepth map, and `frame.camera.transform` is a tracked,
/// gravity-aligned world pose. Every frame can then be placed the way the rear LiDAR arc already is.
///
/// Depth arrives slower than video (about 15 Hz against 60), so frames are taken only when a new depth map is
/// actually attached — `capturedDepthData` is nil on most ARFrames and must never be reused from a previous one.
@MainActor
final class FaceDepthController: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()
    @Published var status = "Starting tracked TrueDepth…"
    @Published var ready = false
    @Published var isRunning = false
    @Published var frameCount = 0
    /// What the device actually gave us, so a first run says plainly whether this path works here.
    @Published var capability = "checking…"
    /// Live tally of what the session is actually delivering. Without this a failure is just "no frames".
    @Published var diag = "no ARFrames yet"
    private var seen = 0            // ARFrames delivered
    private var withDepth = 0       // ...that carried a depth map
    private var rejected: [String: Int] = [:]
    private var accuracySeen = "?"
    private var lastCover = 0.0      // share of the depth map that has a value
    private var lastNear = 0.0       // 10th-percentile depth: how far the NEAREST surface is
    private var lastInRange = 0.0    // share of the map actually at working distance
    /// World tracking has to establish itself against the ROOM before it can survive being pointed at a close
    /// flat drawer. Until it reports .normal once, there is no world frame and the poses are meaningless.
    @Published var trackingReady = false
    private var posed = 0                               // frames captured WITH a world pose
    private var worldTracking = true
    private var triedWithoutWorldTracking = false

    private var frames: [SweepFrame] = []
    private var recording = false
    private var lastTimestamp: Double = 0
    private var lastStatusTimestamp: Double = 0
    private var lastDepthTimestamp: Double = -1
    private let context = CIContext()
    private let maxFrames = 240

    static let nearM = 0.12, farM = 0.90

    /// Coverage, the distance to the NEAREST surface (10th percentile, robust to a few stray close samples),
    /// and how much of the map sits at working distance.
    static func depthStats(_ depth: CVPixelBuffer) -> (cover: Double, near: Double, inRange: Double) {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return (0, 0, 0) }
        var values: [Float] = []
        var samples = 0, inRange = 0
        for y in stride(from: 0, to: CVPixelBufferGetHeight(depth), by: 4) {
            let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(depth)).assumingMemoryBound(to: Float.self)
            for x in stride(from: 0, to: CVPixelBufferGetWidth(depth), by: 4) {
                samples += 1
                let v = row[x]
                guard v.isFinite, v > 0 else { continue }
                values.append(v)
                if Double(v) >= nearM, Double(v) <= farM { inRange += 1 }
            }
        }
        guard !values.isEmpty else { return (0, 0, 0) }
        values.sort()
        return (Double(values.count) / Double(max(1, samples)),
                Double(values[values.count / 10]),
                Double(inRange) / Double(max(1, samples)))
    }

    static var isSupported: Bool { ARFaceTrackingConfiguration.isSupported }
    /// iOS 13+ on A12 and later. Without this the front camera cannot be world-tracked and there is no pose.
    static var supportsWorldTracking: Bool {
        if #available(iOS 13.0, *) { return ARFaceTrackingConfiguration.supportsWorldTracking }
        return false
    }

    func start() {
        guard Self.isSupported else {
            status = "This device has no TrueDepth camera."
            capability = "face tracking unsupported"
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.worldAlignment = .gravity
        // Leave face tracking ON (1 = the default). Setting this to 0 asks ARKit to track nothing, and the
        // TrueDepth stream — the whole reason we are here — may simply never start.
        config.maximumNumberOfTrackedFaces = 1
        if #available(iOS 13.0, *), ARFaceTrackingConfiguration.supportsWorldTracking, worldTracking {
            config.isWorldTrackingEnabled = true
        }
        capability = !Self.supportsWorldTracking ? "NO world tracking on this device"
            : worldTracking ? "world tracking ON (poses)" : "world tracking OFF (depth only — frames need a marker)"
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        trackingReady = false
        status = "Point the screen-side camera at the drawer, 20–50 cm away."
    }

    func stop() {
        session.pause()
        isRunning = false
    }

    func beginRecording() {
        frames.removeAll()
        frameCount = 0
        recording = true
        lastTimestamp = 0
        posed = 0
    }

    func finishRecording() -> [SweepFrame] {
        recording = false
        // No frame got a pose => this was a plain TrueDepth sweep, whatever the configuration asked for. Label it
        // honestly: the server has a depth-only registration path (register_rgbd) that it only uses when every
        // frame is "truedepth" with no poses, and that is a better fit than pretending a tracked scan failed.
        if !frames.contains(where: { $0.transform != nil }) {
            for i in frames.indices { frames[i].sensor = "truedepth" }
        }
        return frames
    }

    // MARK: - ARSessionDelegate

    /// If ARKit refuses the configuration outright there are no frames at all and nothing else reports it.
    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        let text = error.localizedDescription
        Task { @MainActor [weak self] in
            self?.publish("ARKit refused the session: \(text)", false)
            self?.diag = "SESSION FAILED · \(text)"
            self?.isRunning = false
        }
    }

    nonisolated func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor [weak self] in self?.diag = "session interrupted" }
    }

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // ARFrame is not Sendable and is only valid for this call: pull everything out here.
        guard let depthData = frame.capturedDepthData else {
            Task { @MainActor [weak self] in self?.tick(hadDepth: false) }
            return
        }
        let depthStamp = frame.capturedDepthDataTimestamp
        let stamp = frame.timestamp
        let transform = frame.camera.transform
        let k = frame.camera.intrinsics
        let imageSize = frame.camera.imageResolution
        let pixelBuffer = frame.capturedImage
        let tracking = frame.camera.trackingState
        let metric = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let calibration = metric.cameraCalibrationData
        let accurate = metric.depthDataAccuracy == .absolute
        let depthMap = metric.depthDataMap
        let accuracy = metric.depthDataAccuracy == .absolute ? "absolute" : "relative"
        Task { @MainActor [weak self] in
            self?.tick(hadDepth: true, accuracy: accuracy)
            self?.ingest(depthStamp: depthStamp, stamp: stamp, transform: transform, k: k, imageSize: imageSize,
                         pixelBuffer: pixelBuffer, tracking: tracking, depthMap: depthMap,
                         calibration: calibration, accurate: accurate)
        }
    }

    private func ingest(depthStamp: Double, stamp: Double, transform: simd_float4x4, k: simd_float3x3,
                        imageSize: CGSize, pixelBuffer: CVPixelBuffer, tracking: ARCamera.TrackingState,
                        depthMap: CVPixelBuffer, calibration: AVCameraCalibrationData?, accurate: Bool) {
        guard depthStamp != lastDepthTimestamp else { return }   // the same depth map attached to several frames
        lastDepthTimestamp = depthStamp
        guard stamp - lastStatusTimestamp >= 0.2 || recording else { return }
        lastStatusTimestamp = stamp

        if !accurate { note("relative-accuracy depth (using it anyway)") }
        // ARKit already decides whether motion, light or featurelessness is too much and drops out of .normal,
        // and it is far better calibrated than a hand-picked gyro threshold. Use its verdict, and say which.
        // A POSE IS A BONUS, NOT A REQUIREMENT. Front-camera world tracking may never initialise at all when the
        // camera is 30 cm from a dark, flat drawer liner — there is nothing there to build a world frame from.
        // Blocking the scan on it made the app unusable; the server already places pose-less frames by matching
        // them to a placed neighbour. So capture regardless and attach a pose only when ARKit is actually tracking.
        var pose: simd_float4x4?
        var poseQuality = "none"
        if worldTracking {
            switch tracking {
            case .normal:
                pose = transform; poseQuality = "normal"; trackingReady = true
            case .limited(.insufficientFeatures) where trackingReady:
                // Expected once you are down over the drawer: the world frame already exists and the IMU carries it.
                pose = transform; poseQuality = "limited"
            default:
                break                                  // no usable world frame yet — send the frame without one
            }
        }
        // Judge on HOW MUCH of the view is at working distance, not on the median depth. Holding the phone
        // 30 cm over a drawer, anything past the drawer's edge is room — metres away — and it drags the median
        // over the limit, which is why this said "too far" with the drawer plainly in view.
        let st = Self.depthStats(depthMap)
        lastCover = st.cover; lastNear = st.near; lastInRange = st.inRange
        guard st.cover >= 0.08 else {
            note("almost no depth returned"); publish("No depth — check nothing is covering the camera.", false); return
        }
        guard st.inRange >= 0.06 else {
            note(st.near > Self.farM ? "too far" : "too close")
            publish(String(format: "Nearest surface %.0f cm — hold 20–50 cm over the drawer.", st.near * 100), false); return
        }
        guard ProcessInfo.processInfo.thermalState != .critical else {
            publish("Phone is too warm. Stop and let it cool before scanning.", false); return
        }
        publish(frames.count >= maxFrames ? "\(maxFrames) frames captured. Stop and build."
                : pose != nil ? "Tracked · sweep the drawer steadily, overlapping as you go"
                : "Sweep the drawer steadily — OVERLAP EACH VIEW BY HALF, that is what lines the frames up", true)
        guard recording, frames.count < maxFrames, stamp - lastTimestamp >= 1.0 / 6.0 else { return }

        let w = CVPixelBufferGetWidth(pixelBuffer), h = CVPixelBufferGetHeight(pixelBuffer)
        // ARKit reports intrinsics against camera.imageResolution; scale them if the buffer differs.
        let sx = Double(w) / Double(imageSize.width), sy = Double(h) / Double(imageSize.height)
        let intr = CaptureIntrinsics(fx: Double(k.columns.0.x) * sx, fy: Double(k.columns.1.y) * sy,
                                     cx: Double(k.columns.2.x) * sx, cy: Double(k.columns.2.y) * sy, width: w, height: h)
        guard let jpeg = context.jpegRepresentation(of: CIImage(cvPixelBuffer: pixelBuffer), colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]),
              let payload = SweepRecorder.depthPayload(depthMap) else { return }
        // With world tracking off, camera.transform is NOT a world pose (a face config without a face in view
        // reports little more than gravity) — send nothing and let the server match the frame to a neighbour.
        frames.append(SweepFrame(jpeg: jpeg, depth: payload, intrinsics: intr, gravity: nil, transform: pose,
                                 sensor: "truedepth_tracked", use: "depth", poseQuality: poseQuality, timestamp: stamp,
                                 lens: calibration.map(LensCalibration.init)))
        lastTimestamp = stamp
        frameCount = frames.count
        if pose != nil { posed += 1 }          // count what was KEPT, not what the preview saw
    }

    private func publish(_ message: String, _ ok: Bool) {
        status = message
        ready = ok
    }

    /// Count a rejection without spamming; the reason is surfaced in `diag`.
    private func note(_ reason: String) { rejected[reason, default: 0] += 1 }

    private func tick(hadDepth: Bool, accuracy: String = "?") {
        seen += 1
        if hadDepth { withDepth += 1; accuracySeen = accuracy }
        // If world tracking costs us the depth stream entirely, take depth over poses rather than capture
        // nothing: this session then behaves like the plain TrueDepth path and frames need a marker in view.
        if worldTracking, !triedWithoutWorldTracking, withDepth == 0, seen >= 90 {
            triedWithoutWorldTracking = true
            worldTracking = false
            diag = "no depth with world tracking after \(seen) frames — retrying without it"
            seen = 0
            stop()
            start()
            return
        }
        guard seen % 10 == 0 else { return }
        let worst = rejected.max(by: { $0.value < $1.value })
        diag = String(format: "frames %d · depth %d · cover %.0f%% · near %.2f m · in-range %.0f%% · %@",
                      seen, withDepth, lastCover * 100, lastNear, lastInRange * 100,
                      !worldTracking ? "pose OFF" : trackingReady ? "pose OK (\(posed) kept)" : "no pose yet")
            + (withDepth == 0 ? " · NO DEPTH FROM ARKIT" : "")
            + (worst.map { " · mostly: \($0.key)" } ?? "")
    }
}
