import AVFoundation
import Combine
import CoreImage
import CoreMotion

/// Object depth capture, without face tracking. All capture state lives on `queue`.
final class TrueDepthController: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    let session = AVCaptureSession()
    @Published var ready = false
    @Published var isRunning = false
    @Published var status = "Starting TrueDepth…"
    @Published var frameCount = 0
    private let queue = DispatchQueue(label: "toolcutter.truedepth", qos: .userInitiated)
    private let video = AVCaptureVideoDataOutput()
    private let depth = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let context = CIContext()
    private let motion = CMMotionManager()
    private var configured = false
    private var recording = false
    private var frames: [SweepFrame] = []
    private var lastTimestamp = -Double.infinity
    private var lastStatusTimestamp = -Double.infinity
    private var steadySince: Double?

    static var isSupported: Bool {
        AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) != nil
    }

    func start() async {
        let granted: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: granted = true
        case .notDetermined: granted = await AVCaptureDevice.requestAccess(for: .video)
        default: granted = false
        }
        guard granted else {
            await MainActor.run { self.status = "Allow camera access in Settings to scan."; self.ready = false }
            return
        }
        await withCheckedContinuation { continuation in
            queue.async {
                do {
                    if !self.configured { try self.configure() }
                    self.motion.deviceMotionUpdateInterval = 1.0 / 30.0
                    self.motion.startDeviceMotionUpdates()
                    self.session.startRunning()
                    DispatchQueue.main.async { self.isRunning = self.session.isRunning }
                } catch {
                    DispatchQueue.main.async { self.status = error.localizedDescription; self.ready = false }
                }
                continuation.resume()
            }
        }
    }

    func stop() async {
        await withCheckedContinuation { continuation in
            queue.async {
                self.recording = false
                self.session.stopRunning()
                self.motion.stopDeviceMotionUpdates()
                DispatchQueue.main.async { self.ready = false; self.isRunning = false }
                continuation.resume()
            }
        }
    }

    func beginRecording() {
        queue.async {
            self.frames.removeAll()
            self.lastTimestamp = -.infinity
            self.steadySince = nil
            self.recording = true
            DispatchQueue.main.async { self.frameCount = 0 }
        }
    }

    /// Runs after all synchronized callbacks, so the final encoded frame isn't lost.
    func finishRecording() async -> [SweepFrame] {
        await withCheckedContinuation { continuation in
            queue.async {
                self.recording = false
                continuation.resume(returning: self.frames)
            }
        }
    }

    private func configure() throws {
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            throw APIError.transport("This phone has no TrueDepth camera. Select the rear camera.")
        }
        session.beginConfiguration()
        defer {
            if !configured {
                for output in session.outputs { session.removeOutput(output) }
                for input in session.inputs { session.removeInput(input) }
            }
            session.commitConfiguration()
        }
        session.sessionPreset = .inputPriority
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw APIError.transport("TrueDepth camera is busy.") }
        session.addInput(input)
        guard session.canAddOutput(video), session.canAddOutput(depth) else {
            session.removeInput(input)
            throw APIError.transport("Synchronized depth capture is unavailable.")
        }
        session.addOutput(video)
        session.addOutput(depth)
        video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        video.alwaysDiscardsLateVideoFrames = true
        depth.isFilteringEnabled = false // no temporal smearing or invented depth at tool edges
        depth.alwaysDiscardsLateDepthData = true
        // Prefer the greatest depth resolution, then RGB detail; only compatible format pairs.
        func frameDuration(_ rgb: AVCaptureDevice.Format, _ depth: AVCaptureDevice.Format) -> CMTime? {
            func ranges(_ f: AVCaptureDevice.Format) -> [CaptureTiming.Range] {
                f.videoSupportedFrameRateRanges.map { .init(minimum: $0.minFrameDuration, maximum: $0.maxFrameDuration) }
            }
            return CaptureTiming.duration(video: ranges(rgb), depth: ranges(depth))
        }
        let pairs = device.formats.flatMap { rgb in
            rgb.supportedDepthDataFormats.filter {
                let type = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return (type == kCVPixelFormatType_DepthFloat16 || type == kCVPixelFormatType_DepthFloat32) && frameDuration(rgb, $0) != nil
            }.map { (rgb, $0) }
        }
        func area(_ format: AVCaptureDevice.Format) -> Int32 {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return d.width * d.height
        }
        guard let pair = pairs.max(by: {
            area($0.1) == area($1.1) ? area($0.0) < area($1.0) : area($0.1) < area($1.1)
        }) else { throw APIError.transport("No metric TrueDepth format is available.") }
        try device.lockForConfiguration()
        do {
            defer { device.unlockForConfiguration() }
            device.activeFormat = pair.0
            device.activeDepthDataFormat = pair.1
            guard let activeDepth = device.activeDepthDataFormat,
                  let duration = frameDuration(device.activeFormat, activeDepth) else {
                throw APIError.transport("This camera's active color and depth formats have no compatible frame timing.")
            }
            // Set a supported EXACT rational duration. Widen the current range first
            // when slowing down, so neither property temporarily crosses the other.
            if CMTimeCompare(duration, device.activeVideoMaxFrameDuration) > 0 {
                device.activeVideoMaxFrameDuration = duration
                device.activeVideoMinFrameDuration = duration
            } else {
                device.activeVideoMinFrameDuration = duration
                device.activeVideoMaxFrameDuration = duration
            }
        }
        // Keep native sensor orientation and disable selfie mirroring for metric geometry.
        for connection in [video.connection(with: .video), depth.connection(with: .depthData)].compactMap({ $0 }) {
            if connection.isVideoRotationAngleSupported(0) { connection.videoRotationAngle = 0 }
            if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [video, depth])
        synchronizer?.setDelegate(self, queue: queue)
        configured = true
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        guard let rgb = collection.synchronizedData(for: video) as? AVCaptureSynchronizedSampleBufferData,
              let dd = collection.synchronizedData(for: depth) as? AVCaptureSynchronizedDepthData,
              !rgb.sampleBufferWasDropped, !dd.depthDataWasDropped,
              let image = CMSampleBufferGetImageBuffer(rgb.sampleBuffer) else { return }
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(rgb.sampleBuffer))
        guard timestamp - lastStatusTimestamp >= 0.2 else { return }
        lastStatusTimestamp = timestamp
        let metric = dd.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        guard let calibration = metric.cameraCalibrationData, metric.depthDataAccuracy == .absolute else {
            publish("Waiting for calibrated metric depth…", ready: false); return
        }
        let quality = SweepRecorder.depthQuality(metric.depthDataMap)
        let speed = motion.deviceMotion.map { sqrt($0.rotationRate.x * $0.rotationRate.x + $0.rotationRate.y * $0.rotationRate.y + $0.rotationRate.z * $0.rotationRate.z) } ?? 0
        guard quality.fraction >= 0.3, quality.distance >= 0.15, quality.distance <= 0.65 else {
            publish("Aim the screen-side camera at the tools, about 20–50 cm away.", ready: false); return
        }
        let acceleration = motion.deviceMotion.map {
            sqrt($0.userAcceleration.x * $0.userAcceleration.x + $0.userAcceleration.y * $0.userAcceleration.y + $0.userAcceleration.z * $0.userAcceleration.z)
        } ?? 0
        guard speed < 0.2, acceleration < 0.06 else {
            steadySince = nil
            publish("Pause briefly at each view to keep tool edges sharp.", ready: false); return
        }
        if steadySince == nil { steadySince = timestamp }
        guard timestamp - (steadySince ?? timestamp) >= 0.25 else {
            publish("Hold steady…", ready: false); return
        }
        guard ProcessInfo.processInfo.thermalState != .critical else {
            publish("Phone is too warm. Stop and let it cool before scanning.", ready: false); return
        }
        publish(frames.count >= 120 ? "120 frames captured. Stop and build." : "TrueDepth ready · overlap each view by at least half", ready: true)
        guard recording, frames.count < 120, timestamp - lastTimestamp >= 1.0 / 3.0 else { return }
        let w = CVPixelBufferGetWidth(image), h = CVPixelBufferGetHeight(image)
        let ref = calibration.intrinsicMatrixReferenceDimensions
        // Scaling is valid for uncropped, native-orientation RGB/depth pairs.
        guard abs(Double(w) / Double(h) - Double(ref.width / ref.height)) < 0.02 else {
            publish("Camera calibration does not match the image format.", ready: false); return
        }
        let k = calibration.intrinsicMatrix
        let sx = Double(w) / Double(ref.width), sy = Double(h) / Double(ref.height)
        let intr = CaptureIntrinsics(fx: Double(k.columns.0.x) * sx, fy: Double(k.columns.1.y) * sy,
                                     cx: Double(k.columns.2.x) * sx, cy: Double(k.columns.2.y) * sy, width: w, height: h)
        guard let jpeg = context.jpegRepresentation(of: CIImage(cvPixelBuffer: image), colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                    options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.95]),
              let payload = SweepRecorder.depthPayload(metric.depthDataMap) else { return }
        let lens = LensCalibration(calibration)
        frames.append(SweepFrame(jpeg: jpeg, depth: payload, intrinsics: intr, gravity: nil, transform: nil,
                                 sensor: "truedepth", use: "depth", timestamp: timestamp, lens: lens))
        lastTimestamp = timestamp
        let count = frames.count
        DispatchQueue.main.async { self.frameCount = count }
    }

    private func publish(_ message: String, ready: Bool) {
        DispatchQueue.main.async { self.status = message; self.ready = ready }
    }
}

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
