import SwiftUI
import ARKit
import SceneKit
import AVFoundation
import AudioToolbox

private final class DepthPreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

private struct TrueDepthPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> DepthPreviewView {
        let view = DepthPreviewView()
        view.preview.session = session
        view.preview.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ view: DepthPreviewView, context: Context) {
        if let connection = view.preview.connection {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
    }
}

struct ARPreview: UIViewRepresentable {
    let session: ARSession
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        v.session = session
        v.automaticallyUpdatesLighting = false
        v.scene = SCNScene()
        return v
    }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

/// Arc capture: start, glide the phone along the drawer, stop. Frames (photo + LiDAR depth + pose)
/// are fused on the Mac into a to-scale drawer map; the review UI opens on the result.
struct CaptureView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var store: SessionStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var controller = CaptureController()
    @StateObject private var trueDepth = TrueDepthController()
    @StateObject private var faceDepth = FaceDepthController()
    @StateObject private var recorder = SweepRecorder()
    @AppStorage("captureSensor") private var captureSensor = "truedepth"
    /// Two-pass capture: sweep for topography with the front TrueDepth camera, then shoot sharp stills with
    /// the rear one, each triggered by hand. Both passes register off the markers, so they share a grid.
    enum Pass { case setup, depth, photos, send }
    @State private var pass: Pass = .setup
    @State private var depthFrames: [SweepFrame] = []
    @State private var photoFrames: [SweepFrame] = []
    @State private var shooting = false
    @State private var photoFeedback: String?
    @State private var countdown: Int?          // nil = not counting; 3, 2, 1 while it runs
    @State private var countdownRun = 0         // cancel + restart must not leave the old counter running
    @State private var drawerName = ""
    @State private var drawerWidth = ""
    @State private var drawerDepth = ""
    @State private var uploading = false
    @State private var error: String?
    @State private var reviewSession: String?
    @State private var recording = false
    @State private var frameCount = 0
    @State private var startedAt: Date?
    @State private var statusMessage: String?
    private let minFrames = 5
    /// Which sensor runs the depth pass. "tracked" is the one to use where it exists: the same TrueDepth map,
    /// but through ARKit, so every frame carries a world pose and can be placed without seeing a marker.
    private enum DepthSource { case tracked, plainTrueDepth, rear }
    private var depthSource: DepthSource {
        if captureSensor == "truedepth_tracked" && FaceDepthController.isSupported { return .tracked }
        if captureSensor == "truedepth" && TrueDepthController.isSupported { return .plainTrueDepth }
        return .rear
    }
    private var useFaceDepth: Bool { pass == .depth && depthSource == .tracked }
    private var useTrueDepth: Bool { pass == .depth && depthSource == .plainTrueDepth }
    private var frontPass: Bool { useFaceDepth || useTrueDepth }
    private var count: Int { useFaceDepth ? faceDepth.frameCount : useTrueDepth ? trueDepth.frameCount : frameCount }

    var body: some View {
        ZStack {
            if pass == .setup {
                // no preview on the form: running a camera here only burns battery and warms the phone
                // before the scan that actually needs a cool sensor
                LinearGradient(colors: [.black, Color(white: 0.12)], startPoint: .top, endPoint: .bottom).ignoresSafeArea()
            } else if useFaceDepth {
                ARPreview(session: faceDepth.session).ignoresSafeArea()
            } else if useTrueDepth {
                TrueDepthPreview(session: trueDepth.session).ignoresSafeArea()
            } else if CaptureController.isSupported {
                ARPreview(session: controller.session).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("AR camera is not available here (simulator?).").foregroundStyle(.white).padding()
            }
            VStack {
                statusBar
                Spacer()
                if !frontPass && (pass == .depth || pass == .photos) {
                    LevelGauge(tilt: controller.tiltDegrees, offset: controller.tiltOffset)
                }
                Spacer()
                bottomBar
            }
            .padding()
            if let c = countdown {
                Color.black.opacity(0.35).ignoresSafeArea()
                VStack(spacing: 6) {
                    Text("\(c)").font(.system(size: 140, weight: .bold, design: .rounded)).foregroundStyle(.white)
                        .contentTransition(.numericText())
                    Text("get the phone over the drawer").font(.headline).foregroundStyle(.white.opacity(0.9))
                }
                .transaction { $0.animation = .easeOut(duration: 0.2) }
            }
            if uploading {
                Color.black.opacity(0.55).ignoresSafeArea()
                VStack(spacing: 12) {
                    ProgressView().tint(.white)
                    Text(statusMessage ?? "Fusing frames on the Mac…").foregroundStyle(.white).multilineTextAlignment(.center)
                }.padding()
            }
        }
        .task(id: captureSensor + String(describing: pass)) {
            controller.stop()
            faceDepth.stop()
            await trueDepth.stop()
            guard !Task.isCancelled else { return }
            guard pass != .setup else { return }      // nothing to run behind the form
            if useFaceDepth { faceDepth.start() }
            else if useTrueDepth { await trueDepth.start() }
            else { controller.start() }
        }
        .onDisappear { controller.stop(); faceDepth.stop(); recorder.cancel(); Task { await trueDepth.stop() } }
        .onChange(of: count) { _, newCount in
            if recording && newCount > 0 && newCount % 5 == 0 {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            }
        }
        .alert("Capture failed", isPresented: Binding(get: { error != nil }, set: { _ in error = nil })) {
            Button("OK", role: .cancel) {}
        } message: { Text(error ?? "") }
        .navigationDestination(item: $reviewSession) { sid in
            ReviewWebView(url: settings.reviewURL(sessionId: sid), title: drawerName.isEmpty ? "Drawer" : drawerName)
        }
    }

    private var stepTitle: String {
        switch pass {
        case .setup:  return "1 · Drawer"
        case .depth:  return useFaceDepth ? "2 · Depth (front, tracked)" : useTrueDepth ? "2 · Depth (front)" : "2 · Depth (rear)"
        case .photos: return "3 · Photos"
        case .send:   return "4 · Send"
        }
    }
    private var stepIcon: String {
        switch pass {
        case .setup:  return "ruler"
        case .depth:  return "cube.transparent"
        case .photos: return "camera.shutter.button"
        case .send:   return "square.and.arrow.up"
        }
    }

    private var statusBar: some View {
        HStack {
            Label(stepTitle, systemImage: stepIcon)
            Spacer()
            if !depthFrames.isEmpty || !photoFrames.isEmpty {
                Text("\(depthFrames.count) depth · \(photoFrames.count) photos").foregroundStyle(.green)
            } else if recording {
                Label("\(count) frames", systemImage: "record.circle").foregroundStyle(.red)
            } else {
                Text(useFaceDepth ? (faceDepth.ready ? "Ready" : "Aim at tools") : useTrueDepth ? (trueDepth.ready ? "Ready" : "Aim at tools") : (controller.trackingOK ? "Tracking" : "Move slowly…"))
            }
        }
        .font(.footnote.weight(.semibold))
        .padding(10)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))
        .foregroundStyle(.white)
    }

    /// Count down before recording starts, so there is time to bring the phone over the drawer.
    /// On the depth pass the screen is pointed AT the drawer, so the tick has to be a sound and a tap —
    /// the number on screen is only useful for the rear-camera case.
    private func startCountdown(from seconds: Int = 3) {
        guard countdown == nil, !recording else { return }
        countdownRun += 1
        let run = countdownRun                                // cancelling bumps this, stranding the old task
        countdown = seconds
        Task { @MainActor in
            for n in stride(from: seconds, through: 1, by: -1) {
                guard countdownRun == run, countdown != nil else { return }
                countdown = n
                UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
                AudioServicesPlaySystemSound(1103)            // tick
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard countdownRun == run, countdown != nil else { return }
            countdown = nil
            frameCount = 0
            startedAt = Date()
            recorder.onCount = { frameCount = $0 }
            // Exhaustive on purpose: this was `if useTrueDepth … else recorder.start(…)`, which silently sent the
            // TRACKED front path to the photogrammetry recorder, so FaceDepthController never recorded and every
            // scan stopped with "0 depth frames". A switch with no default makes the next added sensor a compile
            // error instead of an empty capture.
            switch depthSource {
            case .tracked:        faceDepth.beginRecording()
            case .plainTrueDepth: trueDepth.beginRecording()
            case .rear:           recorder.start(session: controller.session, interval: 0.5)
            }
            recording = true
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            AudioServicesPlaySystemSound(1113)                // distinct "go"
        }
    }

    private var bottomBar: some View {
        VStack(spacing: 10) {
            switch pass {
            case .setup:   setupStep
            case .depth:   depthStep
            case .photos:  photoStep
            case .send:    sendStep
            }
        }
        .padding()
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))
    }

    /// Step 1 — name the drawer and give its size. Typed size is trusted over anything measured, and it lets
    /// frames that see only one end register; leave it blank and the markers settle it.
    private var setupStep: some View {
        VStack(spacing: 10) {
            Text("Step 1 of 4 · This drawer").font(.headline).foregroundStyle(.white)
            // These inherited .foregroundStyle(.white) from the old "Drawer size" DisclosureGroup and were
            // white-on-white while you typed. They now take the adaptive primary colour, which is readable
            // against the rounded field in both light and dark mode — do not hard-code black here.
            TextField("Drawer name (e.g. Top left)", text: $drawerName)
                .textFieldStyle(.roundedBorder).foregroundStyle(.primary)
            HStack {
                TextField("Width mm", text: $drawerWidth).foregroundStyle(.primary)
                TextField("Depth mm", text: $drawerDepth).foregroundStyle(.primary)
            }
            .keyboardType(.decimalPad).textFieldStyle(.roundedBorder)
            Text("Size is optional — the corner markers can measure it. Type it if you already know it, or if a pass will only ever see one end of the drawer.")
                .font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
            if TrueDepthController.isSupported || FaceDepthController.isSupported {
                Picker("Scanner", selection: $captureSensor) {
                    if FaceDepthController.isSupported { Text("Front · tracked").tag("truedepth_tracked") }
                    if TrueDepthController.isSupported { Text("Front · plain").tag("truedepth") }
                    Text("Rear LiDAR").tag("rear")
                }.pickerStyle(.segmented)
                if captureSensor == "truedepth_tracked" {
                    Text(FaceDepthController.supportsWorldTracking
                         ? "Tracked: every frame carries its position, so frames that see no corner marker still land in the right place."
                         : "This device cannot world-track the front camera — frames will need a marker in view. Use Rear LiDAR instead.")
                        .font(.caption)
                        .foregroundStyle(FaceDepthController.supportsWorldTracking ? .green : .orange)
                        .multilineTextAlignment(.center)
                }
            }
            Button { pass = .depth } label: {
                Label("Next: depth scan", systemImage: "arrow.right.circle.fill").font(.headline).frame(maxWidth: .infinity).padding()
            }.buttonStyle(.borderedProminent)
        }
    }

    /// Step 2 — the topography pass.
    private var depthStep: some View {
        VStack(spacing: 10) {
            Text("Step 2 of 4 · Depth scan").font(.headline).foregroundStyle(.white)
            Text(frontPass
                 ? "Point the screen-side camera at the tools, 20–50 cm away. Pause briefly at each view and overlap by at least half. Keep the corner markers in view. A vibration marks every 5 frames."
                 : "Hold the phone about 50 cm above one end, glide slowly to the other. Keep the level near green.")
                .font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
            Text(useFaceDepth ? faceDepth.status : useTrueDepth ? trueDepth.status : recorder.status)
                .font(.caption).foregroundStyle(.white).multilineTextAlignment(.center)
            if useFaceDepth {
                // what ARKit is actually delivering — the difference between "no frames" and knowing why
                Text(faceDepth.diag)
                    .font(.caption2.monospaced())
                    .foregroundStyle(faceDepth.diag.contains("NO DEPTH") || faceDepth.diag.contains("FAILED") ? .orange : .white.opacity(0.7))
                    .multilineTextAlignment(.center)
            }
            if useFaceDepth && !faceDepth.trackingReady && faceDepth.isRunning {
                // Information, NOT a blocker: pointed at a close flat liner the front camera may never give ARKit
                // enough to build a world frame from, and the scan works without one.
                Text("No world pose yet — frames will be lined up by overlap instead, so overlap each view by half. Panning the room for a moment may earn poses, but you don't have to wait.")
                    .font(.caption2).foregroundStyle(.white.opacity(0.7)).multilineTextAlignment(.center)
                    .padding(6)
                    .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }
            if recording {
                Button {
                    recording = false
                    Task {
                        switch depthSource {
                        case .tracked:        depthFrames = faceDepth.finishRecording()
                        case .plainTrueDepth: depthFrames = await trueDepth.finishRecording()
                        case .rear:           depthFrames = await recorder.stop()
                        }
                        if depthFrames.count >= minFrames { pass = .photos }
                        else if depthFrames.isEmpty {
                            // Nothing at all is a fault, not a short sweep: say what the sensor was reporting.
                            error = "No depth frames were recorded. \(useFaceDepth ? faceDepth.diag : "")"
                        } else { error = "Only \(depthFrames.count) depth frames; sweep a little longer." }
                    }
                } label: {
                    // ALWAYS show the running count. "Stop (need 12+)" hid a capture that was recording nothing.
                    Label(count < minFrames ? "Stop · \(count) frames (need \(minFrames)+)" : "Stop · \(count) frames",
                          systemImage: "stop.circle.fill")
                        .font(.headline).frame(maxWidth: .infinity).padding()
                }.buttonStyle(.borderedProminent).tint(.red)
            } else if countdown != nil {
                Button { countdown = nil; countdownRun += 1 } label: {
                    Label("Cancel", systemImage: "xmark.circle").font(.headline).frame(maxWidth: .infinity).padding()
                }.buttonStyle(.bordered)
            } else {
                Button { startCountdown() } label: {
                    Label(depthFrames.isEmpty ? "Start depth scan" : "Scan again", systemImage: "record.circle")
                        .font(.headline).frame(maxWidth: .infinity).padding()
                }
                .buttonStyle(.borderedProminent)
                .disabled(uploading || (useFaceDepth ? !faceDepth.isRunning
                                        : useTrueDepth ? !trueDepth.isRunning : !controller.trackingOK))
                HStack {
                    Button { pass = .setup } label: { Label("Back", systemImage: "arrow.left") }.buttonStyle(.bordered)
                    if !depthFrames.isEmpty {
                        Button { pass = .photos } label: { Label("Next: photos", systemImage: "arrow.right").frame(maxWidth: .infinity) }
                            .buttonStyle(.borderedProminent)
                    }
                }
            }
        }
    }

    /// Step 3 — sharp stills, one per tap, so none is taken mid-movement.
    private var photoStep: some View {
        VStack(spacing: 10) {
            Text("Step 3 of 4 · Photos").font(.headline).foregroundStyle(.white)
            Text("Start with an overview showing the corner markers. Then move straight across the drawer in overlapping rows, keeping the same camera orientation. Keep about half the previous view in each photo and capture each tool completely where possible.")
                .font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
            Button {
                shooting = true
                Task {
                    defer { shooting = false }
                    do {
                        let previous = photoFrames.last?.jpeg
                        let captured = try await controller.stillFrame()
                        photoFrames.append(captured)
                        if let previous {
                            let overlap = await Task.detached(priority: .userInitiated) {
                                CaptureController.photoOverlap(previous: previous, current: captured.jpeg)
                            }.value
                            if let overlap {
                                photoFeedback = overlap < 0.45
                                    ? "Only about \(Int(overlap * 100))% overlap. Add a photo between the last two views."
                                    : "About \(Int(overlap * 100))% overlap with the previous photo. Continue across the drawer."
                            } else {
                                photoFeedback = "Could not confirm overlap. Move back slightly and include more of the previous view."
                            }
                        } else {
                            photoFeedback = "Overview saved. Now take overlapping close-up photos across the drawer."
                        }
                        UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
                    } catch { self.error = error.localizedDescription }
                }
            } label: {
                Label(shooting ? "…" : "Take photo · \(photoFrames.count) so far", systemImage: "camera.shutter.button")
                    .font(.headline).frame(maxWidth: .infinity).padding()
            }
            .buttonStyle(.borderedProminent)
            .disabled(shooting || uploading || !controller.trackingOK || controller.tiltDegrees > 12)
            if controller.tiltDegrees > 12 {
                Label("Hold the phone level above the tools before shooting", systemImage: "level")
                    .font(.footnote).foregroundStyle(.orange)
            }
            if let photoFeedback {
                Text(photoFeedback).font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
            }
            if !photoFrames.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(Array(photoFrames.enumerated()), id: \.offset) { index, frame in
                            if let image = CaptureController.photoThumbnail(frame.jpeg) {
                                Image(uiImage: image).resizable().scaledToFill().frame(width: 58, height: 52).clipped()
                                    .overlay(alignment: .bottomTrailing) { Text("\(index + 1)").font(.caption2).padding(3).background(.black.opacity(0.7)) }
                            }
                        }
                    }
                }
                Text("The server checks drawer coverage after upload. Include both ends and all four edges.")
                    .font(.caption).foregroundStyle(.white.opacity(0.8))
            }
            HStack {
                Button { pass = .depth } label: { Label("Back", systemImage: "arrow.left") }.buttonStyle(.bordered)
                if !photoFrames.isEmpty {
                    Button { photoFrames.removeLast(); photoFeedback = nil } label: { Label("Undo", systemImage: "arrow.uturn.backward") }.buttonStyle(.bordered)
                    Button { pass = .send } label: { Label("Next: send", systemImage: "arrow.right").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    /// Step 4 — what is about to be sent, then send it.
    private var sendStep: some View {
        VStack(spacing: 10) {
            Text("Step 4 of 4 · Send to server").font(.headline).foregroundStyle(.white)
            VStack(alignment: .leading, spacing: 4) {
                Label("\(drawerName.isEmpty ? "Unnamed drawer" : drawerName)", systemImage: "tray")
                Label("\(depthFrames.count) depth frames", systemImage: "cube.transparent")
                Label("\(photoFrames.count) photos", systemImage: "photo.stack")
                Label(drawerWidth.isEmpty || drawerDepth.isEmpty ? "Size from the markers" : "Size \(drawerWidth) × \(drawerDepth) mm",
                      systemImage: "ruler")
            }
            .font(.footnote).foregroundStyle(.white).frame(maxWidth: .infinity, alignment: .leading)
            Button {
                uploading = true
                Task { await upload(depthFrames + photoFrames) }
            } label: {
                Label("Send to server", systemImage: "square.and.arrow.up.fill").font(.headline).frame(maxWidth: .infinity).padding()
            }
            .buttonStyle(.borderedProminent).tint(.green)
            .disabled(uploading || depthFrames.isEmpty)
            Button { pass = .photos } label: { Label("Back", systemImage: "arrow.left") }.buttonStyle(.bordered)
        }
    }

    private func upload(_ frames: [SweepFrame]) async {
        guard frames.count >= minFrames else { uploading = false; error = "Only \(frames.count) frames; glide a little longer."; return }
        uploading = true
        statusMessage = "Uploading \(frames.count) frames and fusing…"
        defer { uploading = false; statusMessage = nil }
        do {
            // if this drawer was captured before, its size helps register frames that see only one end
            let known = store.drawers.first(where: { !drawerName.isEmpty && $0.name == drawerName && $0.widthMm != nil && $0.heightMm != nil })
            var size: (Double, Double)? = known.map { ($0.widthMm!, $0.heightMm!) }
            if !drawerWidth.isEmpty || !drawerDepth.isEmpty {
                guard let w = Double(drawerWidth.replacingOccurrences(of: ",", with: ".")),
                      let h = Double(drawerDepth.replacingOccurrences(of: ",", with: ".")),
                      w.isFinite, h.isFinite, w > 0, h > 0 else {
                    throw APIError.transport("Enter both drawer dimensions as positive millimetres.")
                }
                size = (w, h)
            }
            let info = try await APIClient(base: settings.apiBase).uploadLidarArc(frames: frames, markerSizeMm: settings.markerSizeMm,
                                                                                  insetMm: settings.markerInsetMm, drawerSize: size)
            let used = info.scan?["frames_used"]?.intValue ?? frames.count
            store.add(DrawerRecord(id: info.id, name: drawerName.isEmpty ? "Drawer \(store.drawers.count + 1)" : drawerName,
                                   capturedAt: Date(), hadDepth: frames.contains { $0.depth != nil }, markers: used,
                                   widthMm: info.mat_mm?.width, heightMm: info.mat_mm?.height, sensor: frames.first?.sensor))
            reviewSession = info.id
        } catch {
            self.error = error.localizedDescription
        }
    }
}

/// Bullseye level: the bubble slides the way the phone is leaning, in any direction, and turns green
/// when the camera points within 3° of straight down. Tilt toward the bubble to bring it home.
struct LevelGauge: View {
    let tilt: Double            // degrees off straight down (= length of `offset`)
    let offset: CGSize          // direction of the lean, screen axes, degrees
    private let fullScale = 15.0    // degrees at the outer ring
    private let radius = 60.0

    var body: some View {
        let ok = tilt < 3
        // pt per degree, with the bubble pinned to the rim once it is past full scale
        let mag = max(hypot(offset.width, offset.height), 0.0001)
        let scale = min(mag, fullScale) / mag * (radius / fullScale)
        ZStack {
            Circle().stroke(.white.opacity(0.7), lineWidth: 2).frame(width: radius * 2, height: radius * 2)
            Circle().stroke(ok ? .green : .white.opacity(0.5), lineWidth: 2).frame(width: 40, height: 40)
            // cross hairs: which way is which is much easier to read with an axis to refer to
            Path { p in
                p.move(to: CGPoint(x: -radius, y: 0)); p.addLine(to: CGPoint(x: radius, y: 0))
                p.move(to: CGPoint(x: 0, y: -radius)); p.addLine(to: CGPoint(x: 0, y: radius))
            }
            .stroke(.white.opacity(0.25), lineWidth: 1)
            .frame(width: radius * 2, height: radius * 2)
            Circle().fill(ok ? .green : .orange).frame(width: 18, height: 18)
                .offset(x: offset.width * scale, y: offset.height * scale)
                .animation(.interactiveSpring(response: 0.18, dampingFraction: 0.75), value: offset)
            // the dot marks straight down, so it sits on the LOW side: say so, or which way to lean is a guess
            Text(ok ? "level" : String(format: "%.0f° · tilt toward the dot", tilt))
                .font(.caption.monospacedDigit())
                .foregroundStyle(ok ? .green : .white)
                .fixedSize()
                .offset(y: 78)
        }
        .frame(width: radius * 2, height: radius * 2)
    }
}
