import SwiftUI
import ARKit
import SceneKit
import AudioToolbox

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

/// One screen, one button (Nolan, 2026-09-29: "make it super simple"; 2026-10-02: "move back to using the front
/// TrueDepth camera, but keep the app flow simple"). Two cameras, one gesture:
///   aim (REAR camera, marker preflight) -> Start -> one OVERVIEW still with the rear camera (measures the drawer
///   exactly from 3+ markers) -> the phone is turned screen-down -> 3-2-1 ticks -> FRONT TrueDepth sweep ->
///   Stop -> upload overview + depth frames -> review opens.
/// The front camera is the one that read a 25.0 mm rule at 24.7 mm; the rear LiDAR (256x144) could not see it at
/// all. Its image arrives mirrored and its poses are unusable — the server handles both (see CLAUDE.md); frames
/// register off the markers and off each other. The screen faces the drawer during the sweep, so there is no live
/// map and the countdown is sound + haptic.
struct CaptureView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var store: SessionStore
    @StateObject private var controller = CaptureController()      // rear: aiming, preflight, overview still
    @StateObject private var faceDepth = FaceDepthController()     // front: the depth sweep

    enum Phase: Equatable { case ready, overview, countdown(Int), recording, uploading, done(String) }
    @State private var phase: Phase = .ready
    @State private var countdownRun = 0
    @State private var overviewStill: SweepFrame?
    @State private var error: String?
    @State private var reviewSession: String?
    @State private var markersSeen: Int? = nil
    @State private var preflightNote: String? = nil
    private let minFrames = 8

    private var recording: Bool { phase == .recording }
    private var frontActive: Bool { if case .countdown = phase { return true }; return phase == .recording }

    var body: some View {
        ZStack {
            if frontActive {
                // screen is face-down over the drawer: nothing to see, so keep it dark and cool
                Color.black.ignoresSafeArea()
            } else if CaptureController.isSupported {
                ARPreview(session: controller.session).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("This needs an iPhone with a TrueDepth camera and LiDAR.").foregroundStyle(.white).padding()
            }

            VStack(spacing: 0) {
                topLine.padding(.top, 8)
                Spacer()
                if phase == .ready { LevelGauge(tilt: controller.tiltDegrees, offset: controller.tiltOffset) }
                if recording {
                    VStack(spacing: 8) {
                        Text("\(faceDepth.frameCount)").font(.system(size: 120, weight: .bold, design: .rounded)).foregroundStyle(.white)
                            .contentTransition(.numericText())
                        Text("frames · a buzz every 5").font(.headline).foregroundStyle(.white.opacity(0.8))
                    }
                }
                Spacer()
                bottom
            }
            .padding()

            if case .countdown(let c) = phase {
                Color.black.ignoresSafeArea()
                VStack(spacing: 10) {
                    Text("\(c)").font(.system(size: 140, weight: .bold, design: .rounded)).foregroundStyle(.white)
                        .contentTransition(.numericText())
                    Text("turn the phone screen-down over one end of the drawer").font(.headline).foregroundStyle(.white.opacity(0.9))
                        .multilineTextAlignment(.center)
                }
                .transaction { $0.animation = .easeOut(duration: 0.2) }
            }
            if phase == .overview {
                Color.black.opacity(0.5).ignoresSafeArea()
                VStack(spacing: 10) { ProgressView().tint(.white); Text("Taking the overview photo…").foregroundStyle(.white) }
            }
            if phase == .uploading {
                Color.black.opacity(0.55).ignoresSafeArea()
                VStack(spacing: 12) {
                    ProgressView().tint(.white)
                    Text("Building the drawer on the Mac…").foregroundStyle(.white)
                    Text("\(faceDepth.frameCount) depth frames + overview").font(.caption.monospaced()).foregroundStyle(.white.opacity(0.8))
                }.padding()
            }
        }
        .task { controller.start() }
        .task(id: phase == .ready) {
            guard phase == .ready else { return }
            while !Task.isCancelled && phase == .ready {
                if let jpeg = controller.previewJPEG() {
                    do { let r = try await APIClient(base: settings.apiBase).markers(jpeg: jpeg); markersSeen = r.count; preflightNote = nil }
                    catch { markersSeen = nil; preflightNote = "server?" }
                }
                try? await Task.sleep(nanoseconds: 1_200_000_000)
            }
        }
        .onDisappear { controller.stop(); faceDepth.stop() }
        .onChange(of: faceDepth.frameCount) { _, n in
            if recording && n > 0 && n % 5 == 0 { UIImpactFeedbackGenerator(style: .medium).impactOccurred() }
        }
        .alert("Scan failed", isPresented: Binding(get: { error != nil }, set: { _ in error = nil })) {
            Button("OK", role: .cancel) {}
        } message: { Text(error ?? "") }
        .navigationDestination(item: $reviewSession) { sid in
            ReviewWebView(url: settings.reviewURL(sessionId: sid), title: "Drawer")
        }
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - pieces

    private var topLine: some View {
        HStack {
            switch phase {
            case .ready:
                markerReadout
                Spacer()
                heightReadout
            case .overview:
                Label("Overview photo", systemImage: "camera")
            case .countdown:
                Label("Get set", systemImage: "timer")
            case .recording:
                Label("Recording", systemImage: "record.circle").foregroundStyle(.red)
                Spacer()
                Text(faceDepth.status).lineLimit(1).foregroundStyle(.white.opacity(0.9))
            case .uploading:
                Label("Uploading", systemImage: "arrow.up.circle")
            case .done:
                Label("Done", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            }
        }
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))
    }

    /// The preflight verdict. Green at 3+ markers: the overview still can measure the drawer from this view.
    private var markerReadout: some View {
        let n = markersSeen
        let ok = (n ?? 0) >= 3
        let text = n == nil ? (preflightNote == nil ? "Looking for markers…" : "Can't reach the server")
                 : ok ? "\(n!) markers · ready" : n == 0 ? "No markers in view — lift higher" : "\(n!) marker\(n! == 1 ? "" : "s") — lift until 3+ show"
        return Label(text, systemImage: ok ? "checkmark.circle.fill" : "viewfinder")
            .foregroundStyle(ok ? .green : n == nil ? .white.opacity(0.7) : .orange)
    }

    /// Height while aiming (rear LiDAR). The overview wants 50-70 cm so all four markers fit in one frame.
    private var heightReadout: some View {
        let cm = controller.cameraHeightMm / 10
        return Text(cm == 0 ? "— cm" : String(format: "%.0f cm", cm))
            .font(.footnote.monospacedDigit().weight(.semibold))
            .foregroundStyle(cm == 0 ? .white.opacity(0.6) : .white)
    }

    @ViewBuilder private var bottom: some View {
        VStack(spacing: 10) {
            switch phase {
            case .ready:
                Text("Lift the phone until 3 or more corner markers show, then press Start. It takes an overview photo; then turn the phone screen-down and glide the drawer 20–50 cm up — ticks count you in, a buzz marks every 5 frames.")
                    .font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
                Button { startScan() } label: {
                    Label("Start scan", systemImage: "record.circle").font(.title3.weight(.semibold))
                        .frame(maxWidth: .infinity).padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!controller.trackingOK || (markersSeen ?? 0) < 3 || !FaceDepthController.isSupported)
                if !FaceDepthController.isSupported {
                    Text("This iPhone has no TrueDepth camera.").font(.caption).foregroundStyle(.orange)
                }
            case .overview:
                EmptyView()
            case .countdown:
                Button { cancelToReady() } label: {
                    Label("Cancel", systemImage: "xmark.circle").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
                }.buttonStyle(.bordered).tint(.white)
            case .recording:
                Button { stopAndUpload() } label: {
                    Label(faceDepth.frameCount < minFrames ? "Stop · \(faceDepth.frameCount) frames (need \(minFrames)+)" : "Stop · \(faceDepth.frameCount) frames", systemImage: "stop.circle.fill")
                        .font(.title3.weight(.semibold)).frame(maxWidth: .infinity).padding(.vertical, 10)
                }.buttonStyle(.borderedProminent).tint(.red)
            case .uploading:
                EmptyView()
            case .done(let sid):
                Button { reviewSession = sid } label: {
                    Label("Open on the Mac", systemImage: "macbook").font(.title3.weight(.semibold))
                        .frame(maxWidth: .infinity).padding(.vertical, 6)
                }.buttonStyle(.borderedProminent)
                Button { phase = .ready } label: {
                    Label("Scan another drawer", systemImage: "arrow.counterclockwise").font(.headline)
                        .frame(maxWidth: .infinity).padding(.vertical, 4)
                }.buttonStyle(.bordered).tint(.white)
            }
        }
        .padding(12)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - flow

    /// Start = overview still (rear) -> switch cameras -> 3-2-1 -> record (front). One tap does all of it.
    private func startScan() {
        guard phase == .ready else { return }
        countdownRun += 1
        let run = countdownRun
        phase = .overview
        Task { @MainActor in
            do {
                overviewStill = try await controller.stillFrame()          // full-res rear photo: markers -> drawer size
            } catch {
                self.error = "Could not take the overview photo: \(error.localizedDescription)"
                phase = .ready; return
            }
            guard countdownRun == run else { return }
            controller.stop()                                             // one ARSession at a time
            faceDepth.start()
            phase = .countdown(3)
            for n in stride(from: 3, through: 1, by: -1) {
                guard countdownRun == run, case .countdown = phase else { return }
                phase = .countdown(n)
                UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
                AudioServicesPlaySystemSound(1103)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard countdownRun == run, case .countdown = phase else { return }
            faceDepth.beginRecording()
            phase = .recording
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            AudioServicesPlaySystemSound(1113)
        }
    }

    private func cancelToReady() {
        countdownRun += 1
        faceDepth.stop()
        controller.start()
        overviewStill = nil
        phase = .ready
    }

    /// Stop -> upload overview + depth -> open. Too few frames goes back to ready with the rear camera running.
    private func stopAndUpload() {
        let depth = faceDepth.finishRecording()
        faceDepth.stop()
        controller.start()
        guard depth.count >= minFrames else {
            phase = .ready
            error = "Only \(depth.count) depth frames — glide a little longer next time."
            return
        }
        phase = .uploading
        let frames = (overviewStill.map { [$0] } ?? []) + depth
        Task {
            do {
                let info = try await APIClient(base: settings.apiBase).uploadLidarArc(
                    frames: frames, markerSizeMm: settings.markerSizeMm, insetMm: settings.markerInsetMm, drawerSize: nil)
                let used = info.scan?["frames_used"]?.intValue ?? frames.count
                let name = "Drawer · " + Date().formatted(date: .abbreviated, time: .shortened)
                store.add(DrawerRecord(id: info.id, name: name, capturedAt: Date(), hadDepth: true,
                                       markers: used, widthMm: info.mat_mm?.width, heightMm: info.mat_mm?.height, sensor: "truedepth_tracked"))
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                phase = .done(info.id)
                reviewSession = info.id
            } catch {
                phase = .ready
                self.error = error.localizedDescription
            }
        }
    }
}

/// Bullseye level: the dot marks straight down, so it sits on the LOW side — tilt toward it. Green within 3°.
struct LevelGauge: View {
    let tilt: Double            // degrees off straight down (= length of `offset`)
    let offset: CGSize          // direction of the lean, screen axes, degrees
    private let fullScale = 15.0    // degrees at the outer ring
    private let radius = 60.0

    var body: some View {
        let ok = tilt < 3
        let mag = max(hypot(offset.width, offset.height), 0.0001)
        let scale = min(mag, fullScale) / mag * (radius / fullScale)
        ZStack {
            Circle().stroke(.white.opacity(0.7), lineWidth: 2).frame(width: radius * 2, height: radius * 2)
            Circle().stroke(ok ? .green : .white.opacity(0.5), lineWidth: 2).frame(width: 40, height: 40)
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
