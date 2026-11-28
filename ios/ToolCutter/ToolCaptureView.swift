import SwiftUI
import UIKit

/// "Capture tools" (Nolan, 2026-10-13: "take pictures of each tool in the app one by one and they automatically
/// outline. Users can use their fingers to retrace incorrect portions of the outline ..."). One screen that loops:
/// aim -> shutter -> automatic outline -> finger corrections -> name + thickness -> save -> next tool.
struct ToolCaptureView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var uploads: PendingUploads
    @StateObject private var controller = CaptureController()
    @StateObject private var model = ToolCaptureModel()
    @State private var toast: String?
    @State private var toastRun = 0
    @State private var openToolboard: URL?
    @State private var choosingJob = false

    var body: some View {
        Group {
            if settings.currentJobId == nil || choosingJob {
                JobPickerView { choosingJob = false }
            } else {
                capture
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(item: $openToolboard) { url in
            ReviewWebView(url: url, title: settings.currentJobName ?? "Toolboard")
        }
    }

    private var jobName: String { settings.currentJobName ?? "this job" }

    private var capture: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if model.phase == .aiming {
                aiming
            } else {
                editing
            }
            if let toast {
                VStack {
                    Text(toast)
                        .font(.subheadline.weight(.semibold)).foregroundStyle(.white)
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(.black.opacity(0.75), in: Capsule())
                        .padding(.top, 8)
                    Spacer()
                }
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .navigationTitle(model.phase == .aiming ? "Capture tools" : "Outline")
        .toolbarBackground(.black, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .onAppear { if model.phase == .aiming { controller.start() } }
        .onDisappear { controller.stop() }
        .onChange(of: model.phase) { old, new in
            // the camera runs only while aiming; editing a photo does not need it
            // (it keeps running through .processing: the still is taken from the live session)
            if new == .aiming && old != .aiming { controller.start() }
            if old == .processing && new == .editing { controller.stop() }
        }
        .task(id: settings.currentJobId) {
            if let id = settings.currentJobId { await model.refreshCount(jobId: id, settings: settings, uploads: uploads) }
        }
        .sheet(isPresented: Binding(get: { model.phase == .details || model.phase == .saving },
                                    set: { if !$0 && model.phase == .details { model.phase = .editing } })) {
            ToolDetailsSheet(model: model, jobName: jobName) { name, thickness, longest in
                Task { await save(name: name, thickness: thickness, longest: longest) }
            }
            .environmentObject(settings)
            .presentationDetents([.medium, .large])
            .interactiveDismissDisabled(model.phase == .saving)
        }
    }

    // MARK: - aiming

    private var aiming: some View {
        ZStack {
            if CaptureController.isSupported {
                ARPreview(session: controller.session).ignoresSafeArea()
            } else {
                Text("This needs an iPhone with ARKit.").foregroundStyle(.white).padding()
            }
            VStack(spacing: 10) {
                HStack {
                    lidarLine
                    Spacer()
                    Button { choosingJob = true } label: {
                        Label(jobName, systemImage: "folder").lineLimit(1)
                    }
                    .font(.footnote.weight(.semibold)).foregroundStyle(.white)
                }
                .padding(10)
                .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))
                if let h = model.hint {
                    Text(h).font(.footnote).foregroundStyle(.orange).multilineTextAlignment(.center)
                        .padding(8).background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                }
                Spacer()
                LevelGauge(tilt: controller.tiltDegrees, offset: controller.tiltOffset)
                Spacer()
                Text("Lay the tool on a flat, plain surface. Fill the frame. Hold level.")
                    .font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
                    .padding(8).background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                HStack(alignment: .center) {
                    counter.frame(maxWidth: .infinity, alignment: .leading)
                    shutterButton
                    toolboardButton.frame(maxWidth: .infinity, alignment: .trailing)
                }
                uploadLine
            }
            .padding()
        }
    }

    private var lidarLine: some View {
        let cm = controller.cameraHeightMm / 10
        let text: String
        let color: Color
        if !CaptureController.supportsDepth { text = "No LiDAR · type sizes"; color = .orange }
        else if !controller.hasDepth || cm == 0 { text = "LiDAR · looking…"; color = .white.opacity(0.7) }
        else { text = String(format: "LiDAR · %.0f cm", cm); color = .green }
        return Label(text, systemImage: "ruler").font(.footnote.weight(.semibold)).foregroundStyle(color)
    }

    private var shutterButton: some View {
        Button {
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
            Task { await model.shutter(controller: controller) }
        } label: {
            ZStack {
                Circle().fill(.white).frame(width: 66, height: 66)
                Circle().stroke(.white, lineWidth: 4).frame(width: 80, height: 80)
            }
        }
        .disabled(controller.isCapturing || !CaptureController.isSupported)
        .accessibilityLabel("Take photo")
    }

    private var counter: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("\(model.toolsInJob)").font(.title2.weight(.bold).monospacedDigit())
            Text("tool\(model.toolsInJob == 1 ? "" : "s") in \(jobName)").font(.caption2).lineLimit(2)
        }
        .foregroundStyle(.white)
    }

    @ViewBuilder private var toolboardButton: some View {
        if let id = settings.currentJobId, !id.hasPrefix("local-") {
            Button { openToolboard = settings.toolboardURL(jobId: id) } label: {
                VStack(spacing: 2) { Image(systemName: "square.grid.3x3"); Text("Toolboard").font(.caption2) }
            }
            .foregroundStyle(.white)
        } else {
            Color.clear.frame(width: 1, height: 1)
        }
    }

    @ViewBuilder private var uploadLine: some View {
        if uploads.pendingCount > 0 {
            Text(uploads.isUploading ? "uploading \(uploads.pendingCount)…"
                 : "\(uploads.pendingCount) queued" + (uploads.lastError.map { " · \($0)" } ?? ""))
                .font(.caption2).foregroundStyle(.white.opacity(0.7)).lineLimit(1)
        }
    }

    // MARK: - editing

    private var editing: some View {
        VStack(spacing: 0) {
            ZStack {
                OutlineEditorView(image: model.workImage, ring: model.outline?.ring, busy: model.busy,
                                  onStroke: { model.applyStroke($0) }, onTap: { model.tap(at: $0) })
                if model.findingOutline || model.workImage == nil {
                    VStack(spacing: 10) {
                        ProgressView().tint(.white)
                        Text(model.workImage == nil ? "Taking the photo…" : "Finding the outline…").foregroundStyle(.white)
                    }
                    .padding(16).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
                }
                VStack {
                    if let h = model.hint {
                        Text(h).font(.footnote).foregroundStyle(.white).multilineTextAlignment(.center)
                            .padding(8).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                            .padding(.top, 8)
                    }
                    Spacer()
                    HStack(spacing: 8) {
                        if model.busy { ProgressView().tint(.white).scaleEffect(0.8) }
                        Text(statusLine).font(.caption2).foregroundStyle(.white.opacity(0.8)).lineLimit(1)
                    }
                    .padding(6).background(.black.opacity(0.45), in: Capsule()).padding(.bottom, 6)
                }
                .allowsHitTesting(false)
            }
            editToolbar
        }
    }

    private var statusLine: String {
        var parts: [String] = []
        if !model.scaleStatus.isEmpty { parts.append(model.scaleStatus) }
        switch model.samStatus {
        case "ready": parts.append(model.encodedForCurrentPhoto ? "smart retrace" : "smart retrace warming up")
        case "unavailable": parts.append("retrace follows your line")
        default: break
        }
        return parts.joined(separator: " · ")
    }

    private var editToolbar: some View {
        HStack(spacing: 6) {
            toolbarButton("Undo", "arrow.uturn.backward") { model.undo() }.disabled(!model.canUndo || model.busy)
            toolbarButton("Reset", "arrow.counterclockwise") { model.reset() }.disabled(model.busy || model.findingOutline)
            toolbarButton("Retake", "camera") { model.retake() }
            Button { model.phase = .details } label: {
                Text("Next →").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(Color(red: 0xf2 / 255, green: 0x6a / 255, blue: 0x1b / 255))
            .disabled(model.outline == nil || model.busy || model.findingOutline)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(Color(white: 0.1))
    }

    private func toolbarButton(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 2) { Image(systemName: icon).font(.title3); Text(title).font(.caption2) }
                .frame(minWidth: 52)
        }
        .foregroundStyle(.white)
    }

    // MARK: - save

    private func save(name: String, thickness: Double?, longest: Double?) async {
        guard let jobId = settings.currentJobId else { return }
        guard let saved = await model.save(name: name, thicknessMm: thickness, longestSideMm: longest, jobId: jobId,
                                           jobName: jobName, uploads: uploads, settings: settings) else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        toastRun += 1
        let run = toastRun
        withAnimation { toast = "Saved \(saved) · \(model.toolsInJob) in \(jobName)" }
        try? await Task.sleep(nanoseconds: 2_500_000_000)
        if run == toastRun { withAnimation { toast = nil } }
    }
}

/// Name, thickness, (longest side when there is no LiDAR scale), and Save.
struct ToolDetailsSheet: View {
    @ObservedObject var model: ToolCaptureModel
    let jobName: String
    var onSave: (_ name: String, _ thicknessMm: Double?, _ longestSideMm: Double?) -> Void
    @State private var name = ""
    @State private var thickness = ""
    @State private var longest = ""
    @FocusState private var focused: Bool

    private var thicknessValue: Double? { Self.number(thickness).flatMap { $0 > 0 && $0 <= 300 ? $0 : nil } }
    private var longestValue: Double? { Self.number(longest).flatMap { $0 > 0 ? $0 : nil } }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Tool \(model.toolsInJob + 1)", text: $name).focused($focused)
                    HStack {
                        Text("Thickness")
                        Spacer()
                        TextField("mm", text: $thickness).keyboardType(.decimalPad).multilineTextAlignment(.trailing).frame(width: 90)
                        Text("mm").foregroundStyle(.secondary)
                    }
                    if !model.hasLidarScale {
                        HStack {
                            Text("Longest side")
                            Spacer()
                            TextField("mm", text: $longest).keyboardType(.decimalPad).multilineTextAlignment(.trailing).frame(width: 90)
                            Text("mm").foregroundStyle(.secondary)
                        }
                    }
                } footer: {
                    VStack(alignment: .leading, spacing: 4) {
                        if thicknessValue == nil {
                            Text("No thickness — you can save anyway and set it on the website; the pocket depth needs it.")
                                .foregroundStyle(.orange)
                        }
                        Text(model.scaleStatus.isEmpty ? "Measuring…" : model.scaleStatus)
                        if !model.hasLidarScale && longestValue == nil {
                            Text("Without a length the size is a guess (camera assumed 40 cm away).").foregroundStyle(.orange)
                        }
                        if let s = model.sizeMm(thicknessMm: thicknessValue, longestSideMm: longestValue) {
                            Text(String(format: "Outline %.0f × %.0f mm", s.width, s.height))
                        }
                    }
                }
                Section {
                    LabeledContent("Job", value: jobName)
                }
                Section {
                    Button {
                        onSave(name, thicknessValue, longestValue)
                    } label: {
                        HStack {
                            Spacer()
                            if model.phase == .saving { ProgressView() } else { Text("Save tool").font(.headline) }
                            Spacer()
                        }
                    }
                    .disabled(model.phase == .saving)
                }
            }
            .navigationTitle("Tool details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Back") { model.phase = .editing }.disabled(model.phase == .saving) }
            }
        }
    }

    /// Accepts "28", "28.5" and "28,5" (decimal-pad comma locales).
    static func number(_ s: String) -> Double? {
        Double(s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: "."))
    }
}
