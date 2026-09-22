import SwiftUI

struct HomeView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var store: SessionStore
    @State private var serverOK: Bool?
    @State private var showSettings = false
    @State private var openSession: String?
    @State private var openTitle = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    NavigationLink {
                        CaptureView()
                    } label: {
                        Label("Scan a drawer", systemImage: "camera.viewfinder").font(.headline)
                    }
                    Link(destination: settings.markerSheetURL) {
                        Label("Print the corner marker sheet", systemImage: "printer")
                    }
                } footer: {
                    Text("Put the four printed markers in the drawer corners, lay the tools out, then glide the phone along the drawer about 50 cm up. Outlines, heights and the drawer size come out to scale; review, arrange and export on the next screen.")
                }

                Section("Drawers") {
                    if store.drawers.isEmpty {
                        Text("No drawers captured yet").foregroundStyle(.secondary)
                    }
                    ForEach(store.drawers) { d in
                        Button {
                            openTitle = d.name
                            openSession = d.id
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(d.name).font(.headline).foregroundStyle(.primary)
                                Text(subtitle(d)).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { idx in idx.map { store.drawers[$0] }.forEach(store.remove) }
                }

                Section("Server") {
                    HStack {
                        Circle().fill(serverOK == nil ? .gray : (serverOK! ? .green : .red)).frame(width: 10, height: 10)
                        Text(settings.apiBase.absoluteString).font(.footnote.monospaced())
                        Spacer()
                        Button("Settings") { showSettings = true }
                    }
                }
            }
            .navigationTitle("ToolCutter")
            .task { await checkServer() }
            .refreshable { await checkServer() }
            .sheet(isPresented: $showSettings) { SettingsView().environmentObject(settings) }
            .navigationDestination(item: $openSession) { sid in
                ReviewWebView(url: settings.reviewURL(sessionId: sid), title: openTitle)
            }
        }
    }

    private func subtitle(_ d: DrawerRecord) -> String {
        var parts: [String] = [d.capturedAt.formatted(date: .abbreviated, time: .shortened)]
        if let w = d.widthMm, let h = d.heightMm { parts.append(String(format: "%.0f × %.0f mm", w, h)) }
        parts.append(d.sensor == "truedepth" ? "TrueDepth" : d.hadDepth ? "LiDAR" : "no depth")
        parts.append("\(d.markers) frames")
        return parts.joined(separator: " · ")
    }

    private func checkServer() async {
        serverOK = (try? await APIClient(base: settings.apiBase).health()) ?? false
    }

}

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            Form {
                Section("ToolCutter server (your Mac)") {
                    TextField("Host or IP", text: $settings.serverHost).textInputAutocapitalization(.never).autocorrectionDisabled()
                    Stepper("API port \(settings.apiPort)", value: $settings.apiPort, in: 1...65535)
                    Stepper("Web port \(settings.webPort)", value: $settings.webPort, in: 1...65535)
                }
                Section {
                    HStack {
                        Text("Marker size")
                        Spacer()
                        TextField("mm", value: $settings.markerSizeMm, format: .number)
                            .keyboardType(.decimalPad).multilineTextAlignment(.trailing).frame(width: 80)
                        Text("mm")
                    }
                    HStack {
                        Text("Inset from drawer corner")
                        Spacer()
                        TextField("mm", value: $settings.markerInsetMm, format: .number)
                            .keyboardType(.decimalPad).multilineTextAlignment(.trailing).frame(width: 80)
                        Text("mm")
                    }
                } header: {
                    Text("Markers")
                } footer: {
                    Text("Marker size is the black square's edge as printed. Inset is how far the marker's outer corner sits from the true drawer corner (0 if you push it into the corner).")
                }
            }
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}
