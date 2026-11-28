import SwiftUI

/// Which job (customer / toolboard) captured tools go into. Lists the server's jobs newest first; "New job…" makes
/// one. With the server unreachable the last chosen job stays, and a new job is made LOCALLY ("local-…") — it is
/// created on the server, and its tools uploaded, the next time PendingUploads gets through.
struct JobPickerView: View {
    @EnvironmentObject var settings: AppSettings
    var onPicked: (() -> Void)? = nil
    @State private var jobs: [LibraryJob] = []
    @State private var loading = false
    @State private var error: String?
    @State private var showNew = false
    @State private var newName = ""

    var body: some View {
        List {
            if let error {
                Section {
                    Label(error, systemImage: "wifi.exclamationmark").foregroundStyle(.orange).font(.footnote)
                    if let name = settings.currentJobName, settings.currentJobId != nil {
                        Button("Keep using “\(name)”") { onPicked?() }
                    }
                }
            }
            Section {
                if loading && jobs.isEmpty { ProgressView() }
                if !loading && jobs.isEmpty && error == nil {
                    Text("No jobs yet — make one for this customer or toolboard.").foregroundStyle(.secondary)
                }
                ForEach(jobs) { job in
                    Button { pick(id: job.id, name: job.name) } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(job.name).font(.headline).foregroundStyle(.primary)
                                Text("\(job.tool_count ?? 0) tool\((job.tool_count ?? 0) == 1 ? "" : "s")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if job.id == settings.currentJobId { Image(systemName: "checkmark").foregroundStyle(.tint) }
                        }
                    }
                }
            } header: { Text("Jobs") }
            Section {
                Button { newName = ""; showNew = true } label: { Label("New job…", systemImage: "plus.circle") }
            }
        }
        .navigationTitle("Choose a job")
        .task { await load() }
        .refreshable { await load() }
        .alert("New job", isPresented: $showNew) {
            TextField("Customer or toolboard name", text: $newName)
            Button("Create") { Task { await create() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Tools you capture are saved into this job; the website lists them for the toolboard.") }
    }

    private func load() async {
        loading = true
        defer { loading = false }
        do {
            jobs = try await LibraryClient(base: settings.apiBase).listJobs()
            error = nil
            // keep the stored name in step with the server's
            if let id = settings.currentJobId, let j = jobs.first(where: { $0.id == id }) { settings.currentJobName = j.name }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func create() async {
        let name = String(newName.trimmingCharacters(in: .whitespacesAndNewlines).prefix(80))
        guard !name.isEmpty else { return }
        do {
            let job = try await LibraryClient(base: settings.apiBase).createJob(name: name)
            pick(id: job.id, name: job.name)
        } catch APIError.transport {
            // offline: make it locally; PendingUploads creates it on the server later
            pick(id: "local-" + ToolCaptureModel.newToolId(), name: name)
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func pick(id: String, name: String) {
        settings.currentJobId = id
        settings.currentJobName = name
        onPicked?()
    }
}
