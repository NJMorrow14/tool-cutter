import Foundation
import Combine

/// Tools saved on the phone but not yet on the Mac. Every save goes through here, so a capture never depends on the
/// network: the tool is written to disk first, then uploaded, and only deleted once the server answered 2xx.
///
/// Layout: Documents/pending/<toolId>/ { tool.json, photo.jpg, job.txt, jobname.txt, ready }. A tool is written into
/// `<toolId>.partial/`, the `ready` marker is written last and the folder is then renamed, so a crash mid-write leaves
/// a `.partial` folder that is ignored (and swept) rather than a half tool that uploads.
/// The phone makes the tool ids, so an upload retried after a lost reply simply replaces the same tool (200).
@MainActor
final class PendingUploads: ObservableObject {
    @Published private(set) var pendingCount = 0
    @Published private(set) var isUploading = false
    @Published private(set) var lastError: String?
    /// Bumped after every successful upload (views refresh their counts on it).
    @Published private(set) var uploadedCount = 0

    private let root: URL
    private let fm = FileManager.default
    /// Offline-made jobs ("local-…") already created on the server: local id -> real id.
    private var localJobMap: [String: String] {
        get { UserDefaults.standard.dictionary(forKey: "localJobMap") as? [String: String] ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: "localJobMap") }
    }

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        root = docs.appendingPathComponent("pending", isDirectory: true)
        try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        for dir in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] where dir.pathExtension == "partial" {
            try? fm.removeItem(at: dir)
        }
        recount()
    }

    /// Write one tool to disk (crash-safe). Throws only if the phone itself cannot write.
    func enqueue(toolId: String, jobId: String, jobName: String, toolJSON: Data, photoJPEG: Data) throws {
        let partial = root.appendingPathComponent(toolId + ".partial", isDirectory: true)
        let final = root.appendingPathComponent(toolId, isDirectory: true)
        try? fm.removeItem(at: partial)
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)
        try toolJSON.write(to: partial.appendingPathComponent("tool.json"), options: .atomic)
        try photoJPEG.write(to: partial.appendingPathComponent("photo.jpg"), options: .atomic)
        try Data(jobId.utf8).write(to: partial.appendingPathComponent("job.txt"), options: .atomic)
        try Data(jobName.utf8).write(to: partial.appendingPathComponent("jobname.txt"), options: .atomic)
        try Data().write(to: partial.appendingPathComponent("ready"), options: .atomic)
        try? fm.removeItem(at: final)
        try fm.moveItem(at: partial, to: final)
        recount()
    }

    /// Tools still waiting for a given job (for "N tools in <job>" while offline).
    func pending(forJob jobId: String) -> Int {
        readyDirs().filter { (try? String(contentsOf: $0.appendingPathComponent("job.txt"), encoding: .utf8)) == jobId }.count
    }

    /// Upload everything waiting, oldest first. A tool the server refuses (4xx) is kept and skipped so it cannot block
    /// the rest; an unreachable server stops the pass (the next save or app activation tries again).
    func flush(settings: AppSettings) {
        guard !isUploading else { return }
        let dirs = readyDirs()
        guard !dirs.isEmpty else { recount(); return }
        isUploading = true
        let client = LibraryClient(base: settings.apiBase)
        Task { @MainActor in
            defer { isUploading = false; recount() }
            var hadError: String?
            uploadLoop: for dir in dirs {
                guard var jobId = try? String(contentsOf: dir.appendingPathComponent("job.txt"), encoding: .utf8),
                      let toolJSON = try? Data(contentsOf: dir.appendingPathComponent("tool.json")),
                      let photo = try? Data(contentsOf: dir.appendingPathComponent("photo.jpg")) else { continue }
                if jobId.hasPrefix("local-") {
                    if let real = localJobMap[jobId] {
                        jobId = real
                    } else {
                        let name = (try? String(contentsOf: dir.appendingPathComponent("jobname.txt"), encoding: .utf8)) ?? "Phone job"
                        do {
                            let job = try await client.createJob(name: name.isEmpty ? "Phone job" : name)
                            localJobMap[jobId] = job.id
                            if settings.currentJobId == jobId { settings.currentJobId = job.id; settings.currentJobName = job.name }
                            jobId = job.id
                        } catch {
                            hadError = error.localizedDescription
                            break uploadLoop
                        }
                    }
                }
                do {
                    _ = try await client.uploadTool(jobId: jobId, toolJSON: toolJSON, photoJPEG: photo)
                    try? fm.removeItem(at: dir)
                    uploadedCount += 1
                    recount()
                } catch APIError.transport(let msg) {
                    hadError = msg
                    break uploadLoop
                } catch {
                    hadError = error.localizedDescription     // refused: keep it, try the others
                }
            }
            lastError = hadError
        }
    }

    private func readyDirs() -> [URL] {
        let dirs = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.creationDateKey])) ?? []
        return dirs
            .filter { $0.pathExtension.isEmpty && fm.fileExists(atPath: $0.appendingPathComponent("ready").path) }
            .sorted {
                let a = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let b = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return a < b
            }
    }

    private func recount() { pendingCount = readyDirs().count }
}
