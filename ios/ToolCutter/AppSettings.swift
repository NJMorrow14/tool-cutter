import Foundation
import Combine

/// Where the ToolCutter server lives (the Mac) and how the marker sheet was printed.
final class AppSettings: ObservableObject {
    @Published var serverHost: String { didSet { UserDefaults.standard.set(serverHost, forKey: "serverHost") } }
    @Published var apiPort: Int { didSet { UserDefaults.standard.set(apiPort, forKey: "apiPort") } }
    @Published var webPort: Int { didSet { UserDefaults.standard.set(webPort, forKey: "webPort") } }
    @Published var markerSizeMm: Double { didSet { UserDefaults.standard.set(markerSizeMm, forKey: "markerSizeMm") } }
    @Published var markerInsetMm: Double { didSet { UserDefaults.standard.set(markerInsetMm, forKey: "markerInsetMm") } }

    init() {
        let d = UserDefaults.standard
        serverHost = d.string(forKey: "serverHost") ?? "192.168.1.193"
        apiPort = d.object(forKey: "apiPort") as? Int ?? 8000
        webPort = d.object(forKey: "webPort") as? Int ?? 3000
        // 2026-10-02: markers went 50 -> 25 mm (a quarter of the area). A 50 stored from before that is the OLD default,
        // not a choice, so it is migrated once; anything else the user typed is kept.
        var stored = d.object(forKey: "markerSizeMm") as? Double ?? 25
        if stored == 50 && !d.bool(forKey: "markerSize25Migrated") { stored = 25 }
        d.set(true, forKey: "markerSize25Migrated")
        markerSizeMm = stored
        markerInsetMm = d.object(forKey: "markerInsetMm") as? Double ?? 0
    }

    var apiBase: URL { URL(string: "http://\(serverHost):\(apiPort)")! }
    var webBase: URL { URL(string: "http://\(serverHost):\(webPort)")! }
    func reviewURL(sessionId: String) -> URL {
        URL(string: "http://\(serverHost):\(webPort)/?session=\(sessionId)")!
    }
    var markerSheetURL: URL {
        URL(string: "http://\(serverHost):\(apiPort)/api/marker_sheet.svg?marker_mm=\(String(format: "%g", markerSizeMm))")!   // %g, not Int: 12.5 must not print as 12
    }
}

/// Drawers captured on this phone (ids live on the server while it runs; names live here).
struct DrawerRecord: Identifiable, Codable, Equatable {
    var id: String            // server session id
    var name: String
    var capturedAt: Date
    var hadDepth: Bool
    var markers: Int
    var widthMm: Double?
    var heightMm: Double?
    var sensor: String? = nil
}

final class SessionStore: ObservableObject {
    @Published var drawers: [DrawerRecord] = [] { didSet { save() } }
    private let key = "drawers.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: key),
           let list = try? JSONDecoder().decode([DrawerRecord].self, from: data) {
            drawers = list
        }
    }

    func add(_ r: DrawerRecord) { drawers.insert(r, at: 0) }
    func remove(_ r: DrawerRecord) { drawers.removeAll { $0.id == r.id } }
    private func save() {
        if let data = try? JSONEncoder().encode(drawers) { UserDefaults.standard.set(data, forKey: key) }
    }
}
