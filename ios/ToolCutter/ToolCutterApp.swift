import SwiftUI

@main
struct ToolCutterApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var store = SessionStore()
    @StateObject private var uploads = PendingUploads()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environmentObject(settings)
                .environmentObject(store)
                .environmentObject(uploads)
        }
        .onChange(of: scenePhase) { _, phase in
            // tools saved while offline go up as soon as the app is back in front
            if phase == .active { uploads.flush(settings: settings) }
        }
    }
}
