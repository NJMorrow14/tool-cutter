import SwiftUI

@main
struct ToolCutterApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var store = SessionStore()

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environmentObject(settings)
                .environmentObject(store)
        }
    }
}
