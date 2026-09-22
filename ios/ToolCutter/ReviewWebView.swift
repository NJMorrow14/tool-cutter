import SwiftUI
import WebKit

/// The review / layout / export UI, served by the Mac, opened on the captured session.
struct ReviewWebView: View {
    let url: URL
    let title: String
    var body: some View {
        WebView(url: url)
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .ignoresSafeArea(edges: .bottom)
    }
}

struct WebView: UIViewRepresentable {
    let url: URL
    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.allowsInlineMediaPlayback = true
        let v = WKWebView(frame: .zero, configuration: cfg)
        v.allowsBackForwardNavigationGestures = true
        v.load(URLRequest(url: url))
        return v
    }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
