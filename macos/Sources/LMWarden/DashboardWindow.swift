import AppKit
import WebKit

/// The dashboard window (BACKLOG-macos P7, mockup screen 5): a plain native
/// window with a WKWebView around the existing web UI at
/// `http://127.0.0.1:<port>/ui/`. Same-origin navigation stays in the window;
/// other http/https links open in the default browser; everything else is
/// cancelled (the rule lives in `DashboardNavigationPolicy`). Closing the
/// window only hides it -- the server keeps running.
///
/// View file: no decisions of its own and never instantiated in tests.
@MainActor
final class DashboardWindowController: NSObject, WKNavigationDelegate, WKUIDelegate, NSWindowDelegate {
    private var window: NSWindow?
    private var webView: WKWebView?
    private var loadedURL: URL?

    func show(url: URL) {
        let isNew = window == nil
        let (window, webView) = ensureWindow()
        if isNew || loadedURL != url {
            loadedURL = url
            webView.load(URLRequest(url: url))
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func ensureWindow() -> (NSWindow, WKWebView) {
        if let window, let webView { return (window, webView) }

        let frame = NSRect(x: 0, y: 0, width: 1200, height: 760)
        let webView = WKWebView(frame: frame, configuration: WKWebViewConfiguration())
        // WebKit holds both delegates weakly, so the controller owning the
        // web view is not a retain cycle.
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.autoresizingMask = [.width, .height]

        let window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "LM Warden"
        window.isReleasedWhenClosed = false
        window.contentView = webView
        window.center()
        // Restores a saved frame if there is one; otherwise keeps the centred default.
        window.setFrameAutosaveName("LMWardenDashboard")
        // NSWindow.delegate is weak too.
        window.delegate = self

        self.window = window
        self.webView = webView
        return (window, webView)
    }

    // MARK: NSWindowDelegate

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    // MARK: WKNavigationDelegate

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
        decisionHandler(apply(to: navigationAction.request.url))
    }

    // MARK: WKUIDelegate

    /// target=_blank and window.open(): never open a second web view. Off-origin
    /// http/https goes to the default browser; a same-origin popup loads in
    /// this window instead.
    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        let url = navigationAction.request.url
        if apply(to: url) == .allow, let url, url.scheme?.lowercased() != "about" {
            webView.load(URLRequest(url: url))
        }
        return nil
    }

    // MARK: Policy

    private func apply(to url: URL?) -> WKNavigationActionPolicy {
        guard let dashboard = loadedURL else { return .cancel }
        switch DashboardNavigationPolicy.decide(url: url, dashboard: dashboard) {
        case .allow:
            return .allow
        case .openExternally:
            if let url { NSWorkspace.shared.open(url) }
            return .cancel
        case .cancel:
            return .cancel
        }
    }
}
