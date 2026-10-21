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
    private var state = DashboardLoadState()
    private var pendingRetry: DispatchWorkItem?

    func show(url: URL) {
        let isNew = window == nil
        let (window, webView) = ensureWindow()
        let action = state.onShow(isNewWindow: isNew, urlChanged: loadedURL != url)
        loadedURL = url
        perform(action, in: webView)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// The server just became healthy: reload the dashboard if its last load
    /// failed, without bringing the window forward.
    func reloadIfFailed() {
        guard let webView, loadedURL != nil else { return }
        perform(state.onServerRunning(), in: webView)
    }

    private func perform(_ action: DashboardLoadAction, in webView: WKWebView) {
        switch action {
        case .none:
            break
        case .load:
            cancelRetry()
            if let url = loadedURL { webView.load(URLRequest(url: url)) }
        case .showPlaceholderAndSchedule(let delay):
            webView.loadHTMLString(Self.placeholderHTML, baseURL: nil)
            scheduleRetry(after: delay, in: webView)
        case .scheduleRetry(let delay):
            scheduleRetry(after: delay, in: webView)
        case .showError:
            cancelRetry()
            webView.loadHTMLString(Self.errorHTML, baseURL: nil)
        case .stopLoading:
            cancelRetry()
            webView.stopLoading()
        }
    }

    private func scheduleRetry(after delay: TimeInterval, in webView: WKWebView) {
        cancelRetry()
        let item = DispatchWorkItem { [weak self, weak webView] in
            MainActor.assumeIsolated {
                guard let self, let webView else { return }
                self.pendingRetry = nil
                self.perform(self.state.onRetryTimer(), in: webView)
            }
        }
        pendingRetry = item
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: item)
    }

    private func cancelRetry() {
        pendingRetry?.cancel()
        pendingRetry = nil
    }

    private func handleFailure(_ webView: WKWebView, _ error: Error) {
        let ns = error as NSError
        let action = state.onFailure(
            domain: ns.domain, code: ns.code,
            failingURL: ns.userInfo[NSURLErrorFailingURLErrorKey] as? URL, now: Date()
        )
        perform(action, in: webView)
    }

    private static func page(_ message: String) -> String {
        """
        <html><body style="font-family:-apple-system;display:flex;align-items:center;\
        justify-content:center;height:100vh;margin:0;color:#888;background:transparent">\
        <p style="max-width:28em;text-align:center">\(message)</p></body></html>
        """
    }

    private static let placeholderHTML = page("Starting the LM Warden server&hellip;")
    private static let errorHTML = page(
        "The LM Warden server did not answer. Choose Show Log from the menu, then Open Dashboard to try again."
    )

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
        if let webView { perform(state.onClose(), in: webView) }
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

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        state.onFinish(url: webView.url)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        handleFailure(webView, error)
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        handleFailure(webView, error)
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
