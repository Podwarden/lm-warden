import Foundation

/// What the dashboard window does with a navigation request
/// (BACKLOG-macos P7, mockup screen 5).
enum DashboardNavigation: Equatable {
    /// Load it inside the dashboard window.
    case allow
    /// Cancel it in the window and open it in the default browser.
    case openExternally
    /// Cancel it and do nothing else.
    case cancel
}

/// The dashboard window's navigation rule, kept free of AppKit/WebKit so it
/// can be unit-tested without a window server.
///
/// - Same origin as the dashboard (scheme + host + port; the path is ignored,
///   default ports 80/443 are filled in) -> `.allow`.
/// - `about:blank` -> `.allow` (WebKit uses it for empty frames).
/// - Any other http/https URL -> `.openExternally`.
/// - Everything else (file, javascript, data, mailto, other about:, nil) -> `.cancel`.
enum DashboardNavigationPolicy {
    static func decide(url: URL?, dashboard: URL) -> DashboardNavigation {
        guard let url, let scheme = url.scheme?.lowercased() else { return .cancel }
        if scheme == "about" {
            return url.absoluteString.lowercased() == "about:blank" ? .allow : .cancel
        }
        guard scheme == "http" || scheme == "https" else { return .cancel }
        if let target = Origin(url), let home = Origin(dashboard), target == home {
            return .allow
        }
        return .openExternally
    }

    private struct Origin: Equatable {
        let scheme: String
        let host: String
        let port: Int

        init?(_ url: URL) {
            guard let scheme = url.scheme?.lowercased(),
                  let host = url.host?.lowercased(), !host.isEmpty else { return nil }
            let port: Int
            if let explicit = url.port {
                port = explicit
            } else if scheme == "http" {
                port = 80
            } else if scheme == "https" {
                port = 443
            } else {
                return nil
            }
            self.scheme = scheme
            self.host = host
            self.port = port
        }
    }
}
