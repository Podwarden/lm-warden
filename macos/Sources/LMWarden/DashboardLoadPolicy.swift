import Foundation

/// What the dashboard window does after a failed load.
enum DashboardRetry: Equatable {
    /// Try the dashboard URL again after this many seconds.
    case retry(after: TimeInterval)
    /// Leave the window as it is.
    case giveUp
}

/// Retry and reload rules for the dashboard window, kept free of AppKit/WebKit
/// so they can be unit-tested. The server may still be starting when the
/// window first opens (connection refused), so those errors are retried with a
/// doubling delay (0.5 s up to 4 s) until `deadline` seconds have passed since
/// the first failure.
enum DashboardLoadPolicy {
    static let initialDelay: TimeInterval = 0.5
    static let maxDelay: TimeInterval = 4
    static let deadline: TimeInterval = 180

    /// The URL-loading errors that mean "the server is not answering (yet)".
    static func isRetryable(domain: String, code: Int) -> Bool {
        guard domain == NSURLErrorDomain else { return false }
        switch code {
        case NSURLErrorCannotConnectToHost,   // -1004 connection refused
             NSURLErrorCannotFindHost,        // -1003
             NSURLErrorTimedOut,              // -1001
             NSURLErrorNetworkConnectionLost: // -1005
            return true
        default:
            return false // -999 cancelled, -1000 bad URL, ...
        }
    }

    /// Delay before retry number `attempt` (0-based): 0.5, 1, 2, 4, 4, ...
    static func delay(attempt: Int) -> TimeInterval {
        let n = max(0, min(attempt, 10))
        return min(initialDelay * pow(2, Double(n)), maxDelay)
    }

    /// `attempt` is how many retries already happened; `elapsed` the seconds
    /// since the first failure of this run of failures.
    static func decide(domain: String, code: Int, attempt: Int, elapsed: TimeInterval) -> DashboardRetry {
        guard isRetryable(domain: domain, code: code), elapsed < deadline else { return .giveUp }
        return .retry(after: delay(attempt: attempt))
    }

    /// `show(url:)` reloads for a new window, a different URL, or when the
    /// last load failed (so the menu's "Open Dashboard" recovers a blank window).
    static func shouldReloadOnShow(isNewWindow: Bool, urlChanged: Bool, lastLoadFailed: Bool) -> Bool {
        isNewWindow || urlChanged || lastLoadFailed
    }
}

/// What the window must do in response to an event; the window only executes these.
enum DashboardLoadAction: Equatable {
    case none
    /// Load the dashboard URL now (cancelling any pending retry).
    case load
    /// Show the "Starting..." placeholder, then call `onRetryTimer()` after the delay.
    case showPlaceholderAndSchedule(TimeInterval)
    /// Call `onRetryTimer()` after the delay.
    case scheduleRetry(TimeInterval)
    /// Replace the placeholder with the error page.
    case showError
    /// Cancel any pending retry and stop the web view's load.
    case stopLoading
}

/// The dashboard window's load state machine, free of AppKit/WebKit.
struct DashboardLoadState {
    private(set) var lastLoadFailed = false
    private(set) var attempt = 0
    private(set) var firstFailure: Date?
    private(set) var retryPending = false
    private(set) var errorShown = false
    private(set) var closed = false

    /// `show(url:)`. Reloads for a new window, a new URL or a failed last load.
    mutating func onShow(isNewWindow: Bool, urlChanged: Bool) -> DashboardLoadAction {
        closed = false
        guard DashboardLoadPolicy.shouldReloadOnShow(
            isNewWindow: isNewWindow, urlChanged: urlChanged, lastLoadFailed: lastLoadFailed
        ) else { return .none }
        return beginLoad()
    }

    /// The supervisor reported the server running: reload only if the last load failed.
    mutating func onServerRunning() -> DashboardLoadAction {
        lastLoadFailed ? beginLoad() : .none
    }

    private mutating func beginLoad() -> DashboardLoadAction {
        lastLoadFailed = false
        attempt = 0
        firstFailure = nil
        retryPending = false
        errorShown = false
        return .load
    }

    /// didFailProvisionalNavigation / didFail.
    mutating func onFailure(domain: String, code: Int, failingURL: URL?, now: Date) -> DashboardLoadAction {
        // The placeholder or error page, a superseded load, an interrupted frame load.
        if failingURL?.scheme?.lowercased() == "about" { return .none }
        if domain == NSURLErrorDomain && code == NSURLErrorCancelled { return .none }
        if domain == "WebKitErrorDomain" && code == 102 { return .none }
        let retryable = DashboardLoadPolicy.isRetryable(domain: domain, code: code)
        // Another error with nothing retried yet: leave the page alone.
        if !retryable && attempt == 0 { return .none }
        lastLoadFailed = true
        if closed || errorShown { return .none }
        if !retryable { return giveUp() }
        let first = firstFailure ?? now
        firstFailure = first
        switch DashboardLoadPolicy.decide(domain: domain, code: code, attempt: attempt,
                                          elapsed: now.timeIntervalSince(first)) {
        case .giveUp:
            return giveUp()
        case .retry(let delay):
            attempt += 1
            retryPending = true
            return attempt == 1 ? .showPlaceholderAndSchedule(delay) : .scheduleRetry(delay)
        }
    }

    private mutating func giveUp() -> DashboardLoadAction {
        retryPending = false
        errorShown = true
        return .showError
    }

    /// didFinish. A finished about: page (placeholder, error) is not a success.
    mutating func onFinish(url: URL?) {
        if url?.scheme?.lowercased() == "about" { return }
        lastLoadFailed = false
        attempt = 0
        firstFailure = nil
        retryPending = false
        errorShown = false
    }

    /// The retry timer fired.
    mutating func onRetryTimer() -> DashboardLoadAction {
        guard retryPending, !closed else { return .none }
        retryPending = false
        return .load
    }

    /// The window was closed (hidden).
    mutating func onClose() -> DashboardLoadAction {
        closed = true
        retryPending = false
        return .stopLoading
    }
}
