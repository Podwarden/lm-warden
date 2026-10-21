import XCTest
@testable import LMWarden

/// Retry and reload rules for the dashboard window. Pure Foundation.
final class DashboardLoadPolicyTests: XCTestCase {
    private func decide(_ code: Int, attempt: Int = 0, elapsed: TimeInterval = 0,
                        domain: String = NSURLErrorDomain) -> DashboardRetry {
        DashboardLoadPolicy.decide(domain: domain, code: code, attempt: attempt, elapsed: elapsed)
    }

    func testConnectionRefusedRetriesWithDoublingDelay() {
        let delays = (0..<6).map { decide(NSURLErrorCannotConnectToHost, attempt: $0) }
        XCTAssertEqual(delays, [.retry(after: 0.5), .retry(after: 1), .retry(after: 2),
                                .retry(after: 4), .retry(after: 4), .retry(after: 4)])
    }

    func testOtherTransientErrorsRetry() {
        for code in [NSURLErrorTimedOut, NSURLErrorNetworkConnectionLost, NSURLErrorCannotFindHost] {
            XCTAssertEqual(decide(code), .retry(after: 0.5))
        }
    }

    func testCancelledAndBadURLDoNotRetry() {
        XCTAssertEqual(decide(NSURLErrorCancelled), .giveUp)
        XCTAssertEqual(decide(NSURLErrorBadURL), .giveUp)
        XCTAssertEqual(decide(NSURLErrorCannotConnectToHost, domain: "WebKitErrorDomain"), .giveUp)
    }

    func testDeadlineStopsRetrying() {
        XCTAssertEqual(decide(NSURLErrorCannotConnectToHost, attempt: 20, elapsed: 179.9), .retry(after: 4))
        XCTAssertEqual(decide(NSURLErrorCannotConnectToHost, attempt: 20, elapsed: 180), .giveUp)
    }

    func testShowReloadsAfterFailedLoadOnly() {
        XCTAssertTrue(DashboardLoadPolicy.shouldReloadOnShow(isNewWindow: true, urlChanged: false, lastLoadFailed: false))
        XCTAssertTrue(DashboardLoadPolicy.shouldReloadOnShow(isNewWindow: false, urlChanged: true, lastLoadFailed: false))
        XCTAssertTrue(DashboardLoadPolicy.shouldReloadOnShow(isNewWindow: false, urlChanged: false, lastLoadFailed: true))
        XCTAssertFalse(DashboardLoadPolicy.shouldReloadOnShow(isNewWindow: false, urlChanged: false, lastLoadFailed: false))
    }
}

/// The dashboard window's load state machine.
final class DashboardLoadStateTests: XCTestCase {
    private let t0 = Date(timeIntervalSince1970: 1_000)
    private let refused = NSURLErrorCannotConnectToHost
    private let dashboardURL = URL(string: "http://127.0.0.1:8080/ui/")!
    private let blank = URL(string: "about:blank")!

    private func fail(_ s: inout DashboardLoadState, _ code: Int, at offset: TimeInterval = 0,
                      url: URL? = nil, domain: String = NSURLErrorDomain) -> DashboardLoadAction {
        s.onFailure(domain: domain, code: code, failingURL: url, now: t0.addingTimeInterval(offset))
    }

    private func started() -> DashboardLoadState {
        var s = DashboardLoadState()
        XCTAssertEqual(s.onShow(isNewWindow: true, urlChanged: true), .load)
        return s
    }

    func testRefusedShowsPlaceholderThenRetriesThenSuccessResets() {
        var s = started()
        XCTAssertEqual(fail(&s, refused), .showPlaceholderAndSchedule(0.5))
        XCTAssertEqual(s.onRetryTimer(), .load)
        XCTAssertEqual(fail(&s, refused, at: 1), .scheduleRetry(1))
        XCTAssertEqual(s.onRetryTimer(), .load)
        s.onFinish(url: dashboardURL)
        XCTAssertFalse(s.lastLoadFailed)
        XCTAssertEqual(s.attempt, 0)
        XCTAssertNil(s.firstFailure)
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .none)
    }

    func testPlaceholderFinishIsNotSuccessAndItsFailureIsIgnored() {
        var s = started()
        _ = fail(&s, refused)
        s.onFinish(url: blank)
        XCTAssertTrue(s.lastLoadFailed)
        XCTAssertEqual(fail(&s, NSURLErrorCancelled, url: blank), .none)
        XCTAssertEqual(s.attempt, 1)
    }

    func testCancelledIsIgnoredAndDoesNotStartTheClock() {
        var s = started()
        XCTAssertEqual(fail(&s, NSURLErrorCancelled), .none)
        XCTAssertFalse(s.lastLoadFailed)
        XCTAssertNil(s.firstFailure)
        // First real failure much later still gets a full deadline.
        XCTAssertEqual(fail(&s, refused, at: 500), .showPlaceholderAndSchedule(0.5))
    }

    func testCancelledAndInterruptedAfterSuccessDoNotMarkFailed() {
        var s = started()
        s.onFinish(url: dashboardURL)
        XCTAssertEqual(fail(&s, NSURLErrorCancelled), .none)
        XCTAssertEqual(fail(&s, 102, domain: "WebKitErrorDomain"), .none)
        XCTAssertFalse(s.lastLoadFailed)
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .none)
    }

    func testCloseCancelsAndBlocksFurtherScheduling() {
        var s = started()
        XCTAssertEqual(fail(&s, refused), .showPlaceholderAndSchedule(0.5))
        XCTAssertEqual(s.onClose(), .stopLoading)
        XCTAssertEqual(s.onRetryTimer(), .none)
        XCTAssertEqual(fail(&s, refused, at: 1), .none)
        XCTAssertTrue(s.lastLoadFailed)
        // Reopening reloads.
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .load)
    }

    func testShowReloadsAfterFailureNotAfterSuccess() {
        var s = started()
        _ = fail(&s, refused)
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .load)
        s.onFinish(url: dashboardURL)
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .none)
    }

    func testDeadlineShowsErrorPageAndStaysFailed() {
        var s = started()
        XCTAssertEqual(fail(&s, refused), .showPlaceholderAndSchedule(0.5))
        XCTAssertEqual(fail(&s, refused, at: 179), .scheduleRetry(1))
        XCTAssertEqual(fail(&s, refused, at: 181), .showError)
        XCTAssertEqual(s.onRetryTimer(), .none)
        XCTAssertTrue(s.lastLoadFailed)
        XCTAssertEqual(fail(&s, refused, at: 200), .none)
        XCTAssertEqual(s.onShow(isNewWindow: false, urlChanged: false), .load)
    }

    func testNonRetryableErrorAfterRetriesShowsErrorPage() {
        var s = started()
        _ = fail(&s, refused)
        XCTAssertEqual(fail(&s, NSURLErrorBadURL), .showError)
    }

    func testNonRetryableErrorBeforeAnyRetryLeavesStateAlone() {
        var s = started()
        XCTAssertEqual(fail(&s, NSURLErrorBadURL), .none)
        XCTAssertFalse(s.lastLoadFailed)
    }

    func testServerRunningReloadsOnlyAFailedLoad() {
        var s = started()
        XCTAssertEqual(s.onServerRunning(), .none)
        _ = fail(&s, refused)
        XCTAssertEqual(s.onServerRunning(), .load)
        XCTAssertEqual(s.onRetryTimer(), .none)
    }
}
