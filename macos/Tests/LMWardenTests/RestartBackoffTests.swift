import XCTest
@testable import LMWarden

final class RestartBackoffTests: XCTestCase {
    /// A negative or zero failure count produces zero delay: no failure means
    /// nothing to wait for.
    func testDelayZeroForNoFailures() {
        let backoff = RestartBackoff()
        XCTAssertEqual(backoff.delay(afterFailures: 0), 0, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: -1), 0, accuracy: 1e-9)
    }

    /// With defaults (base 1, cap 30) the delay doubles on each consecutive
    /// failure, starting at `base` after the first failure.
    func testDelayDoublesFromBase() {
        let backoff = RestartBackoff()
        XCTAssertEqual(backoff.delay(afterFailures: 1), 1, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 2), 2, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 3), 4, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 4), 8, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 5), 16, accuracy: 1e-9)
    }

    /// With a custom `cap` the delay reaches exactly `cap` at the boundary
    /// and stays there afterwards.
    func testDelayCapsAtExactBound() {
        let backoff = RestartBackoff(base: 1, cap: 16, stableUptime: 60, maxConsecutiveFailures: 5)
        XCTAssertEqual(backoff.delay(afterFailures: 5), 16, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 6), 16, accuracy: 1e-9)
    }

    /// With defaults the cap of 30 is first reached at 6 failures (16 < 32
    /// would exceed it) and holds through any count beyond.
    func testDelayCapsJustPast() {
        let backoff = RestartBackoff()
        XCTAssertEqual(backoff.delay(afterFailures: 6), 30, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 7), 30, accuracy: 1e-9)
    }

    /// The cap check is done before multiplying so that huge failure counts
    /// (including Int.max) do not overflow `base * pow(2, n)`.
    func testDelayHugeFailureCountDoesNotOverflow() {
        let backoff = RestartBackoff()
        XCTAssertEqual(backoff.delay(afterFailures: 1000), 30, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: Int.max), 30, accuracy: 1e-9)
    }

    /// An uptime at or above `stableUptime` means the process was healthy
    /// long enough, so the next exit starts a fresh streak at 1 regardless of
    /// how many failures preceded it.
    func testStableUptimeResetsStreak() {
        let backoff = RestartBackoff()
        // Boundary: uptime exactly stableUptime resets the streak.
        XCTAssertEqual(backoff.failures(afterExitWithPrevious: 4, uptime: 60), 1)
        // Far past the boundary still resets.
        XCTAssertEqual(backoff.failures(afterExitWithPrevious: 0, uptime: 3600), 1)
    }

    /// An uptime just below `stableUptime` does not reset the streak: the
    /// failure count is extended by one (and a negative previous is clamped
    /// to zero before incrementing).
    func testShortUptimeExtendsStreak() {
        let backoff = RestartBackoff()
        XCTAssertEqual(backoff.failures(afterExitWithPrevious: 2, uptime: 59.999), 3)
        XCTAssertEqual(backoff.failures(afterExitWithPrevious: 0, uptime: 0), 1)
        XCTAssertEqual(backoff.failures(afterExitWithPrevious: -3, uptime: 0), 1)
    }

    /// `shouldGiveUp` triggers only when consecutive failures exceed
    /// `maxConsecutiveFailures`: 5 is still within the limit, 6 is past it.
    func testGiveUpBoundary() {
        let backoff = RestartBackoff()
        XCTAssertFalse(backoff.shouldGiveUp(failures: 5))
        XCTAssertTrue(backoff.shouldGiveUp(failures: 6))
        XCTAssertFalse(backoff.shouldGiveUp(failures: 0))
    }

    /// Custom `init` parameters are stored and used by the policy functions.
    func testCustomInitStoresValues() {
        let backoff = RestartBackoff(base: 2, cap: 10, stableUptime: 5, maxConsecutiveFailures: 1)
        XCTAssertEqual(backoff.delay(afterFailures: 3), 8, accuracy: 1e-9)
        XCTAssertEqual(backoff.delay(afterFailures: 4), 10, accuracy: 1e-9)
        XCTAssertTrue(backoff.shouldGiveUp(failures: 2))
    }
}
