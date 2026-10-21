import XCTest
@testable import LMWarden

/// Pins the exact copy, arming, hysteresis and rate limits of
/// `NotificationPolicy`. On the base this file does not compile because the
/// types below do not exist yet — that is the accepted red for a new-type task.
final class NotificationPolicyTests: XCTestCase {
    private static let t0 = Date(timeIntervalSinceReferenceDate: 1_000_000)

    private func at(_ seconds: TimeInterval) -> Date {
        Self.t0.addingTimeInterval(seconds)
    }

    private func notify(
        _ policy: inout NotificationPolicy,
        _ event: ShellEvent,
        _ seconds: TimeInterval
    ) -> ShellNotification? {
        policy.handle(event, now: at(seconds))
    }

    // MARK: - (b) Restarted

    func testFirstStartIsSilent() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.running), 1))
    }

    func testRestartAfterBackoffOneSecond() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 1)), 0))
        let n = notify(&p, .supervisor(.running), 2)
        XCTAssertEqual(n?.kind, .restarted)
        XCTAssertEqual(n?.title, "Server restarted")
        XCTAssertEqual(n?.body, "The server stopped unexpectedly and was restarted after 1 second. Requests in flight were lost.")
    }

    func testRestartAfterBackoffTwoSeconds() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 2)), 0))
        let n = notify(&p, .supervisor(.running), 3)
        XCTAssertEqual(n?.kind, .restarted)
        XCTAssertEqual(n?.body, "The server stopped unexpectedly and was restarted after 2 seconds. Requests in flight were lost.")
    }

    func testStoppedClearsRememberedDelay() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 30)), 0))
        XCTAssertNil(notify(&p, .supervisor(.stopped), 1))
        XCTAssertNil(notify(&p, .supervisor(.running), 2))
    }

    // MARK: - (c) Gave up

    func testGaveUpFiresOncePerEntry() {
        var p = NotificationPolicy()
        let first = notify(&p, .supervisor(.failed(failures: 3)), 0)
        XCTAssertEqual(first?.kind, .gaveUp)
        XCTAssertEqual(first?.title, "Server stopped for good")
        XCTAssertEqual(first?.body, "It failed 3 times in a row. Open the menu bar menu to see the log or start it again.")
        let again = notify(&p, .supervisor(.failed(failures: 3)), 10)
        XCTAssertNil(again)
    }

    func testGaveUpSingular() {
        var p = NotificationPolicy()
        let n = notify(&p, .supervisor(.failed(failures: 1)), 0)
        XCTAssertEqual(n?.body, "It failed 1 time in a row. Open the menu bar menu to see the log or start it again.")
    }

    func testGaveUpNeverRateLimited() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .supervisor(.failed(failures: 5)), 0))
        XCTAssertNil(notify(&p, .supervisor(.stopped), 1))
        let again = notify(&p, .supervisor(.failed(failures: 5)), 2)
        XCTAssertEqual(again?.body, "It failed 5 times in a row. Open the menu bar menu to see the log or start it again.")
    }

    // MARK: - (d) Memory

    func testMemoryAtExactly90Fires() {
        var p = NotificationPolicy()
        // 9_000_000_000 used of 10_000_000_000 usable is exactly the 0.90 bound.
        let n = notify(&p, .memory(usedBytes: 9_000_000_000, usableBytes: 10_000_000_000), 0)
        XCTAssertEqual(n?.kind, .memoryTight)
        XCTAssertEqual(n?.title, "Memory is tight")
        XCTAssertEqual(n?.body, "Loaded models use 9.0 of 10.0 GB. New loads may be refused. Unload a model in the dashboard.")
    }

    func testMemoryJustBelow90DoesNotFire() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .memory(usedBytes: 89, usableBytes: 100), 0))
    }

    func testMemoryUsableZeroNeverFires() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .memory(usedBytes: 100, usableBytes: 0), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 100, usableBytes: 0), 1))
    }

    func testMemoryDisarmedUntilBelow80() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 85, usableBytes: 100), 1))
    }

    func testMemoryRearmAtJustBelow80() {
        // minimumInterval 0 isolates the re-arm (used < 0.80 * usable) from the
        // rate limit, which the (f) section exercises at real intervals.
        var p = NotificationPolicy(minimumInterval: 0)
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 79, usableBytes: 100), 1))
        let refire = notify(&p, .memory(usedBytes: 95, usableBytes: 100), 2)
        XCTAssertEqual(refire?.kind, .memoryTight)
    }

    func testMemoryAtExactly80DoesNotRearm() {
        // minimumInterval 0: without it the refire at 2 is rate-limited and
        // this leg could not tell `< 0.80` from `<= 0.80`.
        var p = NotificationPolicy(minimumInterval: 0)
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 80, usableBytes: 100), 1))
        XCTAssertNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 2))
    }

    // MARK: - (e) Thermal

    func testThermalFairToSeriousFires() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .thermal(.fair), 0))
        let n = notify(&p, .thermal(.serious), 1)
        XCTAssertEqual(n?.kind, .thermal)
        XCTAssertEqual(n?.title, "Mac is running hot")
        XCTAssertEqual(n?.body, "macOS is slowing the chip to cool down. Responses will be slower until it cools.")
    }

    func testThermalSeriousToCriticalDoesNotFire() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .thermal(.fair), 0))
        XCTAssertNotNil(notify(&p, .thermal(.serious), 1))
        XCTAssertNil(notify(&p, .thermal(.critical), 2))
    }

    func testThermalRearmedByFair() {
        // minimumInterval 0 isolates the re-arm behaviour from the rate limit,
        // which is exercised separately in the (f) section.
        var p = NotificationPolicy(minimumInterval: 0)
        XCTAssertNil(notify(&p, .thermal(.fair), 0))
        XCTAssertNotNil(notify(&p, .thermal(.serious), 1))
        XCTAssertNil(notify(&p, .thermal(.fair), 2))
        let refire = notify(&p, .thermal(.serious), 3)
        XCTAssertEqual(refire?.kind, .thermal)
    }

    func testThermalNominalToCriticalFires() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .thermal(.nominal), 0))
        XCTAssertEqual(notify(&p, .thermal(.critical), 1)?.kind, .thermal)
    }

    // MARK: - (e) Battery

    func testBatteryACToBatteryWithoutModelSilent() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: false), 0))
        XCTAssertNil(notify(&p, .power(onBattery: true, modelLoaded: false), 1))
    }

    func testBatteryACToBatteryWithModelFires() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: true), 0))
        let n = notify(&p, .power(onBattery: true, modelLoaded: true), 1)
        XCTAssertEqual(n?.kind, .battery)
        XCTAssertEqual(n?.title, "Running on battery")
        XCTAssertEqual(n?.body, "Models drain the battery quickly. Connect power for long jobs.")
    }

    func testBatteryBatteryToACDoesNotFire() {
        var p = NotificationPolicy()
        // Start in a known AC state so the return to AC is an explicit
        // battery -> AC change, which never fires.
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: true), 0))
        XCTAssertNotNil(notify(&p, .power(onBattery: true, modelLoaded: true), 1))
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: true), 2))
    }

    // MARK: - (f) Rate limits

    func testMemoryRateLimitedAtIntervalMinusOne() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        // Disarm, re-arm, then refire just before the interval elapses.
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 1))
        XCTAssertNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 599))
    }

    func testMemoryFiresAtExactlyInterval() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 1))
        let refire = notify(&p, .memory(usedBytes: 95, usableBytes: 100), 600)
        XCTAssertEqual(refire?.kind, .memoryTight)
    }

    func testIntervalMeasuredFromLastFiredNotLastSuppressed() {
        // The suppressed attempt at 599 must not move the anchor: the next
        // armed attempt exactly 600 s after the FIRED one fires.
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 1))
        XCTAssertNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 599))
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 599.5))
        XCTAssertEqual(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 600)?.kind, .memoryTight)
    }

    func testThermalRateLimitedAtIntervalMinusOne() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .thermal(.serious), 0))
        XCTAssertNil(notify(&p, .thermal(.fair), 1))
        XCTAssertNil(notify(&p, .thermal(.serious), 599))
    }

    func testBatteryRateLimitedAtIntervalMinusOne() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .power(onBattery: true, modelLoaded: true), 0))
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: true), 1))
        XCTAssertNil(notify(&p, .power(onBattery: true, modelLoaded: true), 599))
    }

    func testRestartRateLimitedAtRestartIntervalMinusOne() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 1)), 0))
        XCTAssertNotNil(notify(&p, .supervisor(.running), 1))
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 1)), 2))
        XCTAssertNil(notify(&p, .supervisor(.running), 60))
    }

    func testRestartFiresAtExactlyRestartInterval() {
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 1)), 0))
        XCTAssertNotNil(notify(&p, .supervisor(.running), 1))
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 1)), 2))
        let refire = notify(&p, .supervisor(.running), 61)
        XCTAssertEqual(refire?.kind, .restarted)
    }

    func testSuppressedEventStillUpdatesArming() {
        // A thermal event SUPPRESSED by the rate limit still records its level:
        // fire at 0, cool at 1, hot again at 2 (suppressed, inside the window).
        // At 600 the window is open, but critical after serious is no change
        // from cool, so nothing fires.
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .thermal(.serious), 0))
        XCTAssertNil(notify(&p, .thermal(.fair), 1))
        XCTAssertNil(notify(&p, .thermal(.serious), 2))
        XCTAssertNil(notify(&p, .thermal(.critical), 600))
    }

    func testSuppressedMemoryEventStillDisarms() {
        // Oscillating pressure: the suppressed 95 % at 2 disarms, so staying
        // at 95 % when the window opens at 600 stays silent until < 80 %.
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 0))
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 1))
        XCTAssertNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 2))
        XCTAssertNil(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 600))
        XCTAssertNil(notify(&p, .memory(usedBytes: 70, usableBytes: 100), 601))
        XCTAssertEqual(notify(&p, .memory(usedBytes: 95, usableBytes: 100), 602)?.kind, .memoryTight)
    }

    func testSuppressedBatteryEventStillRecordsBattery() {
        var p = NotificationPolicy()
        XCTAssertNotNil(notify(&p, .power(onBattery: true, modelLoaded: true), 0))
        XCTAssertNil(notify(&p, .power(onBattery: false, modelLoaded: true), 1))
        XCTAssertNil(notify(&p, .power(onBattery: true, modelLoaded: true), 2))
        // Still on battery when the window opens: no AC -> battery change.
        XCTAssertNil(notify(&p, .power(onBattery: true, modelLoaded: true), 600))
    }

    // MARK: - Supervisor sequences the state machine produces

    func testRestartThroughStartingFires() {
        // backingOff -> starting (backoffElapsed) -> running (becameHealthy).
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 1, delay: 2)), 0))
        XCTAssertNil(notify(&p, .supervisor(.starting), 2))
        XCTAssertEqual(notify(&p, .supervisor(.running), 3)?.body,
                       "The server stopped unexpectedly and was restarted after 2 seconds. Requests in flight were lost.")
    }

    func testGaveUpFiresAgainAfterStartFromFailed() {
        // failed -> (start it again) starting -> failed is a new entry.
        var p = NotificationPolicy()
        XCTAssertEqual(notify(&p, .supervisor(.failed(failures: 5)), 0)?.kind, .gaveUp)
        XCTAssertNil(notify(&p, .supervisor(.starting), 1))
        XCTAssertEqual(notify(&p, .supervisor(.failed(failures: 5)), 2)?.kind, .gaveUp)
    }

    func testStartAfterGaveUpIsNotARestart() {
        // backingOff -> starting -> failed, then the user starts it again:
        // that .running is not an unexpected restart.
        var p = NotificationPolicy()
        XCTAssertNil(notify(&p, .supervisor(.backingOff(failures: 4, delay: 8)), 0))
        XCTAssertNil(notify(&p, .supervisor(.starting), 8))
        XCTAssertEqual(notify(&p, .supervisor(.failed(failures: 5)), 9)?.kind, .gaveUp)
        XCTAssertNil(notify(&p, .supervisor(.starting), 100))
        XCTAssertNil(notify(&p, .supervisor(.running), 101))
    }

    // MARK: - Equatable conformance

    func testNotificationKindEquatable() {
        XCTAssertEqual(ShellNotification(kind: .restarted, title: "a", body: "b").kind, .restarted)
        XCTAssertNotEqual(ShellNotification(kind: .restarted, title: "a", body: "b").kind, .gaveUp)
    }
}
