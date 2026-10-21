import XCTest
@testable import LMWarden

/// Pins `ShellEventTracker`: one `.supervisor` event per state change, one
/// `.memory` event per new non-nil status, in that order — and that the
/// events it emits drive `NotificationPolicy` to the right notifications.
final class ShellEventTrackerTests: XCTestCase {
    private func status(used: Int64, usable: Int64 = 1000, inFlight: Int = 0) -> ShellStatus {
        ShellStatus(
            loadedModels: [LoadedModel(name: "m", backend: "mlx", bytes: used)],
            usedBytes: used,
            usableBytes: usable,
            inFlight: inFlight == 0 ? [] : [InFlightGroup(who: "web", model: "m", requests: inFlight)]
        )
    }

    // MARK: - Events

    func testFirstSnapshotReportsTheState() {
        var t = ShellEventTracker()
        XCTAssertEqual(t.events(state: .stopped, status: nil), [.supervisor(.stopped)])
    }

    func testUnchangedSnapshotReportsNothing() {
        var t = ShellEventTracker()
        _ = t.events(state: .running, status: status(used: 100))
        XCTAssertEqual(t.events(state: .running, status: status(used: 100)), [])
    }

    func testEachStateChangeIsReportedOnce() {
        var t = ShellEventTracker()
        let states: [SupervisorState] = [
            .starting, .running, .backingOff(failures: 1, delay: 1), .starting, .running, .failed(failures: 5), .stopped,
        ]
        var seen: [ShellEvent] = []
        for state in states {
            seen += t.events(state: state, status: nil)
            seen += t.events(state: state, status: nil)
        }
        XCTAssertEqual(seen, states.map { .supervisor($0) })
    }

    func testNewStatusReportsMemoryWithItsBytes() {
        var t = ShellEventTracker()
        _ = t.events(state: .running, status: nil)
        XCTAssertEqual(
            t.events(state: .running, status: status(used: 300, usable: 1000)),
            [.memory(usedBytes: 300, usableBytes: 1000)]
        )
    }

    func testAStatusChangeWithTheSameBytesStillReportsMemory() {
        var t = ShellEventTracker()
        _ = t.events(state: .running, status: status(used: 300))
        XCTAssertEqual(
            t.events(state: .running, status: status(used: 300, inFlight: 2)),
            [.memory(usedBytes: 300, usableBytes: 1000)]
        )
    }

    func testNilStatusReportsNothingAndTheSameStatusAfterNilReportsAgain() {
        var t = ShellEventTracker()
        _ = t.events(state: .running, status: status(used: 300))
        XCTAssertEqual(t.events(state: .running, status: nil), [])
        XCTAssertEqual(
            t.events(state: .running, status: status(used: 300)),
            [.memory(usedBytes: 300, usableBytes: 1000)]
        )
    }

    func testStateBeforeMemoryWhenBothChange() {
        var t = ShellEventTracker()
        XCTAssertEqual(
            t.events(state: .running, status: status(used: 10)),
            [.supervisor(.running), .memory(usedBytes: 10, usableBytes: 1000)]
        )
    }

    // MARK: - Mapping to NotificationPolicy

    private func notifications(_ snapshots: [(SupervisorState, ShellStatus?)]) -> [NotificationKind] {
        var tracker = ShellEventTracker()
        var policy = NotificationPolicy()
        var kinds: [NotificationKind] = []
        var now = Date(timeIntervalSinceReferenceDate: 1_000_000)
        for (state, status) in snapshots {
            for event in tracker.events(state: state, status: status) {
                if let n = policy.handle(event, now: now) {
                    kinds.append(n.kind)
                }
            }
            now = now.addingTimeInterval(3600)
        }
        return kinds
    }

    func testFirstStartAndRepeatedRefreshesAreSilent() {
        XCTAssertEqual(notifications([(.stopped, nil), (.starting, nil), (.running, nil), (.running, nil)]), [])
    }

    func testCrashAndRestartNotifiesRestartedOnce() {
        XCTAssertEqual(
            notifications([
                (.running, nil),
                (.backingOff(failures: 1, delay: 2), nil),
                (.backingOff(failures: 1, delay: 2), nil),
                (.starting, nil),
                (.running, nil),
                (.running, nil),
            ]),
            [.restarted]
        )
    }

    func testGivingUpNotifiesOnce() {
        XCTAssertEqual(
            notifications([(.running, nil), (.failed(failures: 5), nil), (.failed(failures: 5), nil)]),
            [.gaveUp]
        )
    }

    func testMemoryCrossingNinetyPercentNotifiesOnce() {
        XCTAssertEqual(
            notifications([
                (.running, status(used: 500)),
                (.running, status(used: 950)),
                (.running, status(used: 950, inFlight: 1)),
                (.running, status(used: 960)),
            ]),
            [.memoryTight]
        )
    }
}
