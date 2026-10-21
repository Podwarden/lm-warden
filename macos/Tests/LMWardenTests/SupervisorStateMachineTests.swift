import XCTest
@testable import LMWarden

/// Tests for `SupervisorStateMachine`, the pure transition function the
/// `ServerSupervisor` (Metal design §2) drives with `RestartBackoff`.
///
/// The state machine is pure: no timers, no async, no I/O. It owns the state
/// (`state`, `failures`) and a single `policy: RestartBackoff`, and each
/// `handle(_:)` returns the new state so the caller can observe the transition.
final class SupervisorStateMachineTests: XCTestCase {
    private let accuracy = 1e-9

    // (c) stopped + startRequested -> starting, failures reset to 0.

    func testStartFromStopped() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        XCTAssertEqual(machine.handle(.startRequested), .starting)
        XCTAssertEqual(machine.state, .starting)
        XCTAssertEqual(machine.failures, 0)
    }

    // (c) failed + startRequested -> starting, failures reset to 0.

    func testStartFromFailedResetsFailures() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        // Drive it into .failed first (6 short exits, one per restart attempt).
        machine.handle(.startRequested)
        for _ in 0..<6 {
            machine.handle(.exited(uptime: 0))
            if case .failed = machine.state { break }
            machine.handle(.backoffElapsed)
        }
        XCTAssertEqual(machine.state, .failed(failures: 6))
        // A fresh start from .failed returns to .starting and zeroes failures.
        XCTAssertEqual(machine.handle(.startRequested), .starting)
        XCTAssertEqual(machine.failures, 0)
    }

    // (c) starting + becameHealthy -> running.

    func testBecameHealthyFromStarting() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        XCTAssertEqual(machine.handle(.becameHealthy), .running)
        XCTAssertEqual(machine.state, .running)
    }

    // (c) starting + exited -> failures = policy.failures(afterExitWithPrevious:uptime:).
    // First exit (previous 0, short) -> failures 1 -> backingOff(delay 1).

    func testExitedFromStartingCountsFirstFailure() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        XCTAssertEqual(machine.handle(.exited(uptime: 0)), .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(machine.failures, 1)
    }

    // (c) running + exited -> same failure accounting as starting + exited.
    // A short exit after having been healthy increments the (zero) streak to 1.

    func testExitedFromRunningCountsFailure() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        machine.handle(.becameHealthy)
        XCTAssertEqual(machine.handle(.exited(uptime: 0)), .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(machine.failures, 1)
    }

    // (d) give-up boundary, lower leg: the 5th short exit is still within the
    // limit (maxConsecutiveFailures = 5) -> backingOff with delay 16.

    func testFifthShortExitBacksOffAtDelay16() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        for _ in 0..<4 {
            machine.handle(.exited(uptime: 0))
            machine.handle(.backoffElapsed)
        }
        XCTAssertEqual(machine.failures, 4)
        // 5th short exit: failures 4 -> 5, shouldGiveUp(5) is false -> back off.
        XCTAssertEqual(machine.handle(.exited(uptime: 0)), .backingOff(failures: 5, delay: 16))
        XCTAssertEqual(machine.failures, 5)
    }

    // (d) give-up boundary, upper leg: the 6th short exit exceeds the limit ->
    // failed(failures: 6), no backoff.

    func testSixthShortExitFails() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        for _ in 0..<6 {
            machine.handle(.exited(uptime: 0))
            if case .failed = machine.state { break }
            machine.handle(.backoffElapsed)
        }
        XCTAssertEqual(machine.state, .failed(failures: 6))
        XCTAssertEqual(machine.failures, 6)
    }

    // (d) stable-uptime exit after 4 failures: an exit with uptime 60 (>=
    // stableUptime 60) starts a fresh streak at 1 -> backingOff(delay 1), not
    // a continuation to 5.

    func testStableUptimeExitAfterFourFailuresStartsFreshStreak() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        for _ in 0..<4 {
            machine.handle(.exited(uptime: 0))
            machine.handle(.backoffElapsed)
        }
        XCTAssertEqual(machine.failures, 4)
        // uptime 60 == stableUptime -> failures(afterExitWithPrevious: 4, uptime: 60) = 1.
        XCTAssertEqual(machine.handle(.exited(uptime: 60)), .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(machine.failures, 1)
    }

    // (d) backoffElapsed from .backingOff -> starting (the restart attempt).

    func testBackoffElapsedResumesStart() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        machine.handle(.exited(uptime: 0))
        XCTAssertEqual(machine.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(machine.handle(.backoffElapsed), .starting)
    }

    // (c) ANY state + stopRequested -> stopped, failures reset to 0.
    // From .backingOff.

    func testStopRequestedFromBackingOff() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        machine.handle(.exited(uptime: 0))
        XCTAssertEqual(machine.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(machine.handle(.stopRequested), .stopped)
        XCTAssertEqual(machine.failures, 0)
    }

    // (d) stopRequested from .failed -> stopped, failures reset to 0.

    func testStopRequestedFromFailed() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        for _ in 0..<6 {
            machine.handle(.exited(uptime: 0))
            if case .failed = machine.state { break }
            machine.handle(.backoffElapsed)
        }
        XCTAssertEqual(machine.state, .failed(failures: 6))
        XCTAssertEqual(machine.handle(.stopRequested), .stopped)
        XCTAssertEqual(machine.failures, 0)
    }

    // (d) stopRequested from .running -> stopped.

    func testStopRequestedFromRunning() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        machine.handle(.becameHealthy)
        XCTAssertEqual(machine.handle(.stopRequested), .stopped)
        XCTAssertEqual(machine.failures, 0)
    }

    // (d) ignored pairs leave the state unchanged.
    // stopped + becameHealthy: no start in flight, so health is a no-op.

    func testStoppedIgnoresBecameHealthy() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        XCTAssertEqual(machine.state, .stopped)
        XCTAssertEqual(machine.handle(.becameHealthy), .stopped)
        XCTAssertEqual(machine.state, .stopped)
    }

    // (d) ignored pairs leave the state unchanged.
    // running + backoffElapsed: no backoff pending, so the timer firing is a no-op.

    func testRunningIgnoresBackoffElapsed() {
        var machine = SupervisorStateMachine(policy: RestartBackoff())
        machine.handle(.startRequested)
        machine.handle(.becameHealthy)
        XCTAssertEqual(machine.state, .running)
        XCTAssertEqual(machine.handle(.backoffElapsed), .running)
        XCTAssertEqual(machine.state, .running)
    }

    // (c) every (state, event) pair outside the transition table leaves both
    // the state and the failure streak unchanged. Covers the stray
    // startRequested (it must not reset a backoff streak) and an exit observed
    // outside starting/running (it must not count a failure).

    func testUndefinedPairsLeaveStateAndFailuresUnchanged() {
        let policy = RestartBackoff()
        func reach(_ target: SupervisorState) -> SupervisorStateMachine {
            var machine = SupervisorStateMachine(policy: policy)
            switch target {
            case .stopped:
                break
            case .starting:
                machine.handle(.startRequested)
            case .running:
                machine.handle(.startRequested)
                machine.handle(.becameHealthy)
            case .backingOff:
                machine.handle(.startRequested)
                machine.handle(.exited(uptime: 0))
                machine.handle(.backoffElapsed)
                machine.handle(.exited(uptime: 0))
            case .failed:
                machine.handle(.startRequested)
                for _ in 0..<6 {
                    machine.handle(.exited(uptime: 0))
                    if case .failed = machine.state { break }
                    machine.handle(.backoffElapsed)
                }
            }
            XCTAssertEqual(machine.state, target)
            return machine
        }
        let ignored: [(SupervisorState, [SupervisorEvent])] = [
            (.stopped, [.becameHealthy, .exited(uptime: 0), .backoffElapsed]),
            (.starting, [.startRequested, .backoffElapsed]),
            (.running, [.startRequested, .becameHealthy, .backoffElapsed]),
            (.backingOff(failures: 2, delay: 2), [.startRequested, .becameHealthy, .exited(uptime: 0)]),
            (.failed(failures: 6), [.becameHealthy, .exited(uptime: 0), .backoffElapsed]),
        ]
        for (state, events) in ignored {
            for event in events {
                var machine = reach(state)
                let before = machine.failures
                XCTAssertEqual(machine.handle(event), state, "\(state) + \(event)")
                XCTAssertEqual(machine.failures, before, "\(state) + \(event) changed failures")
            }
        }
    }
}
