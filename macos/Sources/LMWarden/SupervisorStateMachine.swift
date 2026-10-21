// SupervisorStateMachine — pure state transitions for the ServerSupervisor
// (Metal design §2), driven by a `RestartBackoff` policy.
//
// The `ServerSupervisor` actor owns the `Process` and the actual waiting; this
// type owns only the transition table. It is intentionally free of Foundation
// timers, actors, and async: the supervisor feeds it `SupervisorEvent`s (from
// the process's exit handler and its health poller) and reads back the new
// `SupervisorState`. `handle` never throws — an event the current state does
// not react to is simply ignored.
import Foundation

/// The supervisor's lifecycle state.
enum SupervisorState: Equatable {
    case stopped
    case starting
    case running
    case backingOff(failures: Int, delay: TimeInterval)
    case failed(failures: Int)
}

/// An input the supervisor observes and feeds to the state machine.
enum SupervisorEvent: Equatable {
    case startRequested
    case stopRequested
    case becameHealthy
    case exited(uptime: TimeInterval)
    case backoffElapsed
}

/// Pure transition function for the `ServerSupervisor`.
///
/// The machine holds its own `state` and `failures` and a shared `policy`;
/// `handle(_:)` mutates both and returns the new state so the caller can
/// observe the transition (e.g. to schedule a `backoffElapsed` timer for the
/// returned `delay`). Every (state, event) pair that is not a defined
/// transition leaves the state unchanged.
struct SupervisorStateMachine {
    let policy: RestartBackoff
    private(set) var state: SupervisorState = .stopped
    private(set) var failures: Int = 0

    init(policy: RestartBackoff) {
        self.policy = policy
    }

    /// Apply `event` to the current state and return the new state.
    ///
    /// - `stopRequested` is handled first: from ANY state it stops and
    ///   zeroes the failure streak.
    /// - `startRequested` is a no-op unless already stopped/failed, so a
    ///   stray start while already running never resets the streak.
    @discardableResult
    mutating func handle(_ event: SupervisorEvent) -> SupervisorState {
        switch event {
        case .stopRequested:
            state = .stopped
            failures = 0

        case .startRequested:
            if case .stopped = state {
                state = .starting
                failures = 0
            } else if case .failed = state {
                state = .starting
                failures = 0
            }

        case .becameHealthy:
            if case .starting = state {
                state = .running
            }

        case .exited(let uptime):
            if case .starting = state {
                applyExit(uptime)
            } else if case .running = state {
                applyExit(uptime)
            }

        case .backoffElapsed:
            if case .backingOff = state {
                state = .starting
            }
        }
        return state
    }

    /// Fold a process exit into the failure streak and decide whether to back
    /// off or give up. Shared by the `starting` and `running` arms: the
    /// supervisor may observe the exit before or after the health poll.
    private mutating func applyExit(_ uptime: TimeInterval) {
        failures = policy.failures(afterExitWithPrevious: failures, uptime: uptime)
        if policy.shouldGiveUp(failures: failures) {
            state = .failed(failures: failures)
        } else {
            state = .backingOff(failures: failures, delay: policy.delay(afterFailures: failures))
        }
    }
}
