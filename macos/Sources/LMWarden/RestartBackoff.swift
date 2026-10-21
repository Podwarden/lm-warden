// RestartBackoff — pure restart-delay and give-up policy for the
// ServerSupervisor (Metal design §2).
//
// The ServerSupervisor (a Swift actor owning one `Process` and health-polling
// `/healthz`) uses this type to decide how long to wait before restarting the
// server and when to give up. It is intentionally free of any Foundation
// timers, actors, or async: the supervisor owns the waiting; this type owns
// only the arithmetic.
//
// The defaults — 1 s base, 30 s cap, 60 s stable-uptime threshold, and 5
// max consecutive failures — are judgement calls. They are exposed through
// `init` so a future operator can tune them without touching the policy code.
import Foundation

/// Pure restart-delay and give-up policy for the ServerSupervisor.
///
/// `ServerSupervisor` (Metal design §2) owns the `Process` and the actual
/// waiting; this type only computes the numbers. The defaults (1 s / 30 s /
/// 60 s / 5) are judgement calls — see the file header.
struct RestartBackoff: Equatable {
    let base: TimeInterval
    let cap: TimeInterval
    let stableUptime: TimeInterval
    let maxConsecutiveFailures: Int

    init(
        base: TimeInterval = 1,
        cap: TimeInterval = 30,
        stableUptime: TimeInterval = 60,
        maxConsecutiveFailures: Int = 5
    ) {
        self.base = base
        self.cap = cap
        self.stableUptime = stableUptime
        self.maxConsecutiveFailures = maxConsecutiveFailures
    }

    /// Delay in seconds before the `failures`-th restart attempt: `base *
    /// 2^(failures-1)`, capped at `cap`; 0 for `failures` ≤ 0.
    func delay(afterFailures failures: Int) -> TimeInterval {
        guard failures > 0 else { return 0 }
        var delay = min(base, cap)
        // Cap before doubling so a non-power-of-two `cap` (e.g. 30) is never
        // overshot, `Int.max` failures cannot overflow the doubling, a `base`
        // above `cap` is clamped, and a zero `base` cannot spin the loop.
        for _ in 0..<max(failures - 1, 0) {
            if delay >= cap || delay <= 0 { break }
            delay = min(delay * 2, cap)
        }
        return delay
    }

    /// Consecutive-failure count after a process exit: 1 if the process ran
    /// at least `stableUptime` seconds (a new streak), otherwise
    /// `max(previous, 0) + 1`.
    func failures(afterExitWithPrevious previous: Int, uptime: TimeInterval) -> Int {
        if uptime >= stableUptime {
            return 1
        }
        return max(previous, 0) + 1
    }

    /// Whether the supervisor should stop retrying: true when `failures`
    /// exceeds `maxConsecutiveFailures`.
    func shouldGiveUp(failures: Int) -> Bool {
        failures > maxConsecutiveFailures
    }
}
