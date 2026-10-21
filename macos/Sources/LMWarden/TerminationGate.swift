// TerminationGate — the reply bookkeeping behind
// `ShellAppDelegate.applicationShouldTerminate` (planner iter 42). Pure: the
// delegate performs the returned answer/effect, so the tests drive every quit
// path (Quit, Wait-Then-Quit, Cancel, SIGTERM, logout, a second quit while one
// is pending) without AppKit.
//
// Unacknowledged quits (planner iter 58). Boot's Keychain read runs on the
// serial shell queue; while macOS shows a Keychain access prompt that queue is
// blocked, so a forwarded quit would never reach the coordinator and a pending
// `.terminateLater` would never be replied. Every quit the delegate forwards
// arms a main-thread deadline of `unacknowledgedQuitLimit`; the shell queue
// acknowledges as the first thing it does with the quit. A deadline that
// passes unacknowledged terminates without the graceful coordinator path
// (no child has been launched yet: the Keychain read precedes any launch).
// The gate holds no timers: the delegate arms them and reports back.
//
// Pre-confirmed quits (P8 sparkle-updater). About's Install and Relaunch asks
// about in-flight requests itself, then has Sparkle request termination. The
// delegate calls `preconfirmQuit()` right before, so that ONE following
// request is answered `.laterQuitConfirmed` (the coordinator stops the server
// and terminates without asking again). The pre-confirmation is consumed by
// the next request, withdrawn when the install does not start or fails, and
// expires after `preconfirmedQuitLimit` if Sparkle never asks, so a later
// ordinary Quit, SIGTERM or logout still prompts.
import Foundation

/// The reply bookkeeping for `applicationShouldTerminate`. Pure: the delegate
/// performs the returned effect. Exactly one `reply(toApplicationShouldTerminate:)`
/// is issued per `.later` answer.
struct TerminationGate: Equatable {
    enum Phase: Equatable {
        /// No termination in progress.
        case idle
        /// `.terminateLater` was answered; a reply is owed.
        case pending
        /// The coordinator terminated; the next request answers `.terminateNow`.
        case approved
    }

    /// The answer to one `applicationShouldTerminate`.
    enum Answer: Equatable {
        case now
        /// Answer `.terminateLater` and ask the coordinator to quit.
        case laterAskCoordinator
        /// Answer `.terminateLater` and tell the coordinator to quit without
        /// asking: the user already confirmed (a pre-confirmed quit).
        case laterQuitConfirmed
        case cancel
    }

    /// What the delegate must do after a coordinator outcome.
    enum Effect: Equatable {
        case nothing
        /// `NSApp.reply(toApplicationShouldTerminate:)` with this value.
        case reply(Bool)
        /// Call `NSApp.terminate(nil)` (no request is pending).
        case terminate
    }

    /// How long a forwarded quit may wait for the shell queue to pick it up
    /// before the app terminates anyway.
    static let unacknowledgedQuitLimit: TimeInterval = 5
    /// How long a pre-confirmation waits for the termination request it
    /// stands for.
    static let preconfirmedQuitLimit: TimeInterval = 60

    private(set) var phase: Phase = .idle
    /// The generation of the newest forwarded quit; the shell queue
    /// acknowledges it and its deadline names it, so stale timers are inert.
    private(set) var forwardGeneration = 0
    /// True while the newest forwarded quit has not been acknowledged.
    private(set) var awaitingAcknowledgement = false
    /// True while a pre-confirmed quit is expected (see the file header).
    private(set) var preconfirmed = false
    /// The generation of the newest pre-confirmation; its expiry names it.
    private(set) var preconfirmGeneration = 0

    /// AppKit asks whether to terminate. With no coordinator yet (before
    /// launch finished) the app quits at once.
    mutating func shouldTerminate(hasCoordinator: Bool) -> Answer {
        switch phase {
        case .approved:
            return .now
        case .pending:
            // A second request while one is pending changes nothing; the
            // pending one still gets its single reply.
            return .cancel
        case .idle:
            let confirmed = preconfirmed
            preconfirmed = false
            guard hasCoordinator else {
                phase = .approved
                return .now
            }
            phase = .pending
            return confirmed ? .laterQuitConfirmed : .laterAskCoordinator
        }
    }

    // MARK: - Pre-confirmed quits

    /// The user confirmed a quit that is about to be requested (Install and
    /// Relaunch). Returns the generation to arm the expiry for.
    mutating func preconfirmQuit() -> Int {
        preconfirmGeneration += 1
        preconfirmed = true
        return preconfirmGeneration
    }

    /// The install the pre-confirmation stood for did not start or failed:
    /// drop it now (a later SIGTERM or logout must prompt again).
    mutating func withdrawPreconfirmation() {
        preconfirmed = false
    }

    /// The expiry armed for `generation` fired: a pre-confirmation no request
    /// consumed is dropped. A stale expiry is inert.
    mutating func preconfirmationExpired(_ generation: Int) {
        if generation == preconfirmGeneration {
            preconfirmed = false
        }
    }

    /// The coordinator stopped the server and wants the process gone.
    mutating func coordinatorTerminated() -> Effect {
        switch phase {
        case .idle:
            phase = .approved
            return .terminate
        case .pending:
            phase = .approved
            return .reply(true)
        case .approved:
            return .nothing
        }
    }

    /// The user cancelled the quit prompt.
    mutating func quitCancelled() -> Effect {
        switch phase {
        case .pending:
            phase = .idle
            return .reply(false)
        case .idle, .approved:
            return .nothing
        }
    }

    // MARK: - Unacknowledged quits

    /// The delegate forwards a quit to the shell queue. Returns the
    /// generation to arm a deadline for, or nil when no new deadline is due:
    /// one is already running for an unacknowledged forward (that deadline
    /// covers this one too), or termination is already approved. The shell
    /// queue acknowledges `forwardGeneration` either way.
    mutating func quitForwarded() -> Int? {
        guard phase != .approved, !awaitingAcknowledgement else {
            return nil
        }
        forwardGeneration += 1
        awaitingAcknowledgement = true
        return forwardGeneration
    }

    /// The shell queue picked up the forward of `generation`. An
    /// acknowledgement for an older forward does not cover a newer one.
    mutating func quitAcknowledged(_ generation: Int) {
        if generation == forwardGeneration {
            awaitingAcknowledgement = false
        }
    }

    /// The deadline armed for `generation` fired. Unacknowledged: reply the
    /// pending request `true`, or terminate when none is pending. Otherwise
    /// (acknowledged, stale, or already approved) nothing.
    mutating func acknowledgementDeadlinePassed(_ generation: Int) -> Effect {
        guard generation == forwardGeneration, awaitingAcknowledgement else {
            return .nothing
        }
        awaitingAcknowledgement = false
        switch phase {
        case .pending:
            phase = .approved
            return .reply(true)
        case .idle:
            phase = .approved
            return .terminate
        case .approved:
            return .nothing
        }
    }
}
