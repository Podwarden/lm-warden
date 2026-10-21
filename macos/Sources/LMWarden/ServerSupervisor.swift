// ServerSupervisor — composes `SupervisorStateMachine` + a health probe +
// `ProcessLaunching` + `LogRing` (Metal design §2, non-UI).
//
// The supervisor owns one child process and the actual waiting. It is not
// thread-safe by design: the caller (the UI) serialises `start()`, `stop()` and
// the injected `schedule`/probe callbacks on one queue. It owns no Foundation
// timers, actors, or async of its own — `schedule` and `now` are injected so a
// test drives the polling and the clock by hand, and `probe` is a closure so a
// test can return canned `HealthResult`s without any network or URLProtocol.
import Foundation

/// Drives the embedded uvicorn child process: launches it, health-polls
/// `/healthz` while it is starting, and restarts it with `RestartBackoff`
/// backoff when it exits, until it either becomes healthy and stays up or
/// gives up after too many consecutive short exits.
///
/// Not thread-safe: the caller serialises all calls on one queue.
final class ServerSupervisor {
    static let defaultStopGracePeriod: TimeInterval = 15

    private let spec: ServerLaunchSpec
    private let launcher: ProcessLaunching
    private let probe: (@escaping (HealthResult) -> Void) -> Void
    private let schedule: (TimeInterval, @escaping () -> Void) -> Void
    private let now: () -> Date
    private let healthPollInterval: TimeInterval
    private let stopGracePeriod: TimeInterval
    private var machine: SupervisorStateMachine
    private(set) var state: SupervisorState
    private(set) var log: LogRing
    /// Called once per actual state change (never for an ignored event).
    var onStateChange: ((SupervisorState) -> Void)?

    private var currentProcess: LaunchedProcess?
    /// The per-launch box of `currentProcess`: it records that this launch
    /// exited and that stop() already signalled it. Per launch, never shared,
    /// so a late exit of a replaced process cannot mark the current one.
    private var currentBox: ProcessBox?
    private var launchedAt: Date?
    /// Bumped on every launch. The injected `schedule` has no cancel, so each
    /// scheduled poll, probe completion and backoff closure captures the
    /// generation it was issued under and does nothing once a later launch has
    /// superseded it (no stale poll chain survives a stop/start, no stale
    /// backoff cuts a later backoff short). After a bare stop() the stale
    /// closures are already inert: the machine ignores them from `.stopped`.
    private var generation = 0

    init(
        spec: ServerLaunchSpec,
        launcher: ProcessLaunching,
        probe: @escaping (@escaping (HealthResult) -> Void) -> Void,
        schedule: @escaping (TimeInterval, @escaping () -> Void) -> Void,
        now: @escaping () -> Date = Date.init,
        policy: RestartBackoff = RestartBackoff(),
        healthPollInterval: TimeInterval = 0.5,
        stopGracePeriod: TimeInterval = defaultStopGracePeriod,
        logCapacity: Int = 500
    ) {
        self.spec = spec
        self.launcher = launcher
        self.probe = probe
        self.schedule = schedule
        self.now = now
        self.healthPollInterval = healthPollInterval
        self.stopGracePeriod = stopGracePeriod
        self.machine = SupervisorStateMachine(policy: policy)
        self.state = machine.state
        self.log = LogRing(capacity: logCapacity)
    }

    /// Feed `.startRequested` and, when the machine moves to `.starting`,
    /// launch the child and schedule the first health poll.
    func start() {
        guard feed(.startRequested), state == .starting else {
            return
        }
        launchAndPoll()
    }

    /// Feed `.stopRequested` and send SIGTERM to the current process. After a
    /// grace period (`stopGracePeriod`, default 15 s) the process is SIGKILLed
    /// if it has not exited. Any later `onExit` for it changes nothing (the
    /// machine ignores `.exited` from `.stopped`).
    func stop() {
        feed(.stopRequested)
        guard let process = currentProcess, let box = currentBox, !box.stopSignalled else {
            return
        }
        box.stopSignalled = true
        process.terminate()
        // Holds the process and its own launch's box, never the supervisor
        // (AppController.stop() drops it): the kill still fires after release,
        // and is skipped once THIS launch's onExit has fired (no SIGKILL to a
        // reaped, possibly reused pid).
        schedule(stopGracePeriod) {
            if !box.exited {
                process.kill()
            }
        }
    }

    // MARK: - Internals

    /// Launch the child, record its launch time, and schedule a health poll. A
    /// `launch` that throws is treated as an immediate exit with uptime 0.
    private func launchAndPoll() {
        generation += 1
        let gen = generation
        let startedAt = now()
        // The onExit closure needs the launched handle to check identity, but
        // it is passed into launch() before the handle exists, so it reads a
        // box that launch() fills. onExit only ever fires after launch returns,
        // so the box is populated by then.
        let box = ProcessBox()
        do {
            let process = try launcher.launch(
                spec,
                onOutput: { [weak self] data in self?.log.append(data) },
                onExit: { [weak self] _ in
                    // Mark THIS launch as exited before anything else, and
                    // without going through self: a pending grace closure
                    // holds this box and must skip its SIGKILL even when the
                    // supervisor is gone or has launched a newer process.
                    box.exited = true
                    self?.handleExit(from: box.process)
                }
            )
            box.process = process
            currentProcess = process
            currentBox = box
            launchedAt = startedAt
            schedulePoll(gen)
        } catch {
            // A launch that throws is an immediate exit with uptime 0: record
            // the launch time and feed an exit directly (there is no process to
            // compare identity against).
            launchedAt = startedAt
            handleExit(from: nil)
        }
    }

    /// Run one health poll: probe only while `.starting`. A healthy result
    /// moves the machine to `.running`; an unhealthy one schedules the next
    /// poll. No poll is scheduled once running.
    private func poll(_ gen: Int) {
        guard gen == generation, state == .starting else {
            return
        }
        probe { [weak self] result in
            guard let self, gen == self.generation, self.state == .starting else {
                return
            }
            switch result {
            case .healthy:
                self.feed(.becameHealthy)
            case .unhealthy:
                self.schedulePoll(gen)
            }
        }
    }

    private func schedulePoll(_ gen: Int) {
        schedule(healthPollInterval) { [weak self] in self?.poll(gen) }
    }

    /// Fold an exit from `process` into the state machine. `process` is the
    /// launched handle the exit came from, or `nil` for a launch that threw
    /// (there is no process, so identity is not checked). An exit from a
    /// previous (replaced or stopped) process is ignored: it only counts when
    /// it is still the `currentProcess` and we recorded its launch time. On a
    /// real exit the log is flushed first so the child's final unterminated
    /// line is kept.
    private func handleExit(from process: LaunchedProcess?) {
        guard process == nil || process === currentProcess, let started = launchedAt else {
            return
        }
        log.flush()
        let uptime = now().timeIntervalSince(started)
        guard feed(.exited(uptime: uptime)), case .backingOff(_, let delay) = state else {
            return
        }
        currentProcess = nil
        currentBox = nil
        launchedAt = nil
        let gen = generation
        schedule(delay) { [weak self] in self?.onBackoffElapsed(gen) }
    }

    /// A scheduled backoff has elapsed: return to `.starting` and launch again.
    /// A backoff that fires after `stop()` does nothing.
    private func onBackoffElapsed(_ gen: Int) {
        guard gen == generation, feed(.backoffElapsed), state == .starting else {
            return
        }
        launchAndPoll()
    }

    /// Feed `event` to the machine, fire `onStateChange` exactly when the state
    /// actually changed, and report whether it changed.
    @discardableResult
    private func feed(_ event: SupervisorEvent) -> Bool {
        let before = state
        let after = machine.handle(event)
        state = after
        if before != after {
            onStateChange?(after)
        }
        return before != after
    }
}

/// A per-launch cell. It lets the `onExit` closure (passed into `launch`
/// before the handle exists) read the launched `LaunchedProcess` once it is
/// known, and records for the stop-grace closure whether this launch has
/// exited (`exited`, set by its own onExit) and whether stop() already sent
/// it SIGTERM and scheduled its kill (`stopSignalled`).
private final class ProcessBox {
    var process: LaunchedProcess?
    var exited = false
    var stopSignalled = false
}
