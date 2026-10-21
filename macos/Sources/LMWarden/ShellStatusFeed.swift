// ShellStatusFeed — the main-thread poll loop that keeps the latest
// ShellStatus for ShellCoordinator's `status` seam.
//
// While started it asks the injected `endpoint` for the current boot's
// (port, token) and, when there is one, fetches GET /api/shell/status through
// the injected `fetch` (ShellStatusClient by default): once immediately, then
// every `interval` through the injected `schedule`. `latest` holds the last
// reply; `onChange` fires only when that value actually changes.
//
// Failure semantics: no endpoint (server not booted) or a nil reply clears
// `latest` — a stale in-flight count must never hold Wait-Then-Quit open. The
// loop keeps its fixed cadence while the server is down but issues no fetch
// without an endpoint, logs nothing and fires onChange only on the first
// transition to nil, so a dead server costs one closure call per tick.
//
// Stale replies: the injected schedule has no cancel, so every tick and every
// fetch completion carries the generation it was issued under; stop() bumps
// the generation, making all of them inert. A completion is also dropped when
// endpoint() no longer returns the (port, token) it was issued for, so a
// reply from a previous boot never surfaces. At most one fetch is in flight.
//
// Threading: confined to one thread (the main thread in the app), like
// ShellCoordinator. Completions are hopped back through `deliver` before they
// touch any state. The token lives only in the app's endpoint closure and,
// transiently, in the in-flight completion that compares it; it is never a
// stored property and never logged.
import Foundation

final class ShellStatusFeed {
    /// The poll cadence in seconds.
    static let defaultInterval: TimeInterval = 2

    /// The most recent status reply; nil while stopped, while the server is
    /// not booted, or after a failed poll.
    private(set) var latest: ShellStatus?
    /// Called with the new value whenever `latest` changes.
    var onChange: ((ShellStatus?) -> Void)?

    private let endpoint: () -> (port: Int, token: String)?
    private let fetch: (Int, String, @escaping (ShellStatus?) -> Void) -> Void
    private let schedule: (TimeInterval, @escaping () -> Void) -> Void
    private let deliver: (@escaping () -> Void) -> Void
    private let interval: TimeInterval

    private var running = false
    /// Bumped by start() and stop(); scheduled ticks and fetch completions
    /// from an older generation are inert.
    private var generation = 0
    private var fetchInFlight = false

    init(
        endpoint: @escaping () -> (port: Int, token: String)?,
        fetch: @escaping (Int, String, @escaping (ShellStatus?) -> Void) -> Void = { port, token, done in
            ShellStatusClient(port: port, token: token).fetchStatus(completion: done)
        },
        schedule: @escaping (TimeInterval, @escaping () -> Void) -> Void = AppEnvironment.mainQueueSchedule,
        deliver: @escaping (@escaping () -> Void) -> Void = { DispatchQueue.main.async(execute: $0) },
        interval: TimeInterval = ShellStatusFeed.defaultInterval
    ) {
        self.endpoint = endpoint
        self.fetch = fetch
        self.schedule = schedule
        self.deliver = deliver
        self.interval = interval
    }

    /// Starts polling: one poll now, then every `interval`. A no-op while
    /// already started.
    func start() {
        guard !running else {
            return
        }
        running = true
        generation += 1
        fetchInFlight = false
        tick(generation: generation)
    }

    /// Stops polling and clears `latest`. Every tick and fetch completion
    /// issued before this call is inert. Idempotent; a later start() resumes.
    func stop() {
        generation += 1
        running = false
        fetchInFlight = false
        apply(nil)
    }

    private func tick(generation tickGeneration: Int) {
        guard running, tickGeneration == generation else {
            return
        }
        poll(generation: tickGeneration)
        schedule(interval) { [weak self] in
            self?.tick(generation: tickGeneration)
        }
    }

    private func poll(generation pollGeneration: Int) {
        guard let issued = endpoint() else {
            apply(nil)
            return
        }
        guard !fetchInFlight else {
            return
        }
        fetchInFlight = true
        fetch(issued.port, issued.token) { [weak self] result in
            guard let self else {
                return
            }
            self.deliver { [weak self] in
                self?.receive(result, generation: pollGeneration, issued: issued)
            }
        }
    }

    private func receive(
        _ result: ShellStatus?,
        generation replyGeneration: Int,
        issued: (port: Int, token: String)
    ) {
        guard replyGeneration == generation else {
            return
        }
        fetchInFlight = false
        guard let current = endpoint(), current == issued else {
            return
        }
        apply(result)
    }

    private func apply(_ value: ShellStatus?) {
        guard value != latest else {
            return
        }
        latest = value
        onChange?(value)
    }
}
