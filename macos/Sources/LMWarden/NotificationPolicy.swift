// NotificationPolicy — maps supervisor / memory / thermal / battery events to
// the approved notification copy with per-kind rate limits and hysteresis
// (Metal design §2, mockup #4 "Notifications").
//
// Pure Foundation: no AppKit, no SwiftUI, no UserNotifications, no timers, no
// async. The shell's event feed calls `handle(_:now:)` and posts the returned
// `ShellNotification` itself; the policy only decides when and what to say.
// All copy is decided here so the view layer renders it verbatim.
import Foundation

// MARK: - Input and output types

/// The Mac's thermal level as the shell sees it.
enum ShellThermal: Equatable {
    case nominal, fair, serious, critical
}

/// An event the shell observes and feeds to the notification policy.
enum ShellEvent: Equatable {
    case supervisor(SupervisorState)
    case memory(usedBytes: Int64, usableBytes: Int64)
    case thermal(ShellThermal)
    case power(onBattery: Bool, modelLoaded: Bool)
}

/// The kind of notification a policy may produce.
enum NotificationKind: String {
    case restarted, gaveUp, memoryTight, thermal, battery
}

/// The exact copy the shell shows, decided by the policy.
struct ShellNotification: Equatable {
    let kind: NotificationKind
    let title: String
    let body: String
}

// MARK: - Policy

/// Decides which approved notifications fire for a stream of `ShellEvent`s.
///
/// Holds only bookkeeping: the last backoff delay (for the restart copy),
/// the last observed state for each hysteresis/arming rule, and the last time
/// each kind FIRED (the rate-limit anchor). A suppressed event still updates
/// its arming state — only a fired notification advances the clock.
struct NotificationPolicy {
    /// Interval between two `memoryTight`, `thermal` and `battery` firings.
    let minimumInterval: TimeInterval
    /// Interval between two `restarted` firings.
    let restartInterval: TimeInterval

    private var lastBackoffDelay: TimeInterval?
    private var currentlyFailed = false
    private var memoryArmed = true
    private var lastThermal: ShellThermal?
    private var wasOnBattery = false
    private var lastFired: [NotificationKind: Date] = [:]

    init(minimumInterval: TimeInterval = 600, restartInterval: TimeInterval = 60) {
        self.minimumInterval = minimumInterval
        self.restartInterval = restartInterval
    }

    /// Handle one event observed at `now` and return a notification if this
    /// event fires one. `now` is the caller's clock so the policy stays pure.
    mutating func handle(_ event: ShellEvent, now: Date) -> ShellNotification? {
        switch event {
        case .supervisor(let state):
            return handleSupervisor(state, now: now)
        case .memory(let usedBytes, let usableBytes):
            return handleMemory(usedBytes: usedBytes, usableBytes: usableBytes, now: now)
        case .thermal(let level):
            return handleThermal(level, now: now)
        case .power(let onBattery, let modelLoaded):
            return handlePower(onBattery: onBattery, modelLoaded: modelLoaded, now: now)
        }
    }

    // MARK: - Supervisor

    private mutating func handleSupervisor(_ state: SupervisorState, now: Date) -> ShellNotification? {
        // Any state other than .failed ends the give-up episode, so the next
        // .failed (e.g. after "start it again") is a new entry and fires.
        if case .failed = state {} else { currentlyFailed = false }
        switch state {
        case .backingOff(_, let delay):
            lastBackoffDelay = delay
            return nil
        case .running:
            defer { lastBackoffDelay = nil }
            guard let delay = lastBackoffDelay else { return nil }
            return fire(.restarted, now: now, interval: restartInterval) {
                ShellNotification(
                    kind: .restarted,
                    title: "Server restarted",
                    body: "The server stopped unexpectedly and was restarted after \(ShellFormat.count(Int(delay.rounded()), "second", "seconds")). Requests in flight were lost."
                )
            }
        case .stopped:
            lastBackoffDelay = nil
            return nil
        case .failed(let failures):
            // A start after giving up is the user's, not an unexpected
            // restart: forget the backoff so its .running stays silent.
            lastBackoffDelay = nil
            let isEntry = !currentlyFailed
            currentlyFailed = true
            guard isEntry else { return nil }
            return ShellNotification(
                kind: .gaveUp,
                title: "Server stopped for good",
                body: "It failed \(ShellFormat.count(failures, "time", "times")) in a row. Open the menu bar menu to see the log or start it again."
            )
        case .starting:
            return nil
        }
    }

    // MARK: - Memory

    private mutating func handleMemory(usedBytes: Int64, usableBytes: Int64, now: Date) -> ShellNotification? {
        guard usableBytes > 0 else { return nil }
        let used = Double(usedBytes)
        let usable = Double(usableBytes)
        if used < 0.80 * usable {
            memoryArmed = true
        }
        guard memoryArmed, used >= 0.90 * usable else { return nil }
        memoryArmed = false
        return fire(.memoryTight, now: now, interval: minimumInterval) {
            ShellNotification(
                kind: .memoryTight,
                title: "Memory is tight",
                body: "Loaded models use \(ShellFormat.gigabytesNumber(usedBytes)) of \(ShellFormat.gigabytes(usableBytes)). New loads may be refused. Unload a model in the dashboard."
            )
        }
    }

    // MARK: - Thermal

    private mutating func handleThermal(_ level: ShellThermal, now: Date) -> ShellNotification? {
        let previous = lastThermal
        lastThermal = level
        guard level == .serious || level == .critical else { return nil }
        let fromCool = previous == nil || previous == .nominal || previous == .fair
        guard fromCool else { return nil }
        return fire(.thermal, now: now, interval: minimumInterval) {
            ShellNotification(
                kind: .thermal,
                title: "Mac is running hot",
                body: "macOS is slowing the chip to cool down. Responses will be slower until it cools."
            )
        }
    }

    // MARK: - Power

    private mutating func handlePower(onBattery: Bool, modelLoaded: Bool, now: Date) -> ShellNotification? {
        let wasAC = !wasOnBattery
        wasOnBattery = onBattery
        guard onBattery, wasAC, modelLoaded else { return nil }
        return fire(.battery, now: now, interval: minimumInterval) {
            ShellNotification(
                kind: .battery,
                title: "Running on battery",
                body: "Models drain the battery quickly. Connect power for long jobs."
            )
        }
    }

    // MARK: - Rate limit

    /// Build and return `make()`'s notification if `now` is at least `interval`
    /// after this kind's last firing (or it has never fired); otherwise record
    /// nothing and return nil. The anchor is set only when a notification is
    /// actually returned, so a suppressed event never refreshes the clock.
    private mutating func fire(
        _ kind: NotificationKind,
        now: Date,
        interval: TimeInterval,
        make: () -> ShellNotification
    ) -> ShellNotification? {
        if let last = lastFired[kind], now.timeIntervalSince(last) < interval {
            return nil
        }
        lastFired[kind] = now
        return make()
    }
}
