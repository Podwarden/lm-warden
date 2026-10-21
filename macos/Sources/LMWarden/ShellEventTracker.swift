// ShellEventTracker — turns the shell's menu refreshes into the ShellEvents
// NotificationPolicy consumes (planner iter 42). Pure; tested on its own.
import Foundation

/// Turns successive (supervisor state, status) snapshots into the
/// `ShellEvent`s NotificationPolicy consumes: one `.supervisor` per state
/// change (a missing supervisor reads `.stopped`) and one `.memory` per new
/// non-nil status. Pure; NotificationPolicy does the edge detection.
struct ShellEventTracker: Equatable {
    private var lastState: SupervisorState?
    private var lastStatus: ShellStatus?

    mutating func events(state: SupervisorState, status: ShellStatus?) -> [ShellEvent] {
        var events: [ShellEvent] = []
        if state != lastState {
            lastState = state
            events.append(.supervisor(state))
        }
        if status != lastStatus {
            lastStatus = status
            if let status {
                events.append(.memory(usedBytes: status.usedBytes, usableBytes: status.usableBytes))
            }
        }
        return events
    }
}
