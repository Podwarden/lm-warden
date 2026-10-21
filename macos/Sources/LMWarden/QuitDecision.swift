// QuitDecision — whether Quit (⌘Q) quits at once or first shows the approved
// Quit-while-busy prompt, built from the in-flight snapshot (mockup #3).
//
// Pure model: Foundation only, no AppKit, no SwiftUI. The app's quit handler
// asks `decide` for the decision and `shouldQuitAfterPoll` whether its poll
// loop may finally quit.
import Foundation

// MARK: - Value types

/// The approved Quit-while-busy prompt, byte for byte (mockup #3, 2026-10-20).
struct QuitPrompt: Equatable {
    let title: String
    let message: String
    let details: [String]
    let buttons: [String]
    let footnote: String
}

/// What the user picked on the prompt.
enum QuitChoice: Equatable {
    case waitThenQuit
    case quitNow
    case cancel
}

// MARK: - Decision

enum QuitDecision: Equatable {
    /// Quit immediately, no prompt.
    case quitNow
    /// Show this prompt before quitting.
    case confirm(QuitPrompt)

    // MARK: - decide

    /// Quit at once unless the server is running, the feed is up and at least
    /// one request is still in flight; otherwise show the prompt.
    static func decide(serverRunning: Bool, status: ShellStatus?) -> QuitDecision {
        guard serverRunning, let status, status.inFlightCount > 0 else {
            return .quitNow
        }
        return .confirm(makePrompt(status))
    }

    // MARK: - choice

    /// Map a sheet button index onto a choice; anything but 0 and 1 cancels.
    static func choice(forButtonIndex index: Int) -> QuitChoice {
        switch index {
        case 0:
            return .waitThenQuit
        case 1:
            return .quitNow
        default:
            return .cancel
        }
    }

    // MARK: - Poll rule

    /// The Wait-Then-Quit poll loop quits once the feed reports no requests
    /// in flight — or once the feed itself is gone.
    static func shouldQuitAfterPoll(status: ShellStatus?) -> Bool {
        guard let status else {
            return true
        }
        return status.inFlightCount == 0
    }

    // MARK: - Prompt

    private static func makePrompt(_ status: ShellStatus) -> QuitPrompt {
        let count = status.inFlightCount
        let visible = status.inFlight.filter { $0.requests > 0 }
        var details = visible.prefix(4).map { group -> String in
            "\(group.who) \u{2014} \(group.model) \u{00B7} \(ShellFormat.count(group.requests, "request", "requests"))"
        }
        if visible.count > 4 {
            details.append("and \(visible.count - 4) more")
        }
        return QuitPrompt(
            title: "Quit LM Warden?",
            message: "\(ShellFormat.count(count, "request is", "requests are")) still running. Quitting stops them and unloads all models.",
            details: details,
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )
    }
}
