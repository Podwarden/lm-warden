import AppKit

/// The Quit-while-busy prompt (BACKLOG-macos P7, mockup screen 3): an NSAlert
/// that renders the `QuitPrompt` built by `QuitDecision`. No decisions of its
/// own — every string comes from the model, and the response is mapped back
/// through `QuitDecision.choice(forButtonIndex:)`.
///
/// View file: never instantiated in tests.
@MainActor
enum QuitAlertPresenter {
    static func present(_ prompt: QuitPrompt, completion: @escaping (QuitChoice) -> Void) {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = prompt.title
        var parts = [prompt.message]
        if !prompt.details.isEmpty {
            parts.append(prompt.details.joined(separator: "\n"))
        }
        parts.append(prompt.footnote)
        alert.informativeText = parts.joined(separator: "\n\n")
        for title in prompt.buttons {
            alert.addButton(withTitle: title)
        }
        // Quit Now (mockup screen 3) draws red; found by title, not position.
        alert.buttons.first { $0.title == "Quit Now" }?.hasDestructiveAction = true
        NSApp.activate()
        let response = alert.runModal()
        let index = response.rawValue - NSApplication.ModalResponse.alertFirstButtonReturn.rawValue
        completion(QuitDecision.choice(forButtonIndex: index))
    }
}
