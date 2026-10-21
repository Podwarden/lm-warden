import AppKit
import SwiftUI

/// The About window (BACKLOG-macos P7, mockup screen 6): app icon, name,
/// version line, the update notice with Install and Relaunch / Release Notes
/// when the updater has an update, the automatic-update checkbox and the
/// footnote. Every string, enabled state and check mark comes from
/// `AboutModel.make`; the view only renders it and forwards actions to the
/// updater.
///
/// Install and Relaunch stops the server, so it first asks `quitDecision`
/// (the app's `QuitDecision.decide` over the live in-flight snapshot): with
/// nothing in flight it installs at once; otherwise the approved
/// Quit-while-busy prompt is shown through `QuitAlertPresenter` (the same
/// renderer Quit uses) and the choice is honoured
/// (Wait, Then Quit hands the install to `waitThenRun`, the app's poll loop).
///
/// One controller owns one window: `show()` reuses it, closing only hides it.
///
/// View file: no decisions of its own and never instantiated in tests.
@MainActor
final class AboutWindowController {
    private let state: AboutViewState
    private var window: NSWindow?

    /// - Parameters:
    ///   - updater: the update surface (`SparkleUpdater`, or `NoUpdater` in a
    ///     dev build); its `updateState` is re-read on every `show()` and on
    ///     every change.
    ///   - quitDecision: whether relaunching may proceed at once or must ask
    ///     first (the app passes `QuitDecision.decide` over its live status).
    ///   - waitThenRun: runs the given work once no request is in flight (the
    ///     app's Wait-Then-Quit poll loop).
    ///   - confirmQuit: called right before the install starts: the user has
    ///     answered (or nothing was in flight), so the termination Sparkle
    ///     requests next must not ask again.
    ///   - withdrawQuit: called when that install did not start, so the
    ///     pre-confirmation does not outlive it.
    init(
        updater: UpdaterControlling,
        quitDecision: @escaping () -> QuitDecision,
        waitThenRun: @escaping (@escaping () -> Void) -> Void,
        confirmQuit: @escaping () -> Void,
        withdrawQuit: @escaping () -> Void
    ) {
        state = AboutViewState(updater: updater, update: { updater.updateState })
        state.onInstall = { [weak self] in
            self?.confirmInstall(
                quitDecision: quitDecision,
                waitThenRun: waitThenRun,
                install: { Self.install(updater, confirmQuit: confirmQuit, withdrawQuit: withdrawQuit) }
            )
        }
    }

    /// Pre-confirm the quit, start the install, and withdraw the
    /// pre-confirmation when the install did not start. Pre-confirming
    /// first: Sparkle may request termination as soon as the install runs.
    @discardableResult
    nonisolated static func install(
        _ updater: UpdaterControlling,
        confirmQuit: () -> Void,
        withdrawQuit: () -> Void
    ) -> Bool {
        confirmQuit()
        guard updater.installAndRelaunch() else {
            withdrawQuit()
            return false
        }
        return true
    }

    /// Re-read the updater (its state changed); a hidden window is refreshed
    /// again on `show()`.
    func refresh() {
        state.refresh()
    }

    func show() {
        state.refresh()
        let window = ensureWindow()
        if !window.isVisible { window.center() }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func ensureWindow() -> NSWindow {
        if let window { return window }
        let hosting = NSHostingController(rootView: AboutView(state: state))
        // The window follows the view's ideal size, so it grows and shrinks
        // as the update block appears or goes away.
        hosting.sizingOptions = [.preferredContentSize]
        let window = NSWindow(contentViewController: hosting)
        window.styleMask = [.titled, .closable]
        window.title = ""
        window.isReleasedWhenClosed = false
        self.window = window
        return window
    }

    // MARK: Install and Relaunch

    private func confirmInstall(
        quitDecision: () -> QuitDecision,
        waitThenRun: @escaping (@escaping () -> Void) -> Void,
        install: @escaping () -> Void
    ) {
        switch quitDecision() {
        case .quitNow:
            install()
        case .confirm(let prompt):
            // The one renderer of the approved Quit-while-busy prompt
            // (mockup screen 3), shared with Quit.
            QuitAlertPresenter.present(prompt) { choice in
                switch choice {
                case .quitNow:
                    install()
                case .waitThenQuit:
                    waitThenRun { install() }
                case .cancel:
                    break
                }
            }
        }
    }
}

// MARK: - Observable wrapper

/// Holds the current `AboutModel` for SwiftUI and forwards the view's actions.
/// It rebuilds the model from the updater on every change; it decides nothing.
@MainActor
final class AboutViewState: ObservableObject {
    let updater: UpdaterControlling
    private let update: () -> UpdateState
    @Published private(set) var model: AboutModel
    @Published private(set) var updateState: UpdateState
    var onInstall: () -> Void = {}

    init(updater: UpdaterControlling, update: @escaping () -> UpdateState) {
        self.updater = updater
        self.update = update
        let current = update()
        self.updateState = current
        self.model = AboutModel.make(
            version: AppVersion.current,
            update: current,
            automaticallyChecks: updater.automaticallyChecksForUpdates,
            canCheck: updater.canCheckForUpdates
        )
    }

    func refresh() {
        let current = update()
        updateState = current
        model = AboutModel.make(
            version: AppVersion.current,
            update: current,
            automaticallyChecks: updater.automaticallyChecksForUpdates,
            canCheck: updater.canCheckForUpdates
        )
    }

    func setAutomaticallyChecks(_ on: Bool) {
        updater.automaticallyChecksForUpdates = on
        refresh()
    }

    func installAndRelaunch() {
        onInstall()
    }

    func openReleaseNotes() {
        if case .available(_, let url?) = updateState {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - View

struct AboutView: View {
    @ObservedObject var state: AboutViewState

    var body: some View {
        let model = state.model
        VStack(spacing: 8) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 64, height: 64)
                .accessibilityLabel(model.name)
            Text(model.name)
                .font(.system(size: 16, weight: .semibold))
            Text(model.versionLine)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            Divider()
                .padding(.top, 6)

            if let notice = model.updateNotice {
                Text(notice)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    if let primary = model.primaryButton {
                        Button(primary) { state.installAndRelaunch() }
                            .keyboardShortcut(.defaultAction)
                    }
                    if let notes = model.notesButton {
                        Button(notes) { state.openReleaseNotes() }
                    }
                }
            }

            Toggle(
                model.autoCheckLabel,
                isOn: Binding(
                    get: { state.model.autoCheckOn },
                    set: { state.setAutomaticallyChecks($0) }
                )
            )
            .toggleStyle(.checkbox)
            .disabled(!model.autoCheckEnabled)

            Text(model.footnote)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .multilineTextAlignment(.center)
        .padding(20)
        .frame(width: 340)
    }
}
