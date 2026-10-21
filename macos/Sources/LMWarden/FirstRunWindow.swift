import AppKit
import SwiftUI

/// The first-run window (BACKLOG-macos P7, mockup screen 2 "First run
/// setup"): a 720x520 non-resizable window with the four-step sidebar on the
/// left and the current step on the right. Every string, enabled state and
/// check mark comes from `FirstRunModel`; this file only renders it and
/// forwards clicks. Finishing the flow calls `onFinish` with the collected
/// result and closes the window. The window keeps the normal close button;
/// closing it before finishing calls `onCancel` (setup is not done, so the
/// coordinator quits the app) — the window never terminates the app itself.
///
/// View file: no decisions of its own and never instantiated in tests.
@MainActor
final class FirstRunWindowController: NSObject, NSWindowDelegate {
    private let store: FirstRunStore
    private let onFinish: (FirstRunResult) -> Void
    private let onCancel: () -> Void
    private var window: NSWindow?

    init(
        model: FirstRunModel,
        onFinish: @escaping (FirstRunResult) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.store = FirstRunStore(model: model)
        self.onFinish = onFinish
        self.onCancel = onCancel
        super.init()
    }

    func show() {
        let window = ensureWindow()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func ensureWindow() -> NSWindow {
        if let window { return window }

        let view = FirstRunView(
            store: store,
            chooseFolder: { [weak self] in self?.chooseModelsFolder() },
            finish: { [weak self] result in self?.finish(result) }
        )
        let hosting = NSHostingView(rootView: view)
        let frame = NSRect(x: 0, y: 0, width: 720, height: 520)
        hosting.frame = frame

        // Not resizable. Closable: closing before the final step cancels
        // setup through `onCancel` (windowWillClose below).
        let window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = store.model.windowTitle
        window.isReleasedWhenClosed = false
        window.contentView = hosting
        window.delegate = self
        window.center()
        self.window = window
        return window
    }

    /// Directories only, may create one; runs modally on the main thread.
    /// The chosen path goes to the model, which applies its own rules.
    private func chooseModelsFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: store.model.modelsDir, isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.chooseModelsDir(url.path)
    }

    private func finish(_ result: FirstRunResult) {
        onFinish(result)
        // Detach first so this close is not reported as a cancel.
        window?.delegate = nil
        window?.close()
        window = nil
    }

    // MARK: NSWindowDelegate

    /// The close button (or Cmd-W) before the flow finished: setup was
    /// cancelled. The coordinator decides what that means (it quits).
    func windowWillClose(_ notification: Notification) {
        window?.delegate = nil
        window = nil
        onCancel()
    }
}

/// The ObservableObject the SwiftUI view watches. It holds the value-type
/// model and only forwards to it.
@MainActor
final class FirstRunStore: ObservableObject {
    @Published private(set) var model: FirstRunModel

    init(model: FirstRunModel) {
        self.model = model
    }

    func next() -> FirstRunResult? { model.next() }
    func back() { model.back() }
    func chooseNetwork(lan: Bool) { model.chooseNetwork(lan: lan) }
    func chooseModelsDir(_ path: String) { model.chooseModelsDir(path) }
    func toggleExcludeModelsFromBackup() { model.toggleExcludeModelsFromBackup() }
    func toggleOpenAtLogin() { model.toggleOpenAtLogin() }
}

// MARK: - SwiftUI

private enum FirstRunPalette {
    static let accent = Color(red: 0x0A / 255, green: 0x5B / 255, blue: 0xD8 / 255)
    static let cardAccent = Color(red: 0x0A / 255, green: 0x64 / 255, blue: 0xD8 / 255)
    static let cardSelected = Color(red: 0xEE / 255, green: 0xF4 / 255, blue: 0xFE / 255)
    static let border = Color(red: 0xC3 / 255, green: 0xC6 / 255, blue: 0xCB / 255)
    static let hairline = Color(red: 0xE3 / 255, green: 0xE4 / 255, blue: 0xE7 / 255)
    static let sidebar = Color(red: 0xF6 / 255, green: 0xF6 / 255, blue: 0xF7 / 255)
    static let secondary = Color(red: 0x55 / 255, green: 0x56 / 255, blue: 0x5A / 255)
    static let body = Color(red: 0x3A / 255, green: 0x3A / 255, blue: 0x3C / 255)
    static let ok = Color(red: 0x1F / 255, green: 0x6F / 255, blue: 0x32 / 255)
    static let notOK = Color(red: 0xB0 / 255, green: 0x3A / 255, blue: 0x00 / 255)
    static let warningFill = Color(red: 0xFF / 255, green: 0xF4 / 255, blue: 0xE0 / 255)
    static let warningBorder = Color(red: 0xF0 / 255, green: 0xC9 / 255, blue: 0x8A / 255)
    static let warningText = Color(red: 0x6B / 255, green: 0x43 / 255, blue: 0x00 / 255)
}

private struct FirstRunView: View {
    @ObservedObject var store: FirstRunStore
    let chooseFolder: () -> Void
    let finish: (FirstRunResult) -> Void

    private var model: FirstRunModel { store.model }

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            VStack(alignment: .leading, spacing: 14) {
                Text(model.headline)
                    .font(.system(size: 22, weight: .bold))
                    .accessibilityAddTraits(.isHeader)
                stepContent
                Spacer(minLength: 0)
                buttons
            }
            .font(.system(size: 13))
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(width: 720, height: 520)
        .background(Color.white)
        .foregroundColor(Color(red: 0x1D / 255, green: 0x1D / 255, blue: 0x1F / 255))
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(model.stepLabels.enumerated()), id: \.offset) { index, label in
                let current = index + 1 == model.step.rawValue
                Text(label)
                    .font(.system(size: 13, weight: current ? .semibold : .regular))
                    .foregroundColor(current ? FirstRunPalette.accent : FirstRunPalette.secondary)
                    .accessibilityAddTraits(current ? .isSelected : [])
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 20)
        .frame(width: 190, alignment: .topLeading)
        .frame(maxHeight: .infinity, alignment: .topLeading)
        .background(FirstRunPalette.sidebar)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Setup steps")
    }

    // MARK: Steps

    @ViewBuilder
    private var stepContent: some View {
        switch model.step {
        case .welcome:
            welcome
        case .network:
            network
        case .models:
            models
        case .finish:
            finishStep
        }
    }

    private var welcome: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(model.body)
                .foregroundColor(FirstRunPalette.body)
                .fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 8) {
                Text("THIS MAC")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(FirstRunPalette.secondary)
                ForEach(Array(model.macCheckRows.enumerated()), id: \.offset) { _, row in
                    HStack {
                        Text(row.label)
                        Spacer()
                        Text(row.value)
                            .foregroundColor(row.ok ? FirstRunPalette.ok : FirstRunPalette.notOK)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .overlay(
                RoundedRectangle(cornerRadius: 8).stroke(FirstRunPalette.hairline, lineWidth: 1)
            )
        }
    }

    private var network: some View {
        VStack(alignment: .leading, spacing: 14) {
            NetworkCard(
                title: model.localCardTitle,
                detail: model.localCardBody,
                selected: !model.shareOnLAN,
                action: { store.chooseNetwork(lan: false) }
            )
            NetworkCard(
                title: model.lanCardTitle,
                detail: model.lanCardBody,
                selected: model.shareOnLAN,
                action: { store.chooseNetwork(lan: true) }
            )
            if let warning = model.lanWarning {
                Text(warning)
                    .foregroundColor(FirstRunPalette.warningText)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 8).fill(FirstRunPalette.warningFill)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8).stroke(FirstRunPalette.warningBorder, lineWidth: 1)
                    )
            }
        }
    }

    private var models: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(model.body)
                .foregroundColor(FirstRunPalette.body)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 8) {
                Text("Folder")
                    .frame(width: 70, alignment: .leading)
                Text(model.displayedModelsDir)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6).stroke(FirstRunPalette.border, lineWidth: 1)
                    )
                    .accessibilityLabel("Models folder")
                    .accessibilityValue(model.displayedModelsDir)
                Button("Choose\u{2026}", action: chooseFolder)
                    .accessibilityLabel("Choose models folder")
            }
            Text(model.freeLine)
                .foregroundColor(FirstRunPalette.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle(
                model.backupCheckboxLabel,
                isOn: Binding(
                    get: { store.model.excludeModelsFromBackup },
                    set: { value in
                        if value != store.model.excludeModelsFromBackup {
                            store.toggleExcludeModelsFromBackup()
                        }
                    }
                )
            )
            .toggleStyle(.checkbox)
        }
    }

    private var finishStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(model.finishLines.enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .foregroundColor(FirstRunPalette.body)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Toggle(
                model.openAtLoginCheckboxLabel,
                isOn: Binding(
                    get: { store.model.openAtLogin },
                    set: { value in
                        if value != store.model.openAtLogin {
                            store.toggleOpenAtLogin()
                        }
                    }
                )
            )
            .toggleStyle(.checkbox)
            Text(model.footer)
                .foregroundColor(FirstRunPalette.body)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Buttons

    private var buttons: some View {
        HStack {
            Button("Back") { store.back() }
                .disabled(!model.backEnabled)
                .accessibilityLabel("Back")
            Spacer()
            Button(model.nextLabel) {
                if let result = store.next() {
                    finish(result)
                }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.canContinue)
            .accessibilityLabel(model.nextLabel)
        }
        .controlSize(.large)
    }
}

/// One of the two Network step choices: a button drawn as a card.
private struct NetworkCard: View {
    let title: String
    let detail: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).fontWeight(.bold)
                Text(detail)
                    .foregroundColor(FirstRunPalette.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selected ? FirstRunPalette.cardSelected : Color.white)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(
                        selected ? FirstRunPalette.cardAccent : FirstRunPalette.border,
                        lineWidth: selected ? 2 : 1
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(title)
        .accessibilityHint(detail)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }
}
