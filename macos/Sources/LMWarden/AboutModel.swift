// AboutModel — the About window's version line, update notice, buttons and the
// automatic-update checkbox, plus the no-op updater a dev build (no SUFeedURL /
// SUPublicEDKey) runs instead of SparkleUpdater (mockup #6, 2026-10-20).
//
// Pure model: Foundation only, no AppKit, no SwiftUI, no Sparkle. The view
// (P8) renders exactly what `make` produces and routes user actions through
// `UpdaterControlling`, so the decision of what to show lives here, not in the
// view.
import Foundation

// MARK: - Updater

/// The update surface the About window drives: `SparkleUpdater` in a bundle
/// configured for updates, `NoUpdater` otherwise.
protocol UpdaterControlling: AnyObject {
    /// Whether the updater can check for updates at all (gates the checkbox).
    var canCheckForUpdates: Bool { get }
    /// Whether to check for updates automatically; the checkbox edits this.
    var automaticallyChecksForUpdates: Bool { get set }
    /// Ask for an update now (a no-op on `NoUpdater`).
    func checkForUpdates()
    /// Install the pending update and relaunch. Returns whether an install
    /// actually started (false with nothing staged, and always on `NoUpdater`).
    @discardableResult
    func installAndRelaunch() -> Bool
    /// The current update state (always `.none` on `NoUpdater`).
    var updateState: UpdateState { get }
    /// Called on the main thread whenever `updateState` changes.
    var onStateChange: (() -> Void)? { get set }
    /// Called when an install that `installAndRelaunch` started fails.
    var onInstallFailed: (() -> Void)? { get set }
}

/// The no-op updater of a build without an update feed: it cannot check for
/// updates, stores the auto-check flag, and its methods do nothing.
final class NoUpdater: UpdaterControlling {
    var canCheckForUpdates: Bool { false }
    var updateState: UpdateState { .none }
    var onStateChange: (() -> Void)?
    var automaticallyChecksForUpdates: Bool
    init() {
        self.automaticallyChecksForUpdates = false
    }
    func checkForUpdates() {}
    func installAndRelaunch() -> Bool { false }
    var onInstallFailed: (() -> Void)?
}

// MARK: - UpdateState

/// Whether an update is available, with its release-notes URL if any.
enum UpdateState: Equatable {
    case none
    case available(version: String, releaseNotes: URL?)
}

// MARK: - AboutModel

/// Everything the About window shows (mockup #6). Built by `make` from the
/// current version, the update state and the updater's capabilities.
struct AboutModel: Equatable {
    /// The app's display name.
    let name: String
    /// "Version <version> (Apple Silicon)".
    let versionLine: String
    /// Set only when an update is available.
    let updateNotice: String?
    /// "Install and Relaunch", set only when an update is available.
    let primaryButton: String?
    /// "Release Notes", set only when an update is available AND release notes
    /// exist (nil when `releaseNotes` is nil).
    let notesButton: String?
    /// The automatic-update checkbox label.
    let autoCheckLabel: String
    /// The checkbox state, mirroring the updater's auto-check flag.
    let autoCheckOn: Bool
    /// Whether the checkbox is enabled (driven by `canCheck`).
    let autoCheckEnabled: Bool
    /// The window's footnote, always present.
    let footnote: String

    /// Build the About model from the current version, the update state and
    /// the updater's capabilities. The copy is the owner-approved text from
    /// the 2026-10-20 mockups, byte for byte.
    static func make(
        version: String,
        update: UpdateState,
        automaticallyChecks: Bool,
        canCheck: Bool
    ) -> AboutModel {
        var updateNotice: String?
        var primaryButton: String?
        var notesButton: String?
        if case .available(_, let releaseNotes) = update {
            updateNotice = "A new version is available. It installs when you quit, or now."
            primaryButton = "Install and Relaunch"
            if releaseNotes != nil {
                notesButton = "Release Notes"
            }
        }
        return AboutModel(
            name: "LM Warden",
            versionLine: "Version \(version) (Apple Silicon)",
            updateNotice: updateNotice,
            primaryButton: primaryButton,
            notesButton: notesButton,
            autoCheckLabel: "Check for updates automatically",
            autoCheckOn: automaticallyChecks,
            autoCheckEnabled: canCheck,
            footnote: "Relaunching stops running requests. You are asked first if any are in flight."
        )
    }
}
