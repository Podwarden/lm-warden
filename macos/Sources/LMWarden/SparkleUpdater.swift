// SparkleUpdater — the Sparkle-backed `UpdaterControlling` behind the About
// window and the menu's Check for Updates… (BACKLOG-macos P8).
//
// r1 (overseer decision 1): NO Sparkle stock UI. SPUUpdater is driven by
// `HeadlessUserDriver`, which shows no window: a found update (user check or
// scheduled) is downloaded and reaches the About window as `.available`; no
// update / an error shows nothing. Check for Updates… starts a check and
// opens the About window. Release Notes opens the URL in the browser.
//
// Design (P8 sparkle-updater, 2026-10-21):
// - Enabled only when the bundle's Info.plist carries BOTH SUFeedURL and
//   SUPublicEDKey (`UpdaterConfiguration.isEnabled`; write-info-plist.sh writes
//   them only when both are set). A dev build has neither, so the app runs
//   `NoUpdater`: Check for Updates… is disabled, the About window shows no
//   update notice and its checkbox is disabled. Nothing extra appears.
// - One switch, one source of truth: ShellSettings'
//   `lmw.automaticallyChecksForUpdates` (default on). It is pushed into
//   Sparkle's automaticallyChecksForUpdates AND automaticallyDownloadsUpdates
//   at launch (before the updater starts) and on every change, so Sparkle's
//   own SUEnableAutomaticChecks / SUAutomaticallyUpdate defaults are a mirror
//   rewritten from it, never read back. With the switch on, a scheduled check
//   downloads a newer item silently and Sparkle stages it to install when the
//   app quits — exactly the approved notice "It installs when you quit, or
//   now." Sparkle's permission prompt is suppressed (the checkbox is the
//   choice). Only `SparkleUpdater.attach` and its auto-check setter write
//   Sparkle's two flags (pinned by a source-scan test).
// - State: `UpdateState` changes only through `UpdaterEvent`s (the pure
//   `UpdateState.applying`): `.staged` (Sparkle's willInstallUpdateOnQuit)
//   makes it `.available`, keeping the immediate-install handler; `.noUpdate`
//   clears it; `.installing` (installed and relaunched) clears it; an aborted/failed cycle leaves a staged update in place (it
//   still installs on quit).
// - Install and Relaunch calls the staged install handler (the silent path's
//   immediate-install block, or the driver's held Install reply). The
//   in-flight prompt runs BEFORE that (AboutWindowController, the approved
//   Quit-while-busy prompt), and the app's termination that Sparkle then
//   requests is pre-confirmed (TerminationGate.preconfirmQuit) so it does not
//   prompt twice. The pre-confirmation is withdrawn when the install does not
//   run (nothing staged, a used one-shot reply) or fails afterwards
//   (`.failed` after an install attempt -> onInstallFailed).
//
// The Sparkle types are main-actor (NS_SWIFT_UI_ACTOR); this class is used
// only on the main thread and hops with `MainActor.assumeIsolated` where it
// touches them, so the `UpdaterControlling` surface stays nonisolated.
import AppKit
import Foundation
import Sparkle

// MARK: - Configuration

enum UpdaterConfiguration {
    static let feedURLKey = "SUFeedURL"
    static let publicKeyKey = "SUPublicEDKey"

    /// True when `info` (a bundle's Info.plist) names a feed with an http(s)
    /// URL and a non-empty public EdDSA key. Either one missing (a dev build)
    /// or malformed disables the updater.
    static func isEnabled(info: [String: Any]?) -> Bool {
        guard let info,
              let feed = info[feedURLKey] as? String,
              let url = URL(string: feed.trimmingCharacters(in: .whitespaces)),
              let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              url.host?.isEmpty == false,
              let key = info[publicKeyKey] as? String,
              !key.trimmingCharacters(in: .whitespaces).isEmpty
        else {
            return false
        }
        return true
    }
}

// MARK: - State machine

/// What the updater reports; the only inputs to `UpdateState`.
enum UpdaterEvent: Equatable {
    /// Downloaded and staged: installs when the app quits, or now.
    case staged(version: String, releaseNotes: URL?)
    /// A check found nothing newer.
    case noUpdate
    /// An update cycle was aborted or failed.
    case failed
    /// Sparkle is installing; the app is about to terminate.
    case installing
}

extension UpdateState {
    func applying(_ event: UpdaterEvent) -> UpdateState {
        switch event {
        case .staged(let version, let releaseNotes):
            return .available(version: version, releaseNotes: releaseNotes)
        case .noUpdate, .installing:
            return .none
        case .failed:
            return self
        }
    }
}

// MARK: - Engine seam

/// The slice of `SPUUpdater` SparkleUpdater drives; tests use a fake.
protocol SparkleEngine: AnyObject {
    var automaticallyChecksForUpdates: Bool { get set }
    var automaticallyDownloadsUpdates: Bool { get set }
    func checkForUpdates()
}

// MARK: - SparkleUpdater

final class SparkleUpdater: UpdaterControlling {
    private let settings: ShellSettings
    private var engine: SparkleEngine?
    /// Installs the staged update now; false when it could not (a one-shot
    /// Sparkle reply already used).
    private var immediateInstall: (() -> Bool)?
    /// True from an install that ran until the app goes or the install fails.
    private var installAttempted = false
    /// The Sparkle updater, user driver and delegate bridge (live wiring only).
    private var retained: [AnyObject] = []
    private(set) var updateState: UpdateState = .none
    var onStateChange: (() -> Void)?
    var onInstallFailed: (() -> Void)?

    init(settings: ShellSettings) {
        self.settings = settings
    }

    /// Attach the engine and push the one auto-check switch into it. Called
    /// before the Sparkle updater starts, so its first schedule already
    /// follows the setting.
    func attach(_ engine: SparkleEngine) {
        self.engine = engine
        apply(settings.automaticallyChecksForUpdates, to: engine)
    }

    /// True once Sparkle is attached: the updater works "at all" (the
    /// protocol's meaning). A check already in progress is Sparkle's to
    /// handle (it brings its session forward), so this does not flicker.
    var canCheckForUpdates: Bool {
        engine != nil
    }

    /// The one auto-update switch (About's checkbox). Its setter is the only
    /// writer of Sparkle's two flags besides `attach`.
    var automaticallyChecksForUpdates: Bool {
        get { settings.automaticallyChecksForUpdates }
        set {
            settings.automaticallyChecksForUpdates = newValue
            if let engine {
                apply(newValue, to: engine)
            }
        }
    }

    func checkForUpdates() {
        engine?.checkForUpdates()
    }

    /// Run the staged update's install handler. False (and nothing happens)
    /// when no update is staged or its handler can no longer install.
    @discardableResult
    func installAndRelaunch() -> Bool {
        guard let immediateInstall, immediateInstall() else {
            return false
        }
        installAttempted = true
        return true
    }

    func handle(_ event: UpdaterEvent, immediateInstall: (() -> Bool)? = nil) {
        switch event {
        case .staged:
            self.immediateInstall = immediateInstall
        case .noUpdate, .installing:
            self.immediateInstall = nil
        case .failed:
            if installAttempted {
                installAttempted = false
                onInstallFailed?()
            }
        }
        let next = updateState.applying(event)
        if next != updateState {
            updateState = next
            onStateChange?()
        }
    }

    /// Swap the install handler of the staged update without a state change
    /// (Sparkle's retry-termination handler once installing has begun).
    func replaceInstallHandler(_ handler: @escaping () -> Bool) {
        guard case .available = updateState else {
            return
        }
        immediateInstall = handler
    }

    func keepAlive(_ objects: [AnyObject]) {
        retained.append(contentsOf: objects)
    }

    private func apply(_ on: Bool, to engine: SparkleEngine) {
        engine.automaticallyChecksForUpdates = on
        engine.automaticallyDownloadsUpdates = on
    }
}

// MARK: - Headless user driver

/// What the headless driver answers when Sparkle shows an update found.
enum FoundUpdatePolicy: Equatable {
    /// Reply Install now: download (and extract) it, no window.
    case download
    /// Hold the reply as the Install and Relaunch handler: it is already
    /// downloaded and installing, and replying Install would quit at once.
    case holdForInstallAndRelaunch
    /// Reply Dismiss: an information-only item has nothing to install and
    /// no place in the approved About window.
    case dismiss

    static func decide(informationOnly: Bool, alreadyInstalling: Bool) -> FoundUpdatePolicy {
        if informationOnly {
            return .dismiss
        }
        return alreadyInstalling ? .holdForInstallAndRelaunch : .download
    }
}

/// A one-shot wrapper for a Sparkle reply: true the first time it runs.
final class OneShot {
    private var work: (() -> Void)?
    init(_ work: @escaping () -> Void) { self.work = work }
    func run() -> Bool {
        guard let work else {
            return false
        }
        self.work = nil
        work()
        return true
    }
}

/// The SPUUserDriver that shows NO window (P8 r1, overseer decision 1): every
/// Sparkle "show" becomes an UpdaterEvent for the approved About window, or
/// nothing. A found update is downloaded; once ready, the held Install reply
/// is About's Install and Relaunch, and the update also installs on quit.
/// No update found / an error: acknowledged at once, nothing shown.
@MainActor
final class HeadlessUserDriver: NSObject, SPUUserDriver {
    var onEvent: (UpdaterEvent, (() -> Bool)?) -> Void = { _, _ in }
    var onRetryHandler: (@escaping () -> Bool) -> Void = { _ in }
    /// The auto-check switch, for the (suppressed) permission request.
    var automaticChecks: () -> Bool = { true }
    private var item: SUAppcastItem?

    private func staged(_ handler: @escaping () -> Bool) {
        guard let item else {
            return
        }
        onEvent(.staged(version: SparkleDelegateBridge.version(of: item), releaseNotes: SparkleDelegateBridge.notes(of: item)), handler)
    }

    func show(_ request: SPUUpdatePermissionRequest, reply: @escaping (SUUpdatePermissionResponse) -> Void) {
        reply(SUUpdatePermissionResponse(automaticUpdateChecks: automaticChecks(), sendSystemProfile: false))
    }

    func showUserInitiatedUpdateCheck(cancellation: @escaping () -> Void) {}

    func showUpdateFound(with appcastItem: SUAppcastItem, state: SPUUserUpdateState, reply: @escaping (SPUUserUpdateChoice) -> Void) {
        item = appcastItem
        switch FoundUpdatePolicy.decide(informationOnly: appcastItem.isInformationOnlyUpdate, alreadyInstalling: state.stage == .installing) {
        case .download:
            reply(.install)
        case .holdForInstallAndRelaunch:
            let once = OneShot { reply(.install) }
            staged { once.run() }
        case .dismiss:
            reply(.dismiss)
        }
    }

    func showUpdateReleaseNotes(with downloadData: SPUDownloadData) {}
    func showUpdateReleaseNotesFailedToDownloadWithError(_ error: Error) {}

    func showUpdateNotFoundWithError(_ error: Error, acknowledgement: @escaping () -> Void) {
        acknowledgement()
        onEvent(.noUpdate, nil)
    }

    func showUpdaterError(_ error: Error, acknowledgement: @escaping () -> Void) {
        FileHandle.standardError.write(Data("LMWarden: update failed: \(error.localizedDescription)\n".utf8))
        acknowledgement()
        onEvent(.failed, nil)
    }

    func showDownloadInitiated(cancellation: @escaping () -> Void) {}
    func showDownloadDidReceiveExpectedContentLength(_ expectedContentLength: UInt64) {}
    func showDownloadDidReceiveData(ofLength length: UInt64) {}
    func showDownloadDidStartExtractingUpdate() {}
    func showExtractionReceivedProgress(_ progress: Double) {}

    /// Downloaded and extracted: the held Install reply is Install and
    /// Relaunch; left unanswered, Sparkle installs when the app quits.
    func showReady(toInstallAndRelaunch reply: @escaping (SPUUserUpdateChoice) -> Void) {
        let once = OneShot { reply(.install) }
        staged { once.run() }
    }

    /// Sparkle asked the app to quit. If the quit was cancelled, Install and
    /// Relaunch asks again through Sparkle's retry handler.
    func showInstallingUpdate(withApplicationTerminated applicationTerminated: Bool, retryTerminatingApplication: @escaping () -> Void) {
        if !applicationTerminated {
            onRetryHandler { retryTerminatingApplication(); return true }
        }
    }

    func showUpdateInstalledAndRelaunched(_ relaunched: Bool, acknowledgement: @escaping () -> Void) {
        acknowledgement()
        onEvent(.installing, nil)
    }

    func dismissUpdateInstallation() {}

    func showUpdateInFocus() {}
}

// MARK: - Live wiring

/// `SPUUpdater` behind `SparkleEngine`, hopping onto the main actor.
private final class LiveSparkleEngine: SparkleEngine {
    private let updater: SPUUpdater

    init(_ updater: SPUUpdater) {
        self.updater = updater
    }

    var automaticallyChecksForUpdates: Bool {
        get { MainActor.assumeIsolated { updater.automaticallyChecksForUpdates } }
        set { MainActor.assumeIsolated { updater.automaticallyChecksForUpdates = newValue } }
    }

    var automaticallyDownloadsUpdates: Bool {
        get { MainActor.assumeIsolated { updater.automaticallyDownloadsUpdates } }
        set { MainActor.assumeIsolated { updater.automaticallyDownloadsUpdates = newValue } }
    }

    func checkForUpdates() {
        MainActor.assumeIsolated { updater.checkForUpdates() }
    }
}

/// Sparkle's delegate callbacks, turned into `UpdaterEvent`s.
@MainActor
final class SparkleDelegateBridge: NSObject, SPUUpdaterDelegate {
    var onEvent: (UpdaterEvent, (() -> Bool)?) -> Void = { _, _ in }
    /// `--update-probe` only: the outcome of `checkForUpdateInformation`.
    var onProbe: ((String) -> Void)?

    static func version(of item: SUAppcastItem) -> String {
        item.displayVersionString
    }

    static func notes(of item: SUAppcastItem) -> URL? {
        item.releaseNotesURL ?? item.fullReleaseNotesURL
    }

    func updaterShouldPromptForPermissionToCheck(forUpdates updater: SPUUpdater) -> Bool {
        false
    }

    /// Release notes open in the browser (About's Release Notes); Sparkle
    /// never fetches them for a view of its own.
    func updater(_ updater: SPUUpdater, shouldDownloadReleaseNotesForUpdate updateItem: SUAppcastItem) -> Bool {
        false
    }

    /// The silent (scheduled, auto-downloaded) path: staged to install on
    /// quit; the handler may be invoked again after a cancelled quit (2.3+).
    func updater(
        _ updater: SPUUpdater,
        willInstallUpdateOnQuit item: SUAppcastItem,
        immediateInstallationBlock immediateInstallHandler: @escaping () -> Void
    ) -> Bool {
        onEvent(.staged(version: Self.version(of: item), releaseNotes: Self.notes(of: item)), { immediateInstallHandler(); return true })
        return true
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        onProbe?("available \(item.versionString) \(item.displayVersionString)")
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        onProbe?("none")
    }

    func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
        let code = (error as NSError).code
        // SUNoUpdateError is reported here too; it is not a failure, and
        // updaterDidNotFindUpdate already answered the probe for it.
        guard code != Int(SUError.noUpdateError.rawValue) else {
            return
        }
        onEvent(.failed, nil)
        onProbe?("error \(code) \(error.localizedDescription)")
    }
}

enum SparkleUpdaterFactory {
    /// The updater the GUI uses: Sparkle (SPUUpdater + the headless driver)
    /// when the bundle is configured for it and the updater starts, else
    /// `NoUpdater`. The returned object owns the Sparkle objects.
    @MainActor
    static func make(settings: ShellSettings, bundle: Bundle = .main) -> UpdaterControlling {
        guard UpdaterConfiguration.isEnabled(info: bundle.infoDictionary) else {
            return NoUpdater()
        }
        let updater = SparkleUpdater(settings: settings)
        let bridge = SparkleDelegateBridge()
        let driver = HeadlessUserDriver()
        let forward: (UpdaterEvent, (() -> Bool)?) -> Void = { [weak updater] event, handler in
            updater?.handle(event, immediateInstall: handler)
        }
        bridge.onEvent = forward
        driver.onEvent = forward
        driver.onRetryHandler = { [weak updater] handler in
            updater?.replaceInstallHandler(handler)
        }
        driver.automaticChecks = { settings.automaticallyChecksForUpdates }
        let sparkle = SPUUpdater(hostBundle: bundle, applicationBundle: bundle, userDriver: driver, delegate: bridge)
        updater.keepAlive([sparkle, driver, bridge])
        updater.attach(LiveSparkleEngine(sparkle))
        do {
            try sparkle.start()
        } catch {
            FileHandle.standardError.write(Data("LMWarden: updater disabled: \(error.localizedDescription)\n".utf8))
            return NoUpdater()
        }
        return updater
    }
}

// MARK: - --update-probe

/// `LMWarden --update-probe` (run from inside the .app): one informational
/// check against the bundle's own feed, no UI, nothing downloaded or
/// installed. Prints one line and exits:
///   `update-probe: available <version> <display version>` (0),
///   `update-probe: none` (0), `update-probe: disabled` (3, no SUFeedURL /
///   SUPublicEDKey), `update-probe: error <code> <message>` (1, including a
///   30 s timeout). It never writes the auto-check settings.
enum UpdateProbe {
    @MainActor
    static func run(bundle: Bundle = .main) -> Never {
        guard UpdaterConfiguration.isEnabled(info: bundle.infoDictionary) else {
            finish("disabled", code: 3)
        }
        _ = NSApplication.shared
        let bridge = SparkleDelegateBridge()
        bridge.onProbe = { line in
            finish(line, code: line.hasPrefix("error") ? 1 : 0)
        }
        let driver = HeadlessUserDriver()
        let updater = SPUUpdater(hostBundle: bundle, applicationBundle: bundle, userDriver: driver, delegate: bridge)
        do {
            try updater.start()
        } catch {
            finish("error \((error as NSError).code) \(error.localizedDescription)", code: 1)
        }
        updater.checkForUpdateInformation()
        DispatchQueue.main.asyncAfter(deadline: .now() + 30) {
            finish("error timeout", code: 1)
        }
        // Keep the objects alive while the run loop serves the check.
        withExtendedLifetime((bridge, driver, updater)) {
            RunLoop.main.run()
        }
        exit(1)
    }

    private static func finish(_ line: String, code: Int32) -> Never {
        print("update-probe: \(line)")
        fflush(stdout)
        exit(code)
    }
}
