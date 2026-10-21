import XCTest
@testable import LMWarden

private final class UpdaterDictStore: SettingsStore {
    var values: [String: Any] = [:]
    func object(forKey key: String) -> Any? { values[key] }
    func set(_ value: Any?, forKey key: String) { values[key] = value }
}

private final class FakeEngine: SparkleEngine {
    var automaticallyChecksForUpdates = false { didSet { writes += 1 } }
    var automaticallyDownloadsUpdates = false { didSet { writes += 1 } }
    var writes = 0
    var checks = 0
    func checkForUpdates() { checks += 1 }
}

/// Pins SparkleUpdater (P8 sparkle-updater): the Info.plist gate, the
/// UpdateState transitions, the one auto-check switch, Install and Relaunch.
final class SparkleUpdaterTests: XCTestCase {
    private let feed = "https://lmwarden.com/macos/appcast.xml"
    private let key = "c2l4dHktZm91ci1jaGFyYWN0ZXJzLW9mLWJhc2U2NC1rZXktLS0tLS0tLS0tLQ=="

    // MARK: - UpdaterConfiguration

    func testEnabledOnlyWithFeedAndKey() {
        XCTAssertTrue(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": feed, "SUPublicEDKey": key]))
        XCTAssertTrue(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": "http://127.0.0.1:8123/appcast.xml", "SUPublicEDKey": key]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: nil))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: [:]), "a dev build has neither key")
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": feed]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUPublicEDKey": key]))
    }

    func testMalformedValuesDisable() {
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": "", "SUPublicEDKey": key]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": "ftp://x/appcast.xml", "SUPublicEDKey": key]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": "https://", "SUPublicEDKey": key]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": feed, "SUPublicEDKey": "  "]))
        XCTAssertFalse(UpdaterConfiguration.isEnabled(info: ["SUFeedURL": feed, "SUPublicEDKey": 42]))
    }

    // MARK: - UpdateState.applying

    func testTransitions() {
        let notes = URL(string: "https://lmwarden.com/notes")!
        let available = UpdateState.available(version: "2026.11.1", releaseNotes: notes)
        XCTAssertEqual(UpdateState.none.applying(.staged(version: "2026.11.1", releaseNotes: notes)), available)
        XCTAssertEqual(available.applying(.noUpdate), .none)
        XCTAssertEqual(available.applying(.installing), .none)
        XCTAssertEqual(available.applying(.failed), available, "a failed cycle keeps the staged update")
        XCTAssertEqual(UpdateState.none.applying(.failed), .none)
        let newer = available.applying(.staged(version: "2026.11.2", releaseNotes: nil))
        XCTAssertEqual(newer, .available(version: "2026.11.2", releaseNotes: nil))
    }

    // MARK: - SparkleUpdater

    private func make(stored: Bool? = nil) -> (SparkleUpdater, UpdaterDictStore) {
        let store = UpdaterDictStore()
        if let stored {
            store.values[ShellSettings.Key.automaticallyChecksForUpdates] = stored
        }
        return (SparkleUpdater(settings: ShellSettings(store: store)), store)
    }

    func testCannotCheckUntilAttached() {
        let (updater, _) = make()
        XCTAssertFalse(updater.canCheckForUpdates)
        updater.checkForUpdates()
        let engine = FakeEngine()
        updater.attach(engine)
        XCTAssertTrue(updater.canCheckForUpdates)
        updater.checkForUpdates()
        XCTAssertEqual(engine.checks, 1)
    }

    /// The setting is the source of truth: attach pushes it (default on) into
    /// both Sparkle flags, whatever Sparkle held before.
    func testAttachPushesTheSettingIntoSparkle() {
        let (updater, _) = make()
        let engine = FakeEngine()
        updater.attach(engine)
        XCTAssertTrue(engine.automaticallyChecksForUpdates)
        XCTAssertTrue(engine.automaticallyDownloadsUpdates)

        let (off, _) = make(stored: false)
        let other = FakeEngine()
        other.automaticallyChecksForUpdates = true
        other.automaticallyDownloadsUpdates = true
        other.writes = 0
        off.attach(other)
        XCTAssertEqual(other.writes, 2)
        XCTAssertFalse(other.automaticallyChecksForUpdates)
        XCTAssertFalse(other.automaticallyDownloadsUpdates)
        XCTAssertFalse(off.automaticallyChecksForUpdates)
    }

    func testCheckboxWritesTheSettingAndSparkle() {
        let (updater, store) = make()
        let engine = FakeEngine()
        updater.attach(engine)
        updater.automaticallyChecksForUpdates = false
        XCTAssertEqual(store.values[ShellSettings.Key.automaticallyChecksForUpdates] as? Bool, false)
        XCTAssertFalse(engine.automaticallyChecksForUpdates)
        XCTAssertFalse(engine.automaticallyDownloadsUpdates)
        // Sparkle drifting on its own does not change what the checkbox reads.
        engine.automaticallyChecksForUpdates = true
        XCTAssertFalse(updater.automaticallyChecksForUpdates)
        updater.automaticallyChecksForUpdates = true
        XCTAssertTrue(engine.automaticallyChecksForUpdates)
        XCTAssertTrue(engine.automaticallyDownloadsUpdates)
    }

    func testStagedUpdateIsAvailableAndInstallsOnDemand() {
        let (updater, _) = make()
        var changes = 0
        var installs = 0
        updater.onStateChange = { changes += 1 }
        updater.installAndRelaunch()
        XCTAssertEqual(installs, 0)

        updater.handle(.staged(version: "2026.11.1", releaseNotes: nil), immediateInstall: { installs += 1; return true })
        XCTAssertEqual(updater.updateState, .available(version: "2026.11.1", releaseNotes: nil))
        XCTAssertEqual(changes, 1)
        // The same event again changes nothing visible.
        updater.handle(.staged(version: "2026.11.1", releaseNotes: nil), immediateInstall: { installs += 1; return true })
        XCTAssertEqual(changes, 1)

        // A cancelled termination lets the handler run again (Sparkle 2.3+).
        updater.installAndRelaunch()
        updater.installAndRelaunch()
        XCTAssertEqual(installs, 2)

        // A failed cycle keeps the staged update and its handler.
        updater.handle(.failed)
        XCTAssertEqual(changes, 1)
        updater.installAndRelaunch()
        XCTAssertEqual(installs, 3)
    }

    func testNoUpdateOrInstallingDropsTheHandler() {
        for event in [UpdaterEvent.noUpdate, .installing] {
            let (updater, _) = make()
            var installs = 0
            var changes = 0
            updater.onStateChange = { changes += 1 }
            updater.handle(.staged(version: "2", releaseNotes: nil), immediateInstall: { installs += 1; return true })
            updater.handle(event)
            XCTAssertEqual(updater.updateState, .none, "\(event)")
            XCTAssertEqual(changes, 2, "\(event)")
            updater.installAndRelaunch()
            XCTAssertEqual(installs, 0, "\(event)")
        }
    }

    /// The About model over a staged update shows exactly the approved copy.
    func testStagedUpdateDrivesTheApprovedAboutCopy() {
        let (updater, _) = make()
        updater.attach(FakeEngine())
        let notes = URL(string: "https://lmwarden.com/notes")!
        updater.handle(.staged(version: "2026.11.1", releaseNotes: notes), immediateInstall: { true })
        let model = AboutModel.make(
            version: "2026.10.1",
            update: updater.updateState,
            automaticallyChecks: updater.automaticallyChecksForUpdates,
            canCheck: updater.canCheckForUpdates
        )
        XCTAssertEqual(model.updateNotice, "A new version is available. It installs when you quit, or now.")
        XCTAssertEqual(model.primaryButton, "Install and Relaunch")
        XCTAssertEqual(model.notesButton, "Release Notes")
        XCTAssertTrue(model.autoCheckOn)
        XCTAssertTrue(model.autoCheckEnabled)
    }

    /// A dev build (NoUpdater) adds nothing: no notice, no buttons, the
    /// checkbox disabled.
    func testDevBuildShowsNothingExtra() {
        let updater: UpdaterControlling = NoUpdater()
        XCTAssertEqual(updater.updateState, .none)
        let model = AboutModel.make(
            version: "dev",
            update: updater.updateState,
            automaticallyChecks: updater.automaticallyChecksForUpdates,
            canCheck: updater.canCheckForUpdates
        )
        XCTAssertNil(model.updateNotice)
        XCTAssertNil(model.primaryButton)
        XCTAssertNil(model.notesButton)
        XCTAssertFalse(model.autoCheckEnabled)
    }

    // MARK: - r1: install outcome, pre-confirmation, the one switch

    func testInstallReportsWhetherItRan() {
        let (updater, _) = make()
        XCTAssertFalse(updater.installAndRelaunch(), "nothing staged")
        let once = OneShot {}
        updater.handle(.staged(version: "2", releaseNotes: nil), immediateInstall: { once.run() })
        XCTAssertTrue(updater.installAndRelaunch())
        XCTAssertFalse(updater.installAndRelaunch(), "a used one-shot reply installs nothing")
        XCTAssertFalse(NoUpdater().installAndRelaunch())
    }

    func testFailureAfterAnInstallAttemptReportsOnce() {
        let (updater, _) = make()
        var failed = 0
        updater.onInstallFailed = { failed += 1 }
        updater.handle(.staged(version: "2", releaseNotes: nil), immediateInstall: { true })
        updater.handle(.failed)
        XCTAssertEqual(failed, 0, "no install was attempted")
        XCTAssertTrue(updater.installAndRelaunch())
        updater.handle(.failed)
        updater.handle(.failed)
        XCTAssertEqual(failed, 1)
    }

    func testRetryHandlerReplacesOnlyAStagedHandler() {
        let (updater, _) = make()
        var retries = 0
        updater.replaceInstallHandler { retries += 1; return true }
        XCTAssertFalse(updater.installAndRelaunch(), "nothing staged: nothing replaced")
        updater.handle(.staged(version: "2", releaseNotes: nil), immediateInstall: { false })
        var changes = 0
        updater.onStateChange = { changes += 1 }
        updater.replaceInstallHandler { retries += 1; return true }
        XCTAssertTrue(updater.installAndRelaunch())
        XCTAssertEqual(retries, 1)
        XCTAssertEqual(changes, 0, "no visible change")
    }

    /// About's install helper pre-confirms first and withdraws when the
    /// install did not start.
    func testAboutInstallWithdrawsWhenNothingRan() {
        let (updater, _) = make()
        var calls: [String] = []
        XCTAssertFalse(AboutWindowController.install(updater, confirmQuit: { calls.append("confirm") }, withdrawQuit: { calls.append("withdraw") }))
        XCTAssertEqual(calls, ["confirm", "withdraw"])
        calls = []
        updater.handle(.staged(version: "2", releaseNotes: nil), immediateInstall: { calls.append("install"); return true })
        XCTAssertTrue(AboutWindowController.install(updater, confirmQuit: { calls.append("confirm") }, withdrawQuit: { calls.append("withdraw") }))
        XCTAssertEqual(calls, ["confirm", "install"])
    }

    func testFoundUpdatePolicy() {
        XCTAssertEqual(FoundUpdatePolicy.decide(informationOnly: false, alreadyInstalling: false), .download)
        XCTAssertEqual(FoundUpdatePolicy.decide(informationOnly: false, alreadyInstalling: true), .holdForInstallAndRelaunch)
        XCTAssertEqual(FoundUpdatePolicy.decide(informationOnly: true, alreadyInstalling: false), .dismiss)
        XCTAssertEqual(FoundUpdatePolicy.decide(informationOnly: true, alreadyInstalling: true), .dismiss)
    }

    func testOneShotRunsOnce() {
        var runs = 0
        let once = OneShot { runs += 1 }
        XCTAssertTrue(once.run())
        XCTAssertFalse(once.run())
        XCTAssertEqual(runs, 1)
    }

    /// Behaviour: nothing but attach and the auto-check setter writes either
    /// Sparkle flag (checks, staged/failed/no-update events, installs do not).
    func testOnlyTheSwitchWritesSparkleFlags() {
        let (updater, _) = make()
        let engine = FakeEngine()
        updater.attach(engine)
        XCTAssertEqual(engine.writes, 2)
        updater.checkForUpdates()
        for event in [UpdaterEvent.staged(version: "2", releaseNotes: nil), .failed, .noUpdate, .installing] {
            updater.handle(event, immediateInstall: { true })
        }
        _ = updater.installAndRelaunch()
        _ = updater.automaticallyChecksForUpdates
        XCTAssertEqual(engine.writes, 2)
        updater.automaticallyChecksForUpdates = false
        XCTAssertEqual(engine.writes, 4)
    }

    /// Source scan: no other file names Sparkle's download flag, and inside
    /// SparkleUpdater.swift the flags are assigned only by `apply` (engine.)
    /// and the live engine's setters (updater. ... = newValue).
    func testOnlySparkleUpdaterWritesSparkleFlagsInSource() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/LMWarden")
        let files = try FileManager.default.contentsOfDirectory(atPath: sources.path).filter { $0.hasSuffix(".swift") }
        XCTAssertTrue(files.contains("SparkleUpdater.swift"))
        let assignment = try NSRegularExpression(pattern: #"(\w+)\.automatically(Checks|Downloads)(ForUpdates|Updates)\s*=[^=]"#)
        for file in files {
            let text = try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8)
            let matches = assignment.matches(in: text, range: NSRange(text.startIndex..., in: text)).map {
                (text as NSString).substring(with: $0.range(at: 1))
            }
            // No Sparkle stock UI anywhere (r1, overseer decision 1).
            // Any mention counts (`.init(`, `X (`, `X.self`, a typealias), and
            // the legacy SUUpdater (Sparkle 1 API, also the stock UI).
            let stock = try NSRegularExpression(pattern: #"\b(SPUStandardUpdaterController|SPUStandardUserDriver|SUUpdater)\b"#)
            let used = stock.matches(in: text, range: NSRange(text.startIndex..., in: text)).map {
                (text as NSString).substring(with: $0.range)
            }
            XCTAssertEqual(used, [], "\(file) names Sparkle's stock UI")
            if file == "SparkleUpdater.swift" {
                XCTAssertEqual(matches.sorted(), ["engine", "engine", "settings", "updater", "updater"], file)
            } else {
                XCTAssertFalse(text.contains("automaticallyDownloadsUpdates"), file)
                // About's checkbox writes the protocol's switch (SparkleUpdater's
                // setter); NoUpdater's init sets its own stored flag.
                let allowed = file == "AboutModel.swift" ? ["self"] : file == "AboutWindow.swift" ? ["updater"] : []
                XCTAssertTrue(matches.allSatisfy { allowed.contains($0) }, "\(file): \(matches)")
            }
        }
    }
}
