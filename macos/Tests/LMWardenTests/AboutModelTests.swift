import XCTest
@testable import LMWarden

final class AboutModelTests: XCTestCase {
    // MARK: - make: .none

    /// No update available: the notice, primary button and notes button are all
    /// nil, and the version line carries the "(Apple Silicon)" suffix.
    func testNoneStateHidesUpdateUI() {
        let model = AboutModel.make(
            version: "1.2.3",
            update: .none,
            automaticallyChecks: true,
            canCheck: true
        )
        XCTAssertEqual(model.name, "LM Warden")
        XCTAssertEqual(model.versionLine, "Version 1.2.3 (Apple Silicon)")
        XCTAssertNil(model.updateNotice)
        XCTAssertNil(model.primaryButton)
        XCTAssertNil(model.notesButton)
        XCTAssertEqual(model.autoCheckLabel, "Check for updates automatically")
        XCTAssertEqual(model.autoCheckOn, true)
        XCTAssertEqual(model.autoCheckEnabled, true)
        XCTAssertEqual(model.footnote, "Relaunching stops running requests. You are asked first if any are in flight.")
    }

    // MARK: - make: .available with release notes

    /// An available update with release notes: notice and primary button set,
    /// and the notes button present.
    func testAvailableStateWithNotesShowsAll() {
        let notes = URL(string: "https://example.com/notes.html")!
        let model = AboutModel.make(
            version: "2.0.0",
            update: .available(version: "2.1.0", releaseNotes: notes),
            automaticallyChecks: false,
            canCheck: true
        )
        XCTAssertEqual(model.updateNotice, "A new version is available. It installs when you quit, or now.")
        XCTAssertEqual(model.primaryButton, "Install and Relaunch")
        XCTAssertEqual(model.notesButton, "Release Notes")
    }

    // MARK: - make: .available without release notes

    /// An available update with no release notes: the notice and primary button
    /// are still set, but the notes button is nil (the only difference).
    func testAvailableStateWithoutNotesHidesNotesButton() {
        let model = AboutModel.make(
            version: "2.0.0",
            update: .available(version: "2.1.0", releaseNotes: nil),
            automaticallyChecks: true,
            canCheck: false
        )
        XCTAssertEqual(model.updateNotice, "A new version is available. It installs when you quit, or now.")
        XCTAssertEqual(model.primaryButton, "Install and Relaunch")
        XCTAssertNil(model.notesButton)
    }

    // MARK: - auto check flag and enabled

    /// `automaticallyChecks` drives `autoCheckOn` (here false, the .none arm).
    func testAutoCheckOnReflectsAutomaticallyChecksFalse() {
        let model = AboutModel.make(
            version: "1.0.0",
            update: .none,
            automaticallyChecks: false,
            canCheck: true
        )
        XCTAssertEqual(model.autoCheckOn, false)
    }

    /// `canCheck` false disables the checkbox (`autoCheckEnabled` is false),
    /// independent of `automaticallyChecks`.
    func testAutoCheckEnabledFalseWhenCannotCheck() {
        let model = AboutModel.make(
            version: "1.0.0",
            update: .none,
            automaticallyChecks: true,
            canCheck: false
        )
        XCTAssertEqual(model.autoCheckOn, true)
        XCTAssertEqual(model.autoCheckEnabled, false)
    }

    /// `canCheck` true enables the checkbox.
    func testAutoCheckEnabledTrueWhenCanCheck() {
        let model = AboutModel.make(
            version: "1.0.0",
            update: .none,
            automaticallyChecks: true,
            canCheck: true
        )
        XCTAssertEqual(model.autoCheckOn, true)
        XCTAssertEqual(model.autoCheckEnabled, true)
    }

    // MARK: - version line interpolation

    /// The version string is interpolated verbatim into the version line.
    func testVersionLineInterpolatesVersion() {
        let model = AboutModel.make(
            version: "0.9.0-beta.1",
            update: .none,
            automaticallyChecks: true,
            canCheck: true
        )
        XCTAssertEqual(model.versionLine, "Version 0.9.0-beta.1 (Apple Silicon)")
    }

    // MARK: - UpdateState Equatable

    /// `.available` equality is value-based, so a rebuilt `.none` equals itself
    /// and an `.available` compares against its fields.
    func testUpdateStateEquatable() {
        let notes = URL(string: "https://example.com/notes.html")!
        let available = UpdateState.available(version: "2.1.0", releaseNotes: notes)
        let same = UpdateState.available(version: "2.1.0", releaseNotes: notes)
        let different = UpdateState.available(version: "2.2.0", releaseNotes: notes)
        let noNotes = UpdateState.available(version: "2.1.0", releaseNotes: nil)
        XCTAssertEqual(available, same)
        XCTAssertNotEqual(available, different)
        XCTAssertNotEqual(available, noNotes)
        XCTAssertEqual(UpdateState.none, UpdateState.none)
        XCTAssertNotEqual(UpdateState.none, available)
    }

    // MARK: - AboutModel Equatable

    /// Two models built from the same inputs are equal.
    func testAboutModelEquatableSameInputs() {
        let a = AboutModel.make(version: "1.0.0", update: .none, automaticallyChecks: true, canCheck: true)
        let b = AboutModel.make(version: "1.0.0", update: .none, automaticallyChecks: true, canCheck: true)
        XCTAssertEqual(a, b)
    }

    /// Differing inputs yield non-equal models.
    func testAboutModelEquatableDifferentInputs() {
        let a = AboutModel.make(version: "1.0.0", update: .none, automaticallyChecks: true, canCheck: true)
        let b = AboutModel.make(version: "1.0.1", update: .none, automaticallyChecks: true, canCheck: true)
        XCTAssertNotEqual(a, b)
    }

    // MARK: - NoUpdater

    /// The no-op updater: `canCheckForUpdates` is false, the auto-check flag is
    /// stored (settable), and both methods are no-ops that can be called safely.
    func testNoUpdaterDefaultsAndStoredFlag() {
        let updater = NoUpdater()
        XCTAssertFalse(updater.canCheckForUpdates)
        // `automaticallyChecksForUpdates` is stored and settable.
        XCTAssertFalse(updater.automaticallyChecksForUpdates)
        updater.automaticallyChecksForUpdates = true
        XCTAssertTrue(updater.automaticallyChecksForUpdates)
        // The methods are no-ops: they run without throwing or side effects.
        updater.checkForUpdates()
        updater.installAndRelaunch()
    }

    /// `NoUpdater` conforms to `UpdaterControlling`, so it can be held behind
    /// the protocol (the seam the About view will use once Sparkle lands).
    func testNoUpdaterConformsToProtocol() {
        let updater: UpdaterControlling = NoUpdater()
        XCTAssertFalse(updater.canCheckForUpdates)
        updater.automaticallyChecksForUpdates = true
        XCTAssertTrue(updater.automaticallyChecksForUpdates)
    }
}
