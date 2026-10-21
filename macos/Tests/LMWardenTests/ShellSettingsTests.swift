// ShellSettingsTests — the shell's persisted preferences over an injectable
// SettingsStore.
//
// Production uses UserDefaults; every test here drives ShellSettings through
// a dictionary-backed fake store so it never touches the real defaults (the
// M1 worker runs over SSH where the persistent store is shared and slow). The
// wrong-type rule, the relative-modelsDir->nil rule, and the exact key strings
// are each pinned.
import XCTest
@testable import LMWarden

/// A SettingsStore backed by a dictionary, so tests read and write values
/// without touching UserDefaults.
private final class FakeStore: SettingsStore {
    var values: [String: Any] = [:]

    func object(forKey key: String) -> Any? {
        values[key]
    }

    func set(_ value: Any?, forKey key: String) {
        if let value {
            values[key] = value
        } else {
            values[key] = nil
        }
    }
}

final class ShellSettingsTests: XCTestCase {
    func testDefaultsOnEmptyStore() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        XCTAssertEqual(settings.firstRunCompleted, false)
        XCTAssertEqual(settings.shareOnLAN, false)
        XCTAssertNil(settings.modelsDir)
        XCTAssertEqual(settings.excludeModelsFromBackup, true)
        XCTAssertEqual(settings.openAtLogin, true)
        XCTAssertEqual(settings.automaticallyChecksForUpdates, true)
        XCTAssertEqual(store.values.count, 0)
    }

    func testKeyStrings() {
        XCTAssertEqual(ShellSettings.Key.firstRunCompleted, "lmw.firstRunCompleted")
        XCTAssertEqual(ShellSettings.Key.shareOnLAN, "lmw.shareOnLAN")
        XCTAssertEqual(ShellSettings.Key.modelsDir, "lmw.modelsDir")
        XCTAssertEqual(ShellSettings.Key.excludeModelsFromBackup, "lmw.excludeModelsFromBackup")
        XCTAssertEqual(ShellSettings.Key.openAtLogin, "lmw.openAtLogin")
        XCTAssertEqual(ShellSettings.Key.automaticallyChecksForUpdates, "lmw.automaticallyChecksForUpdates")
    }

    func testFirstRunCompletedRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.firstRunCompleted = true
        XCTAssertEqual(settings.firstRunCompleted, true)
        XCTAssertEqual(store.values[ShellSettings.Key.firstRunCompleted] as? Bool, true)
    }

    func testShareOnLANRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.shareOnLAN = true
        XCTAssertEqual(settings.shareOnLAN, true)
        XCTAssertEqual(store.values[ShellSettings.Key.shareOnLAN] as? Bool, true)
    }

    func testExcludeModelsFromBackupRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.excludeModelsFromBackup = false
        XCTAssertEqual(settings.excludeModelsFromBackup, false)
        XCTAssertEqual(store.values[ShellSettings.Key.excludeModelsFromBackup] as? Bool, false)
    }

    func testOpenAtLoginRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.openAtLogin = false
        XCTAssertEqual(settings.openAtLogin, false)
        XCTAssertEqual(store.values[ShellSettings.Key.openAtLogin] as? Bool, false)
    }

    func testAutomaticallyChecksForUpdatesRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.automaticallyChecksForUpdates = false
        XCTAssertEqual(settings.automaticallyChecksForUpdates, false)
        XCTAssertEqual(store.values[ShellSettings.Key.automaticallyChecksForUpdates] as? Bool, false)
    }

    func testModelsDirAbsolutePathRoundTrip() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.modelsDir = "/tmp/models"
        XCTAssertEqual(settings.modelsDir, "/tmp/models")
        XCTAssertEqual(store.values[ShellSettings.Key.modelsDir] as? String, "/tmp/models")
    }

    func testModelsDirRootIsAbsolute() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.modelsDir = "/"
        XCTAssertEqual(settings.modelsDir, "/")
        XCTAssertEqual(store.values[ShellSettings.Key.modelsDir] as? String, "/")
    }

    func testModelsDirRelativeStoredAsNil() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.modelsDir = "models"
        XCTAssertNil(settings.modelsDir)
        XCTAssertNil(store.values[ShellSettings.Key.modelsDir])
    }

    func testModelsDirEmptyStoredAsNil() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.modelsDir = ""
        XCTAssertNil(settings.modelsDir)
        XCTAssertNil(store.values[ShellSettings.Key.modelsDir])
    }

    func testModelsDirNilRemovesStoredValue() {
        let store = FakeStore()
        let settings = ShellSettings(store: store)
        settings.modelsDir = "/tmp/models"
        XCTAssertNotNil(store.values[ShellSettings.Key.modelsDir])
        settings.modelsDir = nil
        XCTAssertNil(settings.modelsDir)
        XCTAssertNil(store.values[ShellSettings.Key.modelsDir])
    }

    func testWrongTypeStringUnderBoolKeyReadsDefault() {
        let store = FakeStore()
        store.values[ShellSettings.Key.firstRunCompleted] = "not a bool"
        store.values[ShellSettings.Key.shareOnLAN] = "not a bool"
        store.values[ShellSettings.Key.excludeModelsFromBackup] = "not a bool"
        store.values[ShellSettings.Key.openAtLogin] = "not a bool"
        store.values[ShellSettings.Key.automaticallyChecksForUpdates] = "not a bool"
        let settings = ShellSettings(store: store)
        XCTAssertEqual(settings.firstRunCompleted, false)
        XCTAssertEqual(settings.shareOnLAN, false)
        XCTAssertEqual(settings.excludeModelsFromBackup, true)
        XCTAssertEqual(settings.openAtLogin, true)
        XCTAssertEqual(settings.automaticallyChecksForUpdates, true)
    }

    func testWrongTypeUnderModelsDirKeyReadsNil() {
        let store = FakeStore()
        store.values[ShellSettings.Key.modelsDir] = 42
        let settings = ShellSettings(store: store)
        XCTAssertNil(settings.modelsDir)
    }

    func testRelativeModelsDirAlreadyInStoreReadsNil() {
        // A relative path written behind the setter's back (e.g. `defaults
        // write`) still reads as nil, so the default models folder is used.
        let store = FakeStore()
        store.values[ShellSettings.Key.modelsDir] = "models"
        let settings = ShellSettings(store: store)
        XCTAssertNil(settings.modelsDir)
    }

    func testUserDefaultsConformsToSettingsStore() {
        // A throwaway suite, wiped before and after, so a crashed earlier run
        // cannot leave shareOnLAN = true behind and no real domain is touched.
        let suite = "lmw.tests.ShellSettingsTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = ShellSettings(store: defaults)
        XCTAssertEqual(settings.shareOnLAN, false)
        settings.shareOnLAN = true
        XCTAssertEqual(settings.shareOnLAN, true)
    }
}
