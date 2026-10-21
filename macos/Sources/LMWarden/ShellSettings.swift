// ShellSettings — the shell's persisted preferences over an injectable
// SettingsStore.
//
// Production uses UserDefaults; tests inject a dictionary-backed fake so they
// never touch the real defaults. Every key is prefixed "lmw.". A stored value
// of the wrong type reads as the default (a String under a Bool key reads the
// Bool default), and a `modelsDir` that does not start with "/" is stored as
// nil (nil = use the default models folder from AppPaths).
import Foundation

/// The seam between ShellSettings and any key/value store. UserDefaults
/// already satisfies both methods, so `extension UserDefaults: SettingsStore {}`
/// is a conformance with no new code; tests use a dictionary-backed fake.
protocol SettingsStore: AnyObject {
    func object(forKey key: String) -> Any?
    func set(_ value: Any?, forKey key: String)
}

extension UserDefaults: SettingsStore {}

final class ShellSettings {
    /// The exact key strings, all prefixed "lmw.".
    enum Key {
        static let firstRunCompleted = "lmw.firstRunCompleted"
        static let shareOnLAN = "lmw.shareOnLAN"
        static let modelsDir = "lmw.modelsDir"
        static let excludeModelsFromBackup = "lmw.excludeModelsFromBackup"
        static let openAtLogin = "lmw.openAtLogin"
        static let automaticallyChecksForUpdates = "lmw.automaticallyChecksForUpdates"
    }

    private let store: SettingsStore

    init(store: SettingsStore) {
        self.store = store
    }

    /// A stored value of the wrong type reads as `default`: a non-Bool under
    /// a Bool key falls through to `default` rather than throwing.
    private func bool(_ key: String, default: Bool) -> Bool {
        store.object(forKey: key) as? Bool ?? `default`
    }

    /// A stored value of the wrong type (a non-String, or a String that does
    /// not start with "/") reads as nil (nil = use the default models folder).
    private func string(_ key: String) -> String? {
        guard let value = store.object(forKey: key) as? String,
              value.hasPrefix("/")
        else {
            return nil
        }
        return value
    }

    var firstRunCompleted: Bool {
        get { bool(Key.firstRunCompleted, default: false) }
        set { store.set(newValue, forKey: Key.firstRunCompleted) }
    }

    var shareOnLAN: Bool {
        get { bool(Key.shareOnLAN, default: false) }
        set { store.set(newValue, forKey: Key.shareOnLAN) }
    }

    /// nil = use the default models folder (AppPaths). A value that does not
    /// start with "/" is stored as nil (removed), so the default is used.
    var modelsDir: String? {
        get { string(Key.modelsDir) }
        set {
            if let newValue, newValue.hasPrefix("/") {
                store.set(newValue, forKey: Key.modelsDir)
            } else {
                store.set(nil, forKey: Key.modelsDir)
            }
        }
    }

    var excludeModelsFromBackup: Bool {
        get { bool(Key.excludeModelsFromBackup, default: true) }
        set { store.set(newValue, forKey: Key.excludeModelsFromBackup) }
    }

    var openAtLogin: Bool {
        get { bool(Key.openAtLogin, default: true) }
        set { store.set(newValue, forKey: Key.openAtLogin) }
    }

    var automaticallyChecksForUpdates: Bool {
        get { bool(Key.automaticallyChecksForUpdates, default: true) }
        set { store.set(newValue, forKey: Key.automaticallyChecksForUpdates) }
    }
}
