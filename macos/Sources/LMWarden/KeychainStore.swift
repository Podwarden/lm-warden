// KeychainStore — validated string secrets over an injectable SecretBackend.
//
// The store keeps *string* secrets (hub tokens, etc.) and never leaks the raw
// Data out of a read that was not UTF-8. Validation of the account runs
// before any backend call so a rejected account is never written, read or
// deleted. `SystemKeychainBackend` is the production backend (SecItem
// generic passwords); tests inject a `FakeBackend` so they never touch the
// real keychain — the M1 worker runs over SSH where the login keychain is
// locked, and a test must not hang on a prompt.
import Foundation
import Security

enum KeychainError: Error, Equatable {
    case invalidAccount
    case undecodable
    case status(OSStatus)
}

/// The seam between KeychainStore and any persistence. A production app uses
/// `SystemKeychainBackend`; tests use an in-memory fake so the real keychain
/// (locked under SSH) is never touched.
protocol SecretBackend {
    func read(service: String, account: String) throws -> Data?
    func write(_ data: Data, service: String, account: String) throws
    func delete(service: String, account: String) throws
}

struct KeychainStore {
    /// The keychain service name: the running bundle's service per
    /// `service(forBundleIdentifier:)`. A shipped bundle (and `swift run`,
    /// `swift test`) resolves to `AppVersion.bundleIdentifier`, so the
    /// existing item is untouched; a test variant such as
    /// `com.podwarden.lmwarden.smoke` gets its own item and never reads,
    /// writes or prompts for the real one.
    static let defaultService = service(forBundleIdentifier: Bundle.main.bundleIdentifier)

    /// Bundle id -> Keychain service. The id itself when it is exactly
    /// `AppVersion.bundleIdentifier` or starts with it plus "." (a test
    /// variant); anything else (nil, a foreign id, the bare prefix without the
    /// dot such as `com.podwarden.lmwardenx`) is `AppVersion.bundleIdentifier`.
    static func service(forBundleIdentifier bundleIdentifier: String?) -> String {
        let app = AppVersion.bundleIdentifier
        guard let bundleIdentifier,
              bundleIdentifier == app || bundleIdentifier.hasPrefix(app + ".")
        else {
            return app
        }
        return bundleIdentifier
    }

    let service: String
    let backend: SecretBackend

    init(backend: SecretBackend, service: String = KeychainStore.defaultService) {
        self.backend = backend
        self.service = service
    }

    /// An account is a printable string of at most 128 Characters. Empty,
    /// longer than 128, or containing any control character (`\n`, `\t`,
    /// `\u{0}`, …) is rejected before it reaches the backend.
    static func validate(account: String) throws {
        guard !account.isEmpty,
              account.count <= 128,
              // Scalars, not UTF-16 units: controlCharacters (Cc + Cf) has
              // members outside the BMP, e.g. U+E0001 LANGUAGE TAG.
              !account.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
        else {
            throw KeychainError.invalidAccount
        }
    }

    func string(for account: String) throws -> String? {
        try KeychainStore.validate(account: account)
        guard let data = try backend.read(service: service, account: account) else {
            return nil
        }
        guard let value = String(data: data, encoding: .utf8) else {
            throw KeychainError.undecodable
        }
        return value
    }

    func set(_ value: String, for account: String) throws {
        try KeychainStore.validate(account: account)
        try backend.write(Data(value.utf8), service: service, account: account)
    }

    func remove(account: String) throws {
        try KeychainStore.validate(account: account)
        try backend.delete(service: service, account: account)
    }
}

/// The production `SecretBackend`: SecItem generic passwords, one item per
/// (service, account) pair.
struct SystemKeychainBackend: SecretBackend {
    func read(service: String, account: String) throws -> Data? {
        var query = Self.query(service: service, account: account)
        query[kSecReturnData] = true
        query[kSecMatchLimit] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(
            query as CFDictionary,
            &result
        )
        guard try Self.check(status, notFoundIsOK: true) else {
            return nil
        }
        return result as? Data
    }

    func write(_ data: Data, service: String, account: String) throws {
        let query = Self.query(service: service, account: account)
        let attributes: [CFString: Data] = [kSecValueData: data]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecSuccess {
            return
        }
        // errSecItemNotFound means there is nothing to update yet: add. Any
        // other status (including notFoundIsOK false) throws from check.
        _ = try Self.check(status, notFoundIsOK: true)
        try Self.add(data: data, service: service, account: account)
    }

    func delete(service: String, account: String) throws {
        let query = Self.query(service: service, account: account)
        let status = SecItemDelete(query as CFDictionary)
        // Not-found on a delete is success: there is nothing to remove.
        _ = try Self.check(status, notFoundIsOK: true)
    }

    private static func query(service: String, account: String) -> [CFString: Any] {
        [
            kSecClass: kSecClassGenericPassword,
            kSecAttrService: service,
            kSecAttrAccount: account,
        ]
    }

    private static func add(data: Data, service: String, account: String) throws {
        var query = Self.query(service: service, account: account)
        query[kSecValueData] = data
        let status = SecItemAdd(query as CFDictionary, nil)
        _ = try Self.check(status, notFoundIsOK: false)
    }

    /// Maps a SecItem OSStatus onto Bool / thrown KeychainError.
    ///
    /// - Returns `true` for `errSecSuccess`.
    /// - Returns `false` for `errSecItemNotFound` when `notFoundIsOK` is set
    ///   (read: "no item", delete: "already gone").
    /// - Throws `KeychainError.status(status)` for anything else, including
    ///   `errSecItemNotFound` when `notFoundIsOK` is false (e.g. a write that
    ///   could not update an existing item and is not a first add).
    static func check(_ status: OSStatus, notFoundIsOK: Bool) throws -> Bool {
        switch status {
        case errSecSuccess:
            return true
        case errSecItemNotFound where notFoundIsOK:
            return false
        default:
            throw KeychainError.status(status)
        }
    }
}
