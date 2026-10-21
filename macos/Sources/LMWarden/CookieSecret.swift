// CookieSecret — VW_COOKIE_SECRET over a KeychainStore.
//
// On first run (no stored value, or a stored value that fails the validity
// rule) a 64-hex-char secret is generated from the system CSPRNG and written
// to the keychain under `CookieSecret.account`. An already-valid stored value
// is returned as-is with no write. `loadOrCreate` is non-UI and idempotent:
// re-invocations against a valid stored secret never regenerate or rewrite.
import Foundation
import Security

enum CookieSecretError: Error, Equatable {
    case randomFailed(OSStatus)
    case generatedInvalid
}

struct CookieSecretResult: Equatable {
    let secret: String
    let created: Bool
}

enum CookieSecret {
    /// The keychain account the cookie secret is stored under.
    static let account = "vw-cookie-secret"

    /// True iff `secret` is at least 32 Characters and contains no whitespace.
    /// Mirrors the `badCookieSecret` rule in `ServerLaunchSpec` and the check
    /// in `app/config.py`, so a secret stored here always passes launch
    /// validation without re-checking downstream.
    static func isValid(_ secret: String) -> Bool {
        secret.count >= 32 && !secret.contains(where: { $0.isWhitespace })
    }

    /// Lowercase hex, two zero-padded digits per byte. 32 bytes -> 64 chars.
    static func hex(_ bytes: [UInt8]) -> String {
        var out = ""
        out.reserveCapacity(bytes.count * 2)
        for byte in bytes {
            out.append(String(format: "%02x", byte))
        }
        return out
    }

    /// `count` bytes from the system CSPRNG (SecRandomCopyBytes). Any status
    /// other than errSecSuccess is wrapped in `.randomFailed(status)` and
    /// re-thrown so the caller can distinguish "RNG refused" from "generated
    /// a value that failed validation".
    static func systemRandomBytes(_ count: Int) throws -> [UInt8] {
        guard count > 0 else { return [] }
        var bytes = [UInt8](repeating: 0, count: count)
        let status = bytes.withUnsafeMutableBytes { buffer in
            SecRandomCopyBytes(kSecRandomDefault, count, buffer.baseAddress!)
        }
        guard status == errSecSuccess else {
            throw CookieSecretError.randomFailed(status)
        }
        return bytes
    }

    /// Load a valid secret from the store, or generate one and store it.
    ///
    /// - A stored value that passes `isValid` is returned with
    ///   `created == false` and no write.
    /// - `nil` (missing), a stored value that fails `isValid`, or a read that
    ///   throws `KeychainError.undecodable` all fall through to generation:
    ///   `randomBytes(32)` is called exactly once, hex-encoded, re-validated
    ///   (a `false` here throws `.generatedInvalid` and writes nothing),
    ///   then written via `store.set(_, for: CookieSecret.account)` and
    ///   returned with `created == true`.
    /// - Any other read error, any error from `randomBytes`, and any write
    ///   error propagate unchanged; nothing is written on those paths.
    static func loadOrCreate(
        store: KeychainStore,
        randomBytes: (Int) throws -> [UInt8] = CookieSecret.systemRandomBytes
    ) throws -> CookieSecretResult {
        var stored: String?
        do {
            stored = try store.string(for: account)
        } catch let error as KeychainError where error == .undecodable {
            stored = nil
        }
        if let stored, isValid(stored) {
            return CookieSecretResult(secret: stored, created: false)
        }
        let bytes = try randomBytes(32)
        let secret = hex(bytes)
        guard isValid(secret) else {
            throw CookieSecretError.generatedInvalid
        }
        try store.set(secret, for: account)
        return CookieSecretResult(secret: secret, created: true)
    }
}
