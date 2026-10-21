// ShellToken — VW_SHELL_TOKEN minted per-launch from the system CSPRNG.
//
// The macOS shell injects `VW_SHELL_TOKEN` into the backend's environment; the
// backend (app/system/shell_auth.py) fails boot unless the value fullmatches
// `[A-Za-z0-9_-]{32,256}`. `generate` mints 32 CSPRNG bytes and hex-encodes
// them to 64 lowercase hex chars, which always satisfies the rule. It is
// per-launch only: nothing is persisted (no Keychain, no UserDefaults, no
// file), and nothing is logged. `isValid` is the same rule on the Swift side,
// so a minted token can be re-validated before it is injected.
import Foundation

enum ShellTokenError: Error, Equatable {
    case generatedInvalid
}

enum ShellToken {
    /// The environment variable the backend reads for the bearer.
    static let environmentKey = "VW_SHELL_TOKEN"

    /// Inclusive length bounds, mirroring `shell_auth._TOKEN_RE` (`{32,256}`).
    static let minLength = 32
    static let maxLength = 256

    /// True iff the token is 32–256 unicode scalars and every scalar is one of
    /// ASCII A-Z, a-z, 0-9, '_', or '-'.
    ///
    /// The check is on `unicodeScalars.value` ranges, not
    /// `Character.isLetter`/`isNumber`, which accept non-ASCII letters such as
    /// `\u{e9}` and would let a token past that `shell_auth` rejects.
    static func isValid(_ token: String) -> Bool {
        guard token.unicodeScalars.count >= minLength,
              token.unicodeScalars.count <= maxLength
        else {
            return false
        }
        for scalar in token.unicodeScalars {
            let value = scalar.value
            let isUpper = (65...90).contains(value)
            let isLower = (97...122).contains(value)
            let isDigit = (48...57).contains(value)
            let isUnderscore = value == 95
            let isDash = value == 45
            if !(isUpper || isLower || isDigit || isUnderscore || isDash) {
                return false
            }
        }
        return true
    }

    /// Mint a token: `randomBytes(32)` exactly once, hex-encoded to 64
    /// lowercase hex chars. If the result fails `isValid` (defensive — the hex
    /// of 32 bytes always passes) throws `.generatedInvalid`. Any error from
    /// `randomBytes` (e.g. `CookieSecretError.randomFailed`) propagates
    /// unchanged. Nothing is persisted or logged.
    static func generate(
        randomBytes: (Int) throws -> [UInt8] = CookieSecret.systemRandomBytes
    ) throws -> String {
        let bytes = try randomBytes(32)
        let token = CookieSecret.hex(bytes)
        guard isValid(token) else {
            throw ShellTokenError.generatedInvalid
        }
        return token
    }
}
