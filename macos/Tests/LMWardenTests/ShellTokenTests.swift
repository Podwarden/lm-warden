// ShellTokenTests — VW_SHELL_TOKEN minted per-launch from the CSPRNG.
//
// The token value must pass the exact rule `app/system/shell_auth.py`
// enforces (fullmatch `[A-Za-z0-9_-]{32,256}`), because a token that fails it
// fails backend boot. The validity rule is pinned at the exact bound (31/32/256/257)
// and on every scalar arm: empty, each allowed class, control chars, printable
// punctuation, and a non-ASCII letter (which `Character.isLetter` would wrongly
// accept). `generate` is driven through an injectable `randomBytes` so the
// RNG is exercised exactly once, the hex shape is pinned, and the error
// paths (RNG failure, short bytes) are isolated from the real CSPRNG.
import XCTest
@testable import LMWarden

final class ShellTokenTests: XCTestCase {

    func testEnvironmentKeyIsVWShellToken() {
        XCTAssertEqual(ShellToken.environmentKey, "VW_SHELL_TOKEN")
    }

    func testIsValidAtLowerBound() {
        XCTAssertTrue(ShellToken.isValid(String(repeating: "a", count: 32)))
    }

    func testIsValidJustBelowLowerBound() {
        XCTAssertFalse(ShellToken.isValid(String(repeating: "a", count: 31)))
    }

    func testIsValidAtUpperBound() {
        XCTAssertTrue(ShellToken.isValid(String(repeating: "a", count: 256)))
    }

    func testIsValidJustPastUpperBound() {
        XCTAssertFalse(ShellToken.isValid(String(repeating: "a", count: 257)))
    }

    func testIsValidEmptyIsFalse() {
        XCTAssertFalse(ShellToken.isValid(""))
    }

    func testIsValidAcceptsEachAllowedClass() {
        for prefix in ["A", "z", "0", "_", "-"] {
            XCTAssertTrue(
                ShellToken.isValid(prefix + String(repeating: "a", count: 31)),
                "expected validity for prefix \(prefix)"
            )
        }
    }

    func testIsValidRejectsControlChar() {
        for prefix in ["\n", "\t"] {
            XCTAssertFalse(
                ShellToken.isValid(prefix + String(repeating: "a", count: 31)),
                "expected rejection for control char \(prefix)"
            )
        }
    }

    func testIsValidRejectsPrintablePunctuation() {
        for prefix in [" ", "=", ".", "+", "/"] {
            XCTAssertFalse(
                ShellToken.isValid(prefix + String(repeating: "a", count: 31)),
                "expected rejection for punctuation \(prefix)"
            )
        }
    }

    func testIsValidRejectsNonASCIILetter() {
        XCTAssertFalse(ShellToken.isValid("\u{e9}" + String(repeating: "a", count: 31)))
    }

    func testGenerateIs64LowercaseHex() {
        let fake = FakeRandomBytes(returnBytes: [UInt8](repeating: 0xAB, count: 32))
        let token = try? ShellToken.generate(randomBytes: fake.bytes)
        XCTAssertEqual(token, String(repeating: "ab", count: 32))
    }

    func testGenerateRequestsExactly32BytesOnce() {
        let fake = FakeRandomBytes(returnBytes: [UInt8](repeating: 0, count: 32))
        _ = try? ShellToken.generate(randomBytes: fake.bytes)
        XCTAssertEqual(fake.requestedCounts, [32])
    }

    func testGeneratePropagatesRandomError() {
        let fake = FakeRandomBytes(error: CookieSecretError.randomFailed(-1))
        do {
            _ = try ShellToken.generate(randomBytes: fake.bytes)
            XCTFail("expected randomFailed to propagate")
        } catch let error as CookieSecretError {
            XCTAssertEqual(error, .randomFailed(-1))
        } catch {
            XCTFail("expected CookieSecretError.randomFailed, got \(error)")
        }
    }

    func testGenerateShortBytesThrowsGeneratedInvalid() {
        let fake = FakeRandomBytes(returnBytes: [UInt8](repeating: 0, count: 15))
        do {
            _ = try ShellToken.generate(randomBytes: fake.bytes)
            XCTFail("expected .generatedInvalid for 15 bytes (30 hex chars)")
        } catch let error as ShellTokenError {
            XCTAssertEqual(error, .generatedInvalid)
        } catch {
            XCTFail("expected ShellTokenError.generatedInvalid, got \(error)")
        }
    }

    func testGenerateSystemRandomIsValidAndDiffers() {
        let a = try? ShellToken.generate()
        let b = try? ShellToken.generate()
        XCTAssertNotNil(a)
        XCTAssertNotNil(b)
        if let a, let b {
            XCTAssertTrue(ShellToken.isValid(a))
            XCTAssertTrue(ShellToken.isValid(b))
            XCTAssertNotEqual(a, b)
        }
    }
}

/// A `randomBytes` fake: returns a fixed byte array (or throws a fixed error)
/// and records the count of every request so a test can assert the RNG was
/// asked for exactly 32 bytes, exactly once. A reference type so a single
/// instance records across the one `generate` call without copy-in/copy-out.
private final class FakeRandomBytes {
    private let returnBytes: [UInt8]
    private let error: Error?

    var requestedCounts: [Int] = []

    init(returnBytes: [UInt8] = [], error: Error? = nil) {
        self.returnBytes = returnBytes
        self.error = error
    }

    func bytes(_ count: Int) throws -> [UInt8] {
        requestedCounts.append(count)
        if let error {
            throw error
        }
        return returnBytes
    }
}
