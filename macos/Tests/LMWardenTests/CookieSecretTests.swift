// CookieSecretTests — VW_COOKIE_SECRET over an injectable KeychainStore backend.
//
// The real keychain is locked on the CI/M1 worker (SSH, no login session), so
// every test here drives the load/create path through a CookieFakeBackend and
// only exercises `systemRandomBytes` directly against the real RNG. The
// validity rule is pinned at the exact bound (31/32/33) and on every
// whitespace arm (space, tab, newline) plus the empty case.
import XCTest
@testable import LMWarden

/// A recorded write: the bytes plus the (service, account) they went to.
/// Equatable so a test can assert `backend.writes == []` after a failed path.
private struct CookieWrite: Equatable {
    let data: Data
    let service: String
    let account: String
}

/// A SecretBackend that records every call and keeps items in memory, keyed
/// by "service|account". The read result and any thrown error are injectable
/// so a test can simulate "missing", "invalid stored value", "undecodable",
/// or a raw read status error. Never touches the real keychain.
private final class CookieFakeBackend: SecretBackend {
    var items: [String: Data] = [:]
    var writes: [CookieWrite] = []
    var readCount = 0

    private let readData: () -> Data?
    private let readError: () -> Error?
    private let writeError: () -> Error?

    init(
        read: @escaping () -> Data? = { nil },
        readError: @escaping () -> Error? = { nil },
        writeError: @escaping () -> Error? = { nil }
    ) {
        self.readData = read
        self.readError = readError
        self.writeError = writeError
    }

    func read(service: String, account: String) throws -> Data? {
        readCount += 1
        if let error = readError() {
            throw error
        }
        return readData()
    }

    func write(_ data: Data, service: String, account: String) throws {
        if let error = writeError() {
            throw error
        }
        writes.append(CookieWrite(data: data, service: service, account: account))
        items["\(service)|\(account)"] = data
    }

    func delete(service: String, account: String) throws {
        items["\(service)|\(account)"] = nil
    }
}

final class CookieSecretTests: XCTestCase {
    func testIsValidBoundary() {
        XCTAssertFalse(CookieSecret.isValid(""))
        XCTAssertFalse(CookieSecret.isValid(String(repeating: "a", count: 31)))
        XCTAssertTrue(CookieSecret.isValid(String(repeating: "a", count: 32)))
        XCTAssertTrue(CookieSecret.isValid(String(repeating: "a", count: 33)))

        // One whitespace character in an otherwise-32-char secret defeats it.
        for ws in [" ", "\t", "\n"] {
            XCTAssertFalse(CookieSecret.isValid("a" + ws + String(repeating: "b", count: 30)))
        }
    }

    func testHexEncoding() {
        XCTAssertEqual(CookieSecret.hex([0x00, 0x0f, 0xa0, 0xff]), "000fa0ff")
        XCTAssertEqual(CookieSecret.hex([]), "")
    }

    func testExistingValidSecretReturnedWithoutWrite() {
        let stored = String(repeating: "c", count: 32)
        let backend = CookieFakeBackend(read: { Data(stored.utf8) })
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store)
        XCTAssertEqual(result.secret, stored)
        XCTAssertFalse(result.created)
        XCTAssertEqual(backend.writes, [])
        XCTAssertEqual(backend.readCount, 1)
    }

    func testMissingSecretGeneratedAndStored() {
        let bytes = [UInt8](0..<32).map { ($0 &* 7) & 0xff }
        let expected = CookieSecret.hex(bytes)
        let backend = CookieFakeBackend(read: { nil })
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store, randomBytes: { count in
            XCTAssertEqual(count, 32)
            return bytes
        })
        XCTAssertTrue(result.created)
        XCTAssertEqual(result.secret, expected)
        XCTAssertEqual(result.secret.count, 64)
        XCTAssertEqual(backend.writes.count, 1)
        XCTAssertEqual(backend.writes[0].account, "vw-cookie-secret")
        XCTAssertEqual(backend.writes[0].data, Data(expected.utf8))
    }

    func testShortStoredSecretReplaced() {
        let stored = String(repeating: "s", count: 31)
        let bytes = [UInt8](repeating: 0xab, count: 32)
        let expected = CookieSecret.hex(bytes)
        let backend = CookieFakeBackend(read: { Data(stored.utf8) })
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store, randomBytes: { _ in bytes })
        XCTAssertTrue(result.created)
        XCTAssertEqual(result.secret, expected)
        XCTAssertEqual(backend.writes.count, 1)
        XCTAssertEqual(backend.writes[0].account, "vw-cookie-secret")
    }

    func testWhitespaceStoredSecretReplaced() {
        // 32 characters with a single space in the middle: 31 'y' + 1 space.
        let spaced = String(repeating: "y", count: 31) + " "
        XCTAssertEqual(spaced.count, 32)
        let bytes = [UInt8](repeating: 0x11, count: 32)
        let expected = CookieSecret.hex(bytes)
        let backend = CookieFakeBackend(read: { Data(spaced.utf8) })
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store, randomBytes: { _ in bytes })
        XCTAssertTrue(result.created)
        XCTAssertEqual(result.secret, expected)
        XCTAssertEqual(backend.writes.count, 1)
    }

    func testUndecodableStoredSecretReplaced() {
        let bytes = [UInt8](repeating: 0x77, count: 32)
        let expected = CookieSecret.hex(bytes)
        let backend = CookieFakeBackend(
            read: { Data([0xFF, 0xFE, 0xFF]) },
            readError: { KeychainError.undecodable }
        )
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store, randomBytes: { _ in bytes })
        XCTAssertTrue(result.created)
        XCTAssertEqual(result.secret, expected)
        XCTAssertEqual(backend.writes.count, 1)
    }

    func testNonUTF8StoredBytesReplaced() {
        // The real decode path: the backend returns bytes that are not UTF-8,
        // KeychainStore.string(for:) throws .undecodable, and the secret is
        // regenerated and overwritten.
        let bytes = [UInt8](repeating: 0x5a, count: 32)
        let expected = CookieSecret.hex(bytes)
        let backend = CookieFakeBackend(read: { Data([0xFF, 0xFE, 0xFF]) })
        let store = KeychainStore(backend: backend)
        let result = try! CookieSecret.loadOrCreate(store: store, randomBytes: { _ in bytes })
        XCTAssertTrue(result.created)
        XCTAssertEqual(result.secret, expected)
        XCTAssertEqual(backend.writes.count, 1)
        XCTAssertEqual(backend.writes[0].data, Data(expected.utf8))
    }

    func testReadErrorPropagatesNoWrite() {
        let backend = CookieFakeBackend(readError: { KeychainError.status(-25308) })
        let store = KeychainStore(backend: backend)
        XCTAssertThrowsError(
            try CookieSecret.loadOrCreate(store: store, randomBytes: { _ in [UInt8](repeating: 1, count: 32) })
        ) {
            // A non-undecodable read error propagates unchanged (not wrapped).
            XCTAssertEqual($0 as? KeychainError, .status(-25308))
        }
        XCTAssertEqual(backend.writes, [])
    }

    func testRandomFailurePropagatesNoWrite() {
        let backend = CookieFakeBackend(read: { nil })
        let store = KeychainStore(backend: backend)
        XCTAssertThrowsError(try CookieSecret.loadOrCreate(store: store, randomBytes: { _ in
            throw CookieSecretError.randomFailed(-25300)
        })) {
            XCTAssertEqual($0 as? CookieSecretError, .randomFailed(-25300))
        }
        XCTAssertEqual(backend.writes, [])
    }

    func testWriteErrorPropagates() {
        let backend = CookieFakeBackend(
            read: { nil },
            writeError: { KeychainError.status(-34018) }
        )
        let store = KeychainStore(backend: backend)
        XCTAssertThrowsError(try CookieSecret.loadOrCreate(store: store, randomBytes: { _ in
            [UInt8](repeating: 2, count: 32)
        })) {
            XCTAssertEqual($0 as? KeychainError, .status(-34018))
        }
        XCTAssertEqual(backend.writes, [])
    }

    func testGeneratedInvalidThrowsNoWrite() {
        let backend = CookieFakeBackend(read: { nil })
        let store = KeychainStore(backend: backend)
        XCTAssertThrowsError(try CookieSecret.loadOrCreate(store: store, randomBytes: { _ in
            [UInt8](repeating: 0x00, count: 8)
        })) {
            XCTAssertEqual($0 as? CookieSecretError, .generatedInvalid)
        }
        XCTAssertEqual(backend.writes, [])
    }

    func testSystemRandomBytesLength() throws {
        let a = try CookieSecret.systemRandomBytes(32)
        XCTAssertEqual(a.count, 32)
        let b = try CookieSecret.systemRandomBytes(32)
        XCTAssertEqual(b.count, 32)
        XCTAssertNotEqual(a, b)
        XCTAssertEqual(try CookieSecret.systemRandomBytes(0), [])
    }
}
