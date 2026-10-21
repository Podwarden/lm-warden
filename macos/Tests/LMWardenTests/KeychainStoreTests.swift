// KeychainStoreTests — string secrets over an injectable SecretBackend.
//
// The real keychain is locked on the CI/M1 worker (SSH, no login session), so
// every test here drives the store through a FakeBackend and only exercises
// SystemKeychainBackend's pure `check` helper. The store's account validation
// is pinned at the exact bounds (128/129) and on every arm (empty, control,
// printable, non-ASCII).
import XCTest
@testable import LMWarden

/// A SecretBackend that records every call and keeps items in memory, keyed
/// by "service|account". Never touches the real keychain.
private final class FakeBackend: SecretBackend {
    var items: [String: Data] = [:]
    var calls: [String] = []

    func read(service: String, account: String) throws -> Data? {
        calls.append("read")
        return items["\(service)|\(account)"]
    }

    func write(_ data: Data, service: String, account: String) throws {
        calls.append("write")
        items["\(service)|\(account)"] = data
    }

    func delete(service: String, account: String) throws {
        calls.append("delete")
        items["\(service)|\(account)"] = nil
    }
}

final class KeychainStoreTests: XCTestCase {
    func testRoundTrip() throws {
        let backend = FakeBackend()
        let store = KeychainStore(backend: backend)
        try store.set("s3cr3t", for: "hub-token")
        XCTAssertEqual(try store.string(for: "hub-token"), "s3cr3t")
        XCTAssertEqual(backend.items["\(KeychainStore.defaultService)|hub-token"], Data("s3cr3t".utf8))
    }

    func testMissingIsNil() throws {
        let store = KeychainStore(backend: FakeBackend())
        XCTAssertNil(try store.string(for: "never-set"))
    }

    func testEmptyValueRoundTrips() throws {
        let backend = FakeBackend()
        let store = KeychainStore(backend: backend)
        try store.set("", for: "empty-ok")
        XCTAssertEqual(try store.string(for: "empty-ok"), "")
        XCTAssertEqual(backend.items["\(KeychainStore.defaultService)|empty-ok"], Data())
    }

    func testRemoveDeletes() throws {
        let backend = FakeBackend()
        let store = KeychainStore(backend: backend)
        try store.set("value", for: "removable")
        XCTAssertNotNil(backend.items["\(KeychainStore.defaultService)|removable"])
        try store.remove(account: "removable")
        XCTAssertNil(backend.items["\(KeychainStore.defaultService)|removable"])
        XCTAssertEqual(try store.string(for: "removable"), nil)
    }

    func testServiceIsolation() throws {
        let backend = FakeBackend()
        let storeA = KeychainStore(backend: backend, service: "a")
        let storeB = KeychainStore(backend: backend, service: "b")
        try storeA.set("from-a", for: "shared-account")
        try storeB.set("from-b", for: "shared-account")
        XCTAssertEqual(try storeA.string(for: "shared-account"), "from-a")
        XCTAssertEqual(try storeB.string(for: "shared-account"), "from-b")
    }

    func testDefaultServiceMatchesBundleIdentifier() {
        XCTAssertEqual(KeychainStore.defaultService, AppVersion.bundleIdentifier)
    }

    // MARK: - bundle id -> service

    func testServiceForExactAppIdIsAppId() {
        XCTAssertEqual(KeychainStore.service(forBundleIdentifier: "com.podwarden.lmwarden"), "com.podwarden.lmwarden")
    }

    func testServiceForSmokeVariantIsVariant() {
        XCTAssertEqual(
            KeychainStore.service(forBundleIdentifier: "com.podwarden.lmwarden.smoke"),
            "com.podwarden.lmwarden.smoke"
        )
    }

    func testServiceForBarePrefixWithoutDotIsDefault() {
        XCTAssertEqual(KeychainStore.service(forBundleIdentifier: "com.podwarden.lmwardenx"), "com.podwarden.lmwarden")
    }

    func testServiceForNilIsDefault() {
        XCTAssertEqual(KeychainStore.service(forBundleIdentifier: nil), "com.podwarden.lmwarden")
    }

    func testServiceForForeignIdIsDefault() {
        XCTAssertEqual(KeychainStore.service(forBundleIdentifier: "com.apple.dt.xctest.tool"), "com.podwarden.lmwarden")
    }

    /// The prefix match is case-sensitive, like bundle ids.
    func testServiceForUppercasedAppIdIsDefault() {
        XCTAssertEqual(
            KeychainStore.service(forBundleIdentifier: "COM.PODWARDEN.LMWARDEN.smoke"),
            "com.podwarden.lmwarden"
        )
    }

    func testAccountEmptyRejected() {
        let store = KeychainStore(backend: FakeBackend())
        XCTAssertThrowsError(try store.set("x", for: "")) {
            XCTAssertEqual($0 as? KeychainError, .invalidAccount)
        }
        XCTAssertThrowsError(try store.string(for: "")) {
            XCTAssertEqual($0 as? KeychainError, .invalidAccount)
        }
        XCTAssertThrowsError(try store.remove(account: "")) {
            XCTAssertEqual($0 as? KeychainError, .invalidAccount)
        }
    }

    func testAccountAt128Allowed() throws {
        let store = KeychainStore(backend: FakeBackend())
        let account = String(repeating: "a", count: 128)
        XCTAssertNoThrow(try store.set("ok", for: account))
        XCTAssertEqual(try store.string(for: account), "ok")
    }

    func testAccountAt129Rejected() {
        let store = KeychainStore(backend: FakeBackend())
        let account = String(repeating: "a", count: 129)
        XCTAssertThrowsError(try store.set("x", for: account)) {
            XCTAssertEqual($0 as? KeychainError, .invalidAccount)
        }
    }

    func testAccountControlCharsRejected() {
        let store = KeychainStore(backend: FakeBackend())
        for account in ["a\nb", "a\tb", "a\u{0}b", "a\u{E0001}b"] {
            XCTAssertThrowsError(try store.set("x", for: account)) {
                XCTAssertEqual($0 as? KeychainError, .invalidAccount)
            }
        }
    }

    func testAccountNonASCIIAllowed() throws {
        let store = KeychainStore(backend: FakeBackend())
        let account = "ключ"
        XCTAssertNoThrow(try store.set("ok", for: account))
        XCTAssertEqual(try store.string(for: account), "ok")
    }

    func testRejectedAccountNeverReachesBackend() {
        let backend = FakeBackend()
        let store = KeychainStore(backend: backend)
        let badAccounts = ["", String(repeating: "a", count: 129), "a\nb", "a\tb", "a\u{0}b"]
        for account in badAccounts {
            XCTAssertThrowsError(try store.set("x", for: account))
            XCTAssertThrowsError(try store.string(for: account))
            XCTAssertThrowsError(try store.remove(account: account))
        }
        XCTAssertEqual(backend.calls, [])
    }

    func testUndecodableDataThrows() {
        let backend = FakeBackend()
        backend.items["\(KeychainStore.defaultService)|garbled"] = Data([0xFF, 0xFE])
        let store = KeychainStore(backend: backend)
        XCTAssertThrowsError(try store.string(for: "garbled")) {
            XCTAssertEqual($0 as? KeychainError, .undecodable)
        }
    }

    func testCheckSuccess() throws {
        XCTAssertEqual(try SystemKeychainBackend.check(errSecSuccess, notFoundIsOK: false), true)
    }

    func testCheckNotFoundOK() throws {
        XCTAssertEqual(try SystemKeychainBackend.check(errSecItemNotFound, notFoundIsOK: true), false)
    }

    func testCheckNotFoundNotOKThrows() {
        XCTAssertThrowsError(try SystemKeychainBackend.check(errSecItemNotFound, notFoundIsOK: false)) {
            XCTAssertEqual($0 as? KeychainError, .status(errSecItemNotFound))
        }
    }

    func testCheckOtherStatusThrows() {
        XCTAssertThrowsError(try SystemKeychainBackend.check(errSecAuthFailed, notFoundIsOK: true)) {
            XCTAssertEqual($0 as? KeychainError, .status(errSecAuthFailed))
        }
    }
}
