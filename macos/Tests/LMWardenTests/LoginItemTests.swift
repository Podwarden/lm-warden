// LoginItemTests — "Open at Login" through SMAppService.mainApp, driven
// through an injectable LoginItemBackend so tests never touch the real login
// item (which needs a signed app bundle).
//
// LoginItem.set(_:) registers or unregisters only when the requested state
// differs from the backend's current state. A thrown backend error never
// crashes: set returns the backend's state after the failure and keeps the
// error in lastError. A registration awaiting approval reads as needsApproval.
import XCTest
@testable import LMWarden

/// A LoginItemBackend that records every call, changes `status` only when a
/// call does not throw, and can be told to throw.
private final class FakeLoginItemBackend: LoginItemBackend {
    var registerCalls = 0
    var unregisterCalls = 0
    var openSettingsCalls = 0
    /// When non-nil, register throws it.
    var registerError: Error?
    /// When non-nil, unregister throws it.
    var unregisterError: Error?
    /// The status a successful register leaves behind (the system may hold
    /// the item for user approval instead of enabling it).
    var statusAfterRegister: LoginItemStatus = .enabled
    /// The backend's current status, changed by register/unregister.
    var status: LoginItemStatus

    /// The backend's current enabled state.
    var enabled: Bool {
        status == .enabled
    }

    init(enabled: Bool = false) {
        self.status = enabled ? .enabled : .notRegistered
    }

    init(status: LoginItemStatus) {
        self.status = status
    }

    func register() throws {
        registerCalls += 1
        if let error = registerError {
            throw error
        }
        status = statusAfterRegister
    }

    func unregister() throws {
        unregisterCalls += 1
        if let error = unregisterError {
            throw error
        }
        status = .notRegistered
    }

    func openSystemSettingsLoginItems() {
        openSettingsCalls += 1
    }
}

/// Errors thrown by the recording fake so a test can distinguish the two
/// failure arms.
private enum LoginItemTestsError: Error, Equatable {
    case registerFailed
    case unregisterFailed
}

final class LoginItemTests: XCTestCase {
    /// Off -> on: set(true) calls register exactly once and reports the
    /// resulting enabled state as true.
    func testSetOnFromOffRegistersOnce() {
        let backend = FakeLoginItemBackend(enabled: false)
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.isEnabled)
        XCTAssertTrue(item.set(true))
        XCTAssertEqual(backend.registerCalls, 1)
        XCTAssertEqual(backend.unregisterCalls, 0)
        XCTAssertTrue(item.isEnabled)
    }

    /// On -> on: set(true) while already enabled makes no backend call and
    /// still reports true.
    func testSetOnWhenAlreadyOnMakesNoCall() {
        let backend = FakeLoginItemBackend(enabled: true)
        let item = LoginItem(backend: backend)
        XCTAssertTrue(item.set(true))
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertEqual(backend.unregisterCalls, 0)
        XCTAssertTrue(item.isEnabled)
    }

    /// On -> off: set(false) calls unregister exactly once and reports the
    /// resulting disabled state as false.
    func testSetOffFromOnUnregistersOnce() {
        let backend = FakeLoginItemBackend(enabled: true)
        let item = LoginItem(backend: backend)
        XCTAssertTrue(item.isEnabled)
        XCTAssertFalse(item.set(false))
        XCTAssertEqual(backend.unregisterCalls, 1)
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertFalse(item.isEnabled)
    }

    /// Off -> off: set(false) while already disabled makes no backend call
    /// and still reports false.
    func testSetOffWhenAlreadyOffMakesNoCall() {
        let backend = FakeLoginItemBackend(enabled: false)
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(false))
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertEqual(backend.unregisterCalls, 0)
        XCTAssertFalse(item.isEnabled)
    }

    /// A register that throws is swallowed: set(true) returns the backend's
    /// (still off) state, does not crash, and the single failed call is
    /// recorded.
    func testRegisterThrowsReturnsFalseAndRecordsOneCall() {
        let backend = FakeLoginItemBackend(enabled: false)
        backend.registerError = LoginItemTestsError.registerFailed
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(true))
        XCTAssertEqual(backend.registerCalls, 1)
        XCTAssertEqual(backend.unregisterCalls, 0)
        XCTAssertFalse(backend.enabled)
        XCTAssertFalse(item.isEnabled)
    }

    /// An unregister that throws is swallowed: set(false) returns the
    /// backend's (still on) state and does not crash.
    func testUnregisterThrowsReturnsTrue() {
        let backend = FakeLoginItemBackend(enabled: true)
        backend.unregisterError = LoginItemTestsError.unregisterFailed
        let item = LoginItem(backend: backend)
        XCTAssertTrue(item.set(false))
        XCTAssertEqual(backend.unregisterCalls, 1)
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertTrue(backend.enabled)
        XCTAssertTrue(item.isEnabled)
    }

    /// A register error is surfaced in lastError, not dropped.
    func testRegisterErrorIsKeptInLastError() {
        let backend = FakeLoginItemBackend(enabled: false)
        backend.registerError = LoginItemTestsError.registerFailed
        let item = LoginItem(backend: backend)
        XCTAssertNil(item.lastError)
        item.set(true)
        XCTAssertEqual(item.lastError as? LoginItemTestsError, .registerFailed)
    }

    /// An unregister error is surfaced in lastError, not dropped.
    func testUnregisterErrorIsKeptInLastError() {
        let backend = FakeLoginItemBackend(enabled: true)
        backend.unregisterError = LoginItemTestsError.unregisterFailed
        let item = LoginItem(backend: backend)
        item.set(false)
        XCTAssertEqual(item.lastError as? LoginItemTestsError, .unregisterFailed)
    }

    /// The next set that does not fail clears a previous error.
    func testSuccessfulSetClearsLastError() {
        let backend = FakeLoginItemBackend(enabled: false)
        backend.registerError = LoginItemTestsError.registerFailed
        let item = LoginItem(backend: backend)
        item.set(true)
        XCTAssertNotNil(item.lastError)
        backend.registerError = nil
        XCTAssertTrue(item.set(true))
        XCTAssertNil(item.lastError)
        XCTAssertEqual(backend.registerCalls, 2)
    }

    /// A no-op set (state already matches) also clears a previous error.
    func testNoOpSetClearsLastError() {
        let backend = FakeLoginItemBackend(enabled: false)
        backend.registerError = LoginItemTestsError.registerFailed
        let item = LoginItem(backend: backend)
        item.set(true)
        XCTAssertNotNil(item.lastError)
        item.set(false)
        XCTAssertNil(item.lastError)
        XCTAssertEqual(backend.unregisterCalls, 0)
    }

    /// A register the system holds for approval reads as off with
    /// needsApproval set, not as on.
    func testRegisterThatNeedsApprovalReturnsFalseAndFlagsApproval() {
        let backend = FakeLoginItemBackend(enabled: false)
        backend.statusAfterRegister = .requiresApproval
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.needsApproval)
        XCTAssertFalse(item.set(true))
        XCTAssertEqual(backend.registerCalls, 1)
        XCTAssertTrue(item.needsApproval)
        XCTAssertEqual(item.status, .requiresApproval)
        XCTAssertFalse(item.isEnabled)
        XCTAssertNil(item.lastError)
    }

    /// On while already pending approval makes no second register call.
    func testSetOnWhilePendingApprovalMakesNoCall() {
        let backend = FakeLoginItemBackend(status: .requiresApproval)
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(true))
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertEqual(backend.unregisterCalls, 0)
        XCTAssertTrue(item.needsApproval)
    }

    /// Off while pending approval unregisters, so no approval stays pending.
    func testSetOffWhilePendingApprovalUnregisters() {
        let backend = FakeLoginItemBackend(status: .requiresApproval)
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(false))
        XCTAssertEqual(backend.unregisterCalls, 1)
        XCTAssertEqual(backend.registerCalls, 0)
        XCTAssertFalse(item.needsApproval)
        XCTAssertEqual(item.status, .notRegistered)
    }

    /// From notFound, on registers (notFound is not registered).
    func testSetOnFromNotFoundRegisters() {
        let backend = FakeLoginItemBackend(status: .notFound)
        let item = LoginItem(backend: backend)
        XCTAssertTrue(item.set(true))
        XCTAssertEqual(backend.registerCalls, 1)
    }

    /// From notFound, off makes no call.
    func testSetOffFromNotFoundMakesNoCall() {
        let backend = FakeLoginItemBackend(status: .notFound)
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(false))
        XCTAssertEqual(backend.unregisterCalls, 0)
    }

    /// The real status wins on read: a change made outside the app (the
    /// user turns it off in System Settings) shows without any set.
    func testRealStatusWinsOnRead() {
        let backend = FakeLoginItemBackend(enabled: true)
        let item = LoginItem(backend: backend)
        XCTAssertTrue(item.isEnabled)
        backend.status = .notRegistered
        XCTAssertFalse(item.isEnabled)
        backend.status = .requiresApproval
        XCTAssertTrue(item.needsApproval)
    }

    /// openSystemSettings goes to the backend once.
    func testOpenSystemSettingsCallsBackend() {
        let backend = FakeLoginItemBackend(enabled: false)
        let item = LoginItem(backend: backend)
        item.openSystemSettings()
        XCTAssertEqual(backend.openSettingsCalls, 1)
    }

    /// Outside a .app bundle (the .xctest bundle here) the system backend
    /// reports notFound and refuses to register or unregister, without ever
    /// reaching SMAppService.
    func testSystemBackendOutsideAppBundleRefuses() {
        let bundle = Bundle(for: LoginItemTests.self)
        XCTAssertNotEqual(bundle.bundleURL.pathExtension, "app")
        let backend = SystemLoginItemBackend(bundle: bundle)
        XCTAssertEqual(backend.status, .notFound)
        XCTAssertFalse(backend.isEnabled)
        XCTAssertThrowsError(try backend.register()) { error in
            XCTAssertEqual(error as? LoginItemError, .notInAppBundle)
        }
        XCTAssertThrowsError(try backend.unregister()) { error in
            XCTAssertEqual(error as? LoginItemError, .notInAppBundle)
        }
        let item = LoginItem(backend: backend)
        XCTAssertFalse(item.set(true))
        XCTAssertEqual(item.lastError as? LoginItemError, .notInAppBundle)
    }
}
