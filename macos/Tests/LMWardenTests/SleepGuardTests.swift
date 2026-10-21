// SleepGuardTests — hold one IOPM PreventUserIdleSystemSleep assertion while a
// model is loaded, driven through an injectable PowerAssertionBackend.
//
// The sg is idempotent: a second load while already held creates no second
// assertion, and an unload with nothing held is a no-op. Tests drive the sg
// through a recording FakeBackend so they never touch the real power manager;
// a single live test exercises the real SystemPowerAssertionBackend.
import XCTest
import IOKit.pwr_mgt
@testable import LMWarden

/// A PowerAssertionBackend that records every call and returns canned ids.
/// Never touches the real power manager.
private final class FakeBackend: PowerAssertionBackend {
    var nextId = 1
    var createReasons: [String] = []
    var released: [UInt32] = []
    /// When non-nil, create throws .createFailed(value).
    var failCreate: Int32?

    func create(reason: String) throws -> UInt32 {
        if let status = failCreate {
            throw SleepGuardError.createFailed(status)
        }
        createReasons.append(reason)
        let id = UInt32(nextId)
        nextId += 1
        return id
    }

    func release(_ id: UInt32) {
        released.append(id)
    }
}

final class SleepGuardTests: XCTestCase {
    /// A fresh load creates exactly one assertion.
    func testFirstLoadCreatesOne() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        XCTAssertFalse(sg.isHeld)
        try sg.update(modelLoaded: true)
        XCTAssertTrue(sg.isHeld)
        XCTAssertEqual(backend.createReasons.count, 1)
        XCTAssertEqual(backend.released, [])
    }

    /// A second load while already held creates no second assertion.
    func testRepeatedLoadCreatesNoSecond() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        try sg.update(modelLoaded: true)
        try sg.update(modelLoaded: true)
        try sg.update(modelLoaded: true)
        XCTAssertEqual(backend.createReasons.count, 1)
        XCTAssertTrue(sg.isHeld)
    }

    /// An unload releases the same id that was created.
    func testUnloadReleasesSameId() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        try sg.update(modelLoaded: true)
        let createdId = UInt32(backend.nextId - 1)
        try sg.update(modelLoaded: false)
        XCTAssertFalse(sg.isHeld)
        XCTAssertEqual(backend.released, [createdId])
    }

    /// An unload with nothing held does nothing (no create, no release).
    func testUnloadWhenNotHeldDoesNothing() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        try sg.update(modelLoaded: false)
        XCTAssertFalse(sg.isHeld)
        XCTAssertEqual(backend.createReasons.count, 0)
        XCTAssertEqual(backend.released, [])
    }

    /// A create that throws leaves the sg not held and rethrows.
    func testCreateFailureLeavesNotHeld() throws {
        let backend = FakeBackend()
        let status: Int32 = -536870120 // 0xE0000318 as Int32: a non-success IOReturn
        backend.failCreate = status
        let sg = SleepGuard(backend: backend)
        XCTAssertThrowsError(try sg.update(modelLoaded: true)) { error in
            XCTAssertEqual(error as? SleepGuardError, .createFailed(status))
        }
        XCTAssertFalse(sg.isHeld)
        // No id was ever handed out, so a later unload must not release anything.
        try sg.update(modelLoaded: false)
        XCTAssertEqual(backend.released, [])
    }

    /// Releasing everything is equivalent to an unload without throwing.
    func testReleaseAllReleasesHeld() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        try sg.update(modelLoaded: true)
        let createdId = UInt32(backend.nextId - 1)
        sg.releaseAll()
        XCTAssertFalse(sg.isHeld)
        XCTAssertEqual(backend.released, [createdId])
    }

    /// Releasing everything when nothing is held is a no-op.
    func testReleaseAllWhenNotHeldDoesNothing() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        sg.releaseAll()
        XCTAssertFalse(sg.isHeld)
        XCTAssertEqual(backend.createReasons.count, 0)
        XCTAssertEqual(backend.released, [])
    }

    /// The reason is passed through to the backend verbatim.
    func testReasonPassedThrough() throws {
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend, reason: "custom reason")
        try sg.update(modelLoaded: true)
        XCTAssertEqual(backend.createReasons, ["custom reason"])
    }

    /// The default reason is the documented constant.
    func testDefaultReason() {
        XCTAssertEqual(SleepGuard.defaultReason, "LM Warden is serving a loaded model")
        let backend = FakeBackend()
        let sg = SleepGuard(backend: backend)
        try? sg.update(modelLoaded: true)
        XCTAssertEqual(backend.createReasons, [SleepGuard.defaultReason])
    }

    /// When the sg deallocates while still holding an assertion, deinit
    /// releases it so the process does not leak a sleep assertion on exit.
    func testDeinitReleases() {
        let backend = FakeBackend()
        autoreleasepool {
            let sg = SleepGuard(backend: backend)
            try? sg.update(modelLoaded: true)
            XCTAssertTrue(sg.isHeld)
        }
        XCTAssertEqual(backend.createReasons.count, 1)
        XCTAssertEqual(backend.released.count, 1)
    }

    /// The real SystemPowerAssertionBackend creates a non-zero id whose
    /// assertion the power manager reports with the PreventUserIdleSystemSleep
    /// type and our reason as its name, and release makes it disappear.
    /// Skipped off macOS so the suite stays portable.
    func testSystemBackendCreateAndRelease() throws {
        #if os(macOS)
        let backend = SystemPowerAssertionBackend()
        let id = try backend.create(reason: SleepGuard.defaultReason)
        XCTAssertNotEqual(id, 0)
        let props = IOPMAssertionCopyProperties(IOPMAssertionID(id))?
            .takeRetainedValue() as? [String: Any]
        XCTAssertEqual(props?[kIOPMAssertionTypeKey] as? String,
                       kIOPMAssertPreventUserIdleSystemSleep)
        XCTAssertEqual(props?[kIOPMAssertionNameKey] as? String, SleepGuard.defaultReason)
        backend.release(id)
        XCTAssertNil(IOPMAssertionCopyProperties(IOPMAssertionID(id)),
                     "the assertion must be gone after release")
        #else
        throw XCTSkip("only meaningful on macOS")
        #endif
    }
}
