// SleepGuard — hold one IOPM PreventUserIdleSystemSleep assertion while a
// model is loaded.
//
// While a model is loaded the app is doing real work and the user expects the
// machine to stay awake, so we hold a
// `kIOPMAssertPreventUserIdleSystemSleep` assertion. The guard is idempotent:
// `update(modelLoaded: true)` creates at most one assertion, a repeated call
// while held is a no-op, and `update(modelLoaded: false)` releases the held id
// exactly once (and is a no-op when nothing is held). `SystemPowerAssertionBackend`
// is the production backend (IOPM); tests inject a `PowerAssertionBackend`
// fake so they never touch the real power manager.
import Foundation
import IOKit.pwr_mgt

enum SleepGuardError: Error, Equatable {
    case createFailed(Int32)
}

/// The seam between SleepGuard and the power manager. Production uses
/// `SystemPowerAssertionBackend`; tests use a recording fake.
protocol PowerAssertionBackend: AnyObject {
    func create(reason: String) throws -> UInt32
    func release(_ id: UInt32)
}

/// The production `PowerAssertionBackend`: IOPM sleep assertions.
final class SystemPowerAssertionBackend: PowerAssertionBackend {
    func create(reason: String) throws -> UInt32 {
        var id: IOPMAssertionID = 0
        let status = IOPMAssertionCreateWithName(
            kIOPMAssertPreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            reason as CFString,
            &id
        )
        guard status == kIOReturnSuccess else {
            throw SleepGuardError.createFailed(status)
        }
        return UInt32(id)
    }

    func release(_ id: UInt32) {
        IOPMAssertionRelease(IOPMAssertionID(id))
    }
}

/// Holds at most one sleep assertion, created on first load and released on
/// unload (or on deinit if still held).
final class SleepGuard {
    static let defaultReason = "LM Warden is serving a loaded model"

    private let backend: PowerAssertionBackend
    private let reason: String
    private var heldID: UInt32?

    var isHeld: Bool {
        heldID != nil
    }

    init(backend: PowerAssertionBackend, reason: String = SleepGuard.defaultReason) {
        self.backend = backend
        self.reason = reason
    }

    /// Reflect the loaded-model state. `true` creates one assertion if none is
    /// held; `false` releases the held id if there is one. Idempotent in both
    /// directions. A create that throws leaves the guard not held and rethrows.
    func update(modelLoaded: Bool) throws {
        if modelLoaded {
            guard heldID == nil else { return }
            let id = try backend.create(reason: reason)
            heldID = id
        } else {
            if let id = heldID {
                backend.release(id)
                heldID = nil
            }
        }
    }

    /// Release the held assertion without throwing, then forget the id. A
    /// no-op when nothing is held. Called from `deinit` so a guard destroyed
    /// while still holding an assertion does not leak one on process exit.
    func releaseAll() {
        if let id = heldID {
            backend.release(id)
            heldID = nil
        }
    }

    deinit {
        releaseAll()
    }
}
