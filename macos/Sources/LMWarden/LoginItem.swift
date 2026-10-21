// LoginItem — "Open at Login" for the menu-bar app, driven through an
// injectable LoginItemBackend.
//
// Production uses `SystemLoginItemBackend`, which wraps
// `SMAppService.mainApp`. Tests inject a recording fake so they never touch
// the real login item (SMAppService needs a signed app bundle). `LoginItem.set`
// registers or unregisters only when the requested state differs from the
// backend's current state. A thrown backend error never crashes: `set`
// returns the backend's state after the failure and keeps the error in
// `lastError` so the shell can show it. A registration the user still has to
// approve in System Settings > Login Items reads as `needsApproval`, not as on.
import Foundation
import ServiceManagement

/// The login-item states the shell cares about, mirroring
/// `SMAppService.Status`.
enum LoginItemStatus: Equatable {
    case enabled
    case notRegistered
    case requiresApproval
    case notFound
}

/// Why the system backend refused to act.
enum LoginItemError: Error, Equatable {
    /// The process is not running from a `.app` bundle (`swift run`, tests):
    /// registering would add the bare executable as a login item.
    case notInAppBundle
}

/// The seam between LoginItem and the login-item service. Production uses
/// `SystemLoginItemBackend`; tests use a recording fake.
protocol LoginItemBackend: AnyObject {
    var status: LoginItemStatus { get }
    func register() throws
    func unregister() throws
    func openSystemSettingsLoginItems()
}

extension LoginItemBackend {
    var isEnabled: Bool { status == .enabled }
}

/// The production `LoginItemBackend`: `SMAppService.mainApp`. Outside a
/// `.app` bundle it reports `.notFound` and refuses to register or
/// unregister without calling SMAppService.
final class SystemLoginItemBackend: LoginItemBackend {
    private let isAppBundle: Bool

    init(bundle: Bundle = .main) {
        isAppBundle = bundle.bundleURL.pathExtension == "app"
    }

    var status: LoginItemStatus {
        guard isAppBundle else { return .notFound }
        switch SMAppService.mainApp.status {
        case .enabled: return .enabled
        case .requiresApproval: return .requiresApproval
        case .notFound: return .notFound
        case .notRegistered: return .notRegistered
        @unknown default: return .notRegistered
        }
    }

    func register() throws {
        guard isAppBundle else { throw LoginItemError.notInAppBundle }
        try SMAppService.mainApp.register()
    }

    func unregister() throws {
        guard isAppBundle else { throw LoginItemError.notInAppBundle }
        try SMAppService.mainApp.unregister()
    }

    func openSystemSettingsLoginItems() {
        SMAppService.openSystemSettingsLoginItems()
    }
}

/// Wraps the login-item backend. The backend's real status always wins on
/// read; `set` calls the backend only when the requested state differs.
final class LoginItem {
    private let backend: LoginItemBackend

    /// The error from the last `set` that reached the backend and failed;
    /// nil after any `set` that did not fail.
    private(set) var lastError: Error?

    init(backend: LoginItemBackend) {
        self.backend = backend
    }

    var status: LoginItemStatus {
        backend.status
    }

    var isEnabled: Bool {
        backend.status == .enabled
    }

    /// Registered, but the user must still approve it in System Settings >
    /// Login Items (`openSystemSettings()`).
    var needsApproval: Bool {
        backend.status == .requiresApproval
    }

    /// Register or unregister so the backend matches `on`. A no-op (no backend
    /// call) when the state already matches; a registration pending approval
    /// counts as registered (turning it off unregisters it, turning it on
    /// again makes no call). A thrown backend error is kept in `lastError`;
    /// the resulting `isEnabled` is returned either way.
    @discardableResult
    func set(_ on: Bool) -> Bool {
        lastError = nil
        let current = backend.status
        let registered = current == .enabled || current == .requiresApproval
        guard on != registered else {
            return isEnabled
        }
        do {
            if on {
                try backend.register()
            } else {
                try backend.unregister()
            }
        } catch {
            lastError = error
        }
        return isEnabled
    }

    /// Open System Settings > Login Items so the user can approve the item.
    func openSystemSettings() {
        backend.openSystemSettingsLoginItems()
    }
}
