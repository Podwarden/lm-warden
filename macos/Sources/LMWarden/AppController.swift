// AppController — the non-UI composition root (Metal design s2/s4).
//
// It bundles the app's injectable dependencies (an `AppEnvironment`) and, on
// `boot()`, composes them in a fixed order: resolve the app paths, prepare the
// directories, load or create the cookie secret, pick a free port, build the
// `ServerLaunchSpec`, and start a `ServerSupervisor`. It returns the UI URL
// the WKWebView points at.
//
// The root is entirely non-UI: it owns no Foundation timers, actors, or async,
// and every dependency it needs is injected, so a test can drive the whole
// boot with fakes (in-memory keychain, closure port check, recording prepare,
// FakeLauncher-backed supervisor).
import Foundation

/// Failure to start the server from the composition root.
enum AppControllerError: Error, Equatable {
    /// `ports.choose()` returned nil: no candidate port was free.
    case noFreePort
}

/// Options for a single `boot(options:)` call.
struct BootOptions: Equatable {
    /// When true the server binds 0.0.0.0 (LAN sharing); when false (default)
    /// it binds 127.0.0.1.
    var shareOnLAN: Bool = false
    /// An absolute path to override the default models directory. A relative
    /// path throws `AppPathsError.notAbsolute` before anything launches.
    var modelsDir: String? = nil
}

/// The injectable dependencies the composition root needs to boot the server.
///
/// Each field is a seam a test can fake: a path string, the parent pid, the
/// base child environment, a `KeychainStore` (with an in-memory backend), a
/// `PortSelector` (with a closure availability check), and two closures —
/// `prepare` (create the directories) and `makeSupervisor` (build the
/// supervisor from a spec).
struct AppEnvironment {
    let pythonPath: String
    let applicationSupport: String
    let parentPID: Int32
    let baseEnvironment: [String: String]
    let store: KeychainStore
    let ports: PortSelector
    let prepare: (AppPaths) throws -> Void
    let makeSupervisor: (ServerLaunchSpec) -> ServerSupervisor
    /// The bundled UI export handed to the server as VW_WEBUI_DIR; nil when
    /// the bundle carries none (the server then serves no /ui).
    var webuiDir: String? = nil
}

/// The non-UI composition root. `boot()` assembles the server and returns the
/// UI URL; `supervisor` and `secretCreated` expose what the boot produced.
final class AppController {
    private let environment: AppEnvironment
    private(set) var supervisor: ServerSupervisor?
    /// True iff this boot generated (and stored) a new cookie secret.
    private(set) var secretCreated: Bool = false
    /// The URL a previous successful boot returned; a re-boot returns it as-is
    /// without relaunching.
    private var uiURL: URL?
    /// The port the server bound on the last successful boot; `nil` before the
    /// first boot and after `stop()`.
    private(set) var port: Int?
    /// The per-launch shell token minted on the last successful boot; `nil`
    /// before the first boot and after `stop()`. In-memory only: never
    /// persisted or logged.
    private(set) var shellToken: String?
    /// Mints the per-launch shell token. Injectable so a test can pin the value
    /// and count calls; the default is `ShellToken.generate()`.
    private let makeShellToken: () throws -> String

    init(
        environment: AppEnvironment,
        makeShellToken: @escaping () throws -> String = { try ShellToken.generate() }
    ) {
        self.environment = environment
        self.makeShellToken = makeShellToken
    }

    /// Boot the embedded server and return the UI URL.
    ///
    /// Steps, in order, each failure propagating and stopping before the next
    /// (no supervisor created, nothing launched):
    ///   1. `AppPaths.resolve(applicationSupport:modelsOverride:)`
    ///   2. `environment.prepare(paths)`
    ///   3. `CookieSecret.loadOrCreate(store:)`
    ///   4. `environment.ports.choose()` (nil → `.noFreePort`)
    ///   4a. `makeShellToken()` (a throw propagates; the token is never printed)
    ///   5. `ServerLaunchSpec.make(...)` with the host from `options.shareOnLAN`
    ///       and the minted `shellToken:`
    ///   6. `environment.makeSupervisor(spec)` + `start()`
    ///
    /// The returned URL is always `http://127.0.0.1:<port>/ui/` — the
    /// WKWebView is on this Mac even when the server binds 0.0.0.0.
    ///
    /// A second `boot(options:)` while a supervisor already exists returns the
    /// same URL without relaunching; the new options are ignored.
    func boot(options: BootOptions) throws -> URL {
        if supervisor != nil, let uiURL {
            return uiURL
        }

        let paths = try AppPaths.resolve(applicationSupport: environment.applicationSupport, modelsOverride: options.modelsDir)
        try environment.prepare(paths)

        let secretResult = try CookieSecret.loadOrCreate(store: environment.store)
        secretCreated = secretResult.created

        guard let port = environment.ports.choose() else {
            throw AppControllerError.noFreePort
        }

        let shellToken = try makeShellToken()

        let spec = try ServerLaunchSpec.make(
            pythonPath: environment.pythonPath,
            port: port,
            dataDir: paths.dataDir,
            parentPID: environment.parentPID,
            cookieSecret: secretResult.secret,
            host: options.shareOnLAN ? "0.0.0.0" : "127.0.0.1",
            hfCacheDir: paths.modelsDir,
            baseEnvironment: environment.baseEnvironment,
            shellToken: shellToken,
            webuiDir: environment.webuiDir
        )

        let newSupervisor = environment.makeSupervisor(spec)
        newSupervisor.start()
        let url = URL(string: "http://127.0.0.1:\(port)/ui/")!
        self.supervisor = newSupervisor
        self.uiURL = url
        self.port = port
        self.shellToken = shellToken
        return url
    }

    /// Boot with default options: loopback only, default models directory.
    func boot() throws -> URL {
        try boot(options: BootOptions())
    }

    /// Stop the supervisor a boot started, then forget it, so a later `boot()`
    /// launches afresh. Idempotent: a second `stop()`, or one before any
    /// successful boot, does nothing.
    func stop() {
        guard let supervisor else {
            return
        }
        supervisor.stop()
        self.supervisor = nil
        uiURL = nil
        port = nil
        shellToken = nil
    }
}
