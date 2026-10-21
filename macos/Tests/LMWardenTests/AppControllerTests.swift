// AppControllerTests — the non-UI composition root (Metal design s2/s4).
//
// AppController bundles the injected dependencies, then on boot() resolves the
// app paths, prepares the directories, loads/creates the cookie secret, picks
// a free port, builds the launch spec, and starts a ServerSupervisor —
// returning the UI URL. Every test here is fake-driven: an in-memory
// SecretBackend, a PortSelector(isAvailable:) closure, a recording prepare
// closure, and a FakeLauncher-backed supervisor. Nothing touches the real
// keychain, filesystem, socket, or process.
import XCTest
@testable import LMWarden

/// A SecretBackend that records every write and keeps items in memory, keyed
/// by "service|account" — the CookieFakeBackend idiom from CookieSecretTests.
/// The read result and any read/write error are injectable.
private final class AppSecretBackend: SecretBackend {
    var items: [String: Data] = [:]
    var writes: [(service: String, account: String, data: Data)] = []
    var readCount = 0
    var writeCount = 0

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
        writeCount += 1
        if let error = writeError() {
            throw error
        }
        writes.append((service, account, data))
        items["\(service)|\(account)"] = data
    }

    func delete(service: String, account: String) throws {
        items["\(service)|\(account)"] = nil
    }
}

/// A `LaunchedProcess` the test can inspect: a fixed pid and recorded signals.
private final class AppFakeProcess: LaunchedProcess {
    let pid: Int32 = 1000
    private(set) var terminateCount = 0
    func terminate() { terminateCount += 1 }
    func kill() {}
}

/// A `ProcessLaunching` that records every `ServerLaunchSpec` it is handed and
/// returns a fresh `AppFakeProcess` per launch.
private final class AppFakeLauncher: ProcessLaunching {
    private(set) var specs: [ServerLaunchSpec] = []
    private(set) var processes: [AppFakeProcess] = []

    var launchCount: Int { processes.count }

    func launch(
        _ spec: ServerLaunchSpec,
        onOutput: @escaping (Data) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws -> LaunchedProcess {
        specs.append(spec)
        let process = AppFakeProcess()
        processes.append(process)
        return process
    }
}

/// A `ServerSupervisor` the composition root can return from `makeSupervisor`.
/// A fixed, no-op schedule closure so `start()` schedules a health poll without
/// any real timer, and the default `Date.init` clock (the tests never fire a
/// poll, so no network or clock is touched).
private func makeAppSupervisor(spec: ServerLaunchSpec, launcher: AppFakeLauncher) -> ServerSupervisor {
    ServerSupervisor(
        spec: spec,
        launcher: launcher,
        probe: { _ in },
        schedule: { _, _ in },
        now: Date.init
    )
}

final class AppControllerTests: XCTestCase {
    private let dataDir = "/tmp/warden-app/data"
    private let modelsDir = "/tmp/warden-app/data/models"
    private let pythonPath = "/usr/bin/python3"
    private let parentPID: Int32 = 4242
    private let port: Int = 8091

    /// An `AppController` the test drives with an injected `makeShellToken`
    /// closure (so a test can pin the minted token and count calls), and a
    /// shared `AppRecording` for the launched spec.
    private func makeController(
        appPaths: AppPaths,
        secret: String,
        freePort: Int?,
        recording: AppRecording,
        makeShellToken: @escaping () throws -> String,
        backend: AppSecretBackend? = nil
    ) -> AppController {
        AppController(
            environment: makeEnvironment(
                appPaths: appPaths, secret: secret, freePort: freePort,
                recording: recording, backend: backend
            ),
            makeShellToken: makeShellToken
        )
    }

    /// An `AppEnvironment` whose `prepare` and `makeSupervisor` record into the
    /// caller's `recording` holder. The cookie secret is pre-seeded in the
    /// backend so `loadOrCreate` reads it and `created` is false.
    private func makeEnvironment(
        appPaths: AppPaths,
        secret: String,
        freePort: Int?,
        baseEnvironment: [String: String] = [:],
        prepareError: Error? = nil,
        recording: AppRecording,
        backend: AppSecretBackend? = nil
    ) -> AppEnvironment {
        let storeBackend = backend ?? AppSecretBackend(read: { Data(secret.utf8) })
        let ports = PortSelector(isAvailable: { candidate in
            freePort == nil ? false : candidate == freePort
        })
        let prepare: (AppPaths) throws -> Void = { paths in
            recording.prepareCalls.append(paths)
            if let prepareError {
                throw prepareError
            }
        }
        return AppEnvironment(
            pythonPath: pythonPath,
            applicationSupport: "/tmp/warden-app/support",
            parentPID: parentPID,
            baseEnvironment: baseEnvironment,
            store: KeychainStore(backend: storeBackend),
            ports: ports,
            prepare: prepare,
            makeSupervisor: { spec in
                recording.lastSpec = spec
                recording.supervisorCalls += 1
                return makeAppSupervisor(spec: spec, launcher: recording.launcher)
            }
        )
    }

    // (a) boot() returns the UI URL for the chosen port.
    func testBootReturnsUiUrl() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        let url = try controller.boot()

        XCTAssertEqual(url, URL(string: "http://127.0.0.1:\(port)/ui/"))
        XCTAssertNotNil(controller.supervisor)
        XCTAssertFalse(controller.secretCreated)
    }

    // P8: the environment's bundled UI export reaches the spec as
    // VW_WEBUI_DIR; without one the variable is absent.
    func testSpecCarriesTheBundledWebUIDir() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "c", count: 32)
        let recording = AppRecording()
        var environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)
        environment.webuiDir = "/Applications/LM Warden.app/Contents/Resources/webui"
        _ = try AppController(environment: environment).boot()
        XCTAssertEqual(
            recording.lastSpec?.environment["VW_WEBUI_DIR"],
            "/Applications/LM Warden.app/Contents/Resources/webui"
        )

        let bare = AppRecording()
        let plain = makeEnvironment(
            appPaths: appPaths, secret: secret, freePort: port,
            baseEnvironment: ["VW_WEBUI_DIR": "/inherited"], recording: bare
        )
        _ = try AppController(environment: plain).boot()
        XCTAssertNil(bare.lastSpec?.environment["VW_WEBUI_DIR"])
    }

    // (a) the spec handed to makeSupervisor carries the paths, secret, port and
    // the platform profile, and advertises the port on the command line.
    func testSpecCarriesPathsSecretPortAndProfile() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "b", count: 40)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        _ = try controller.boot()
        let spec = recording.lastSpec!

        XCTAssertEqual(spec.environment["VW_DATA_DIR"], appPaths.dataDir)
        XCTAssertEqual(spec.environment["VW_HF_CACHE_DIR"], appPaths.modelsDir)
        XCTAssertEqual(spec.environment["VW_COOKIE_SECRET"], secret)
        XCTAssertEqual(spec.environment["LMW_PLATFORM_PROFILE"], "apple")
        XCTAssertEqual(spec.environment["VW_PARENT_PID"], String(parentPID))
        let portIndex = spec.arguments.firstIndex(of: "--port")
        XCTAssertNotNil(portIndex)
        XCTAssertEqual(spec.arguments[portIndex! + 1], String(port))
    }

    // (a) first boot (empty store) generates and stores a secret.
    func testFirstBootCreatesSecret() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let backend = AppSecretBackend() // nothing stored
        let recording = AppRecording()
        let environment = makeEnvironment(
            appPaths: appPaths, secret: "", freePort: port, recording: recording, backend: backend
        )

        let controller = AppController(environment: environment)
        _ = try controller.boot()

        XCTAssertTrue(controller.secretCreated)
        XCTAssertEqual(backend.writeCount, 1)
        let written = backend.writes[0]
        XCTAssertEqual(written.account, CookieSecret.account)
        let decoded = String(data: written.data, encoding: .utf8)!
        XCTAssertEqual(decoded.count, 64)
        XCTAssertEqual(decoded.unicodeScalars.allSatisfy { "0123456789abcdef".unicodeScalars.contains($0) }, true)
    }

    // (a) a second controller over the same store reads the stored secret and
    // does not regenerate or rewrite it.
    func testSecondControllerReusesSecret() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let stored = String(repeating: "c", count: 64)
        let backend = AppSecretBackend(read: { Data(stored.utf8) })
        let recording = AppRecording()
        let environment = makeEnvironment(
            appPaths: appPaths, secret: stored, freePort: port, recording: recording, backend: backend
        )

        let controller = AppController(environment: environment)
        _ = try controller.boot()

        XCTAssertFalse(controller.secretCreated)
        XCTAssertEqual(backend.writeCount, 0)
        XCTAssertEqual(recording.lastSpec!.environment["VW_COOKIE_SECRET"], stored)
    }

    // (b) no free port: boot() throws .noFreePort and neither a supervisor is
    // created nor anything launched.
    func testNoFreePortThrowsAndLaunchesNothing() {
        let appPaths = try! AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let backend = AppSecretBackend(read: { Data(secret.utf8) })
        let recording = AppRecording()
        let environment = makeEnvironment(
            appPaths: appPaths, secret: secret, freePort: nil, recording: recording, backend: backend
        )

        let controller = AppController(environment: environment)
        XCTAssertThrowsError(try controller.boot()) { error in
            XCTAssertEqual(error as? AppControllerError, AppControllerError.noFreePort)
        }
        // The secret step ran before the port step (paths -> secret -> port).
        XCTAssertEqual(backend.readCount, 1)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.supervisorCalls, 0)
        XCTAssertEqual(recording.launcher.launchCount, 0)
    }

    // (b) prepare() failing stops boot() before the secret is loaded or
    // written: prepare threw, secret was never read, nothing launched.
    func testPrepareFailureStopsBeforeSecret() {
        let appPaths = try! AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let backend = AppSecretBackend(read: { Data(secret.utf8) })
        let recording = AppRecording()
        let environment = makeEnvironment(
            appPaths: appPaths,
            secret: secret,
            freePort: port,
            prepareError: AppPathsError.createFailed(appPaths.dataDir),
            recording: recording,
            backend: backend
        )

        let controller = AppController(environment: environment)
        XCTAssertThrowsError(try controller.boot()) { error in
            XCTAssertEqual(error as? AppPathsError, AppPathsError.createFailed(appPaths.dataDir))
        }
        XCTAssertEqual(recording.prepareCalls.count, 1)
        XCTAssertEqual(backend.readCount, 0)
        XCTAssertFalse(controller.secretCreated)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.launcher.launchCount, 0)
    }

    // (b) a second boot() while a supervisor exists returns the same URL
    // without relaunching.
    func testBootTwiceDoesNotRelaunch() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        let first = try controller.boot()
        let second = try controller.boot()

        XCTAssertEqual(first, second)
        XCTAssertEqual(recording.supervisorCalls, 1)
        XCTAssertEqual(recording.launcher.launchCount, 1)
    }

    // stop() terminates the booted child once, forgets the supervisor, and is
    // idempotent; a stop() before any boot does nothing; a boot() after stop()
    // launches afresh.
    func testStopIsIdempotentAndAllowsReboot() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        controller.stop()
        XCTAssertNil(controller.supervisor)

        _ = try controller.boot()
        controller.stop()
        controller.stop()

        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.launcher.processes.map(\.terminateCount), [1])

        _ = try controller.boot()
        XCTAssertNotNil(controller.supervisor)
        XCTAssertEqual(recording.supervisorCalls, 2)
        XCTAssertEqual(recording.launcher.launchCount, 2)
    }

    // (b) shareOnLAN true: the spec advertises --host 0.0.0.0.
    func testBootOptionsShareOnLANPassesAllInterfacesHost() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        let url = try controller.boot(options: BootOptions(shareOnLAN: true))

        let hostIndex = recording.lastSpec!.arguments.firstIndex(of: "--host")
        XCTAssertEqual(recording.lastSpec!.arguments[hostIndex! + 1], "0.0.0.0")
        // The UI URL stays loopback: the WKWebView is on this Mac.
        XCTAssertEqual(url, URL(string: "http://127.0.0.1:\(port)/ui/"))
    }

    // (b) default options: the spec keeps --host 127.0.0.1.
    func testBootDefaultOptionsKeepLoopbackHost() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        _ = try controller.boot(options: BootOptions())

        let hostIndex = recording.lastSpec!.arguments.firstIndex(of: "--host")
        XCTAssertEqual(recording.lastSpec!.arguments[hostIndex! + 1], "127.0.0.1")

        // The bare boot() (the headless path) is the default-options path too:
        // it must bind loopback only, never all interfaces. Its own recording,
        // so this leg reads the spec the bare boot() built.
        let recording2 = AppRecording()
        let controller2 = AppController(environment: makeEnvironment(
            appPaths: appPaths, secret: secret, freePort: port,
            recording: recording2, backend: AppSecretBackend(read: { Data(secret.utf8) })
        ))
        _ = try controller2.boot()
        XCTAssertEqual(recording2.launcher.launchCount, 1)
        let args2 = recording2.lastSpec!.arguments
        let hostIndex2 = args2.firstIndex(of: "--host")
        XCTAssertEqual(args2[hostIndex2! + 1], "127.0.0.1")
        XCTAssertFalse(args2.contains("0.0.0.0"))
        XCTAssertEqual(recording2.lastSpec!.environment["VW_HF_CACHE_DIR"], appPaths.modelsDir)
    }

    // (b) a modelsDir override reaches the spec's VW_HF_CACHE_DIR, and the
    // recorded prepare call carries it as paths.modelsDir.
    func testBootModelsOverrideReachesHfCacheDir() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let override = "/tmp/warden-app/custom-models"
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        _ = try controller.boot(options: BootOptions(modelsDir: override))

        XCTAssertEqual(recording.lastSpec!.environment["VW_HF_CACHE_DIR"], override)
        XCTAssertEqual(recording.prepareCalls[0].modelsDir, override)
        // dataDir is unaffected by a models override.
        XCTAssertEqual(recording.lastSpec!.environment["VW_DATA_DIR"], appPaths.dataDir)
    }

    // (b) a relative modelsDir override throws .notAbsolute before anything
    // launches: no supervisor created, launcher never called.
    func testBootRelativeModelsOverrideThrowsAndCreatesNoSupervisor() {
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let appPaths = try! AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        XCTAssertThrowsError(try controller.boot(options: BootOptions(modelsDir: "models"))) { error in
            XCTAssertEqual(error as? AppPathsError, AppPathsError.notAbsolute("models"))
        }
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.supervisorCalls, 0)
        XCTAssertEqual(recording.launcher.launchCount, 0)
    }

    // (c) boot sets port; stop() clears it.
    func testBootSetsPortAndStopClearsIt() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        XCTAssertNil(controller.port)
        _ = try controller.boot()
        XCTAssertEqual(controller.port, port)
        controller.stop()
        XCTAssertNil(controller.port)
    }

    // (c) a re-boot while running returns the same URL even with different
    // options; no relaunch happens.
    func testRebootWhileRunningIgnoresNewOptions() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let environment = makeEnvironment(appPaths: appPaths, secret: secret, freePort: port, recording: recording)

        let controller = AppController(environment: environment)
        let first = try controller.boot(options: BootOptions())
        let second = try controller.boot(options: BootOptions(shareOnLAN: true, modelsDir: "/tmp/warden-app/other"))

        XCTAssertEqual(first, second)
        XCTAssertEqual(recording.supervisorCalls, 1)
        XCTAssertEqual(recording.launcher.launchCount, 1)
    }

    // (d) shellToken is nil before the first boot.
    func testShellTokenNilBeforeBoot() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        var genCalls = 0
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            genCalls += 1
            return String(repeating: "a", count: 64)
        }

        XCTAssertNil(controller.shellToken)
        XCTAssertEqual(genCalls, 0)
    }

    // (d) boot() injects the minted token into the spec's environment and
    // exposes it read-only: lastSpec.environment["VW_SHELL_TOKEN"] and
    // controller.shellToken equal the generator's value.
    func testBootInjectsShellTokenIntoSpecEnvironment() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let minted = String(repeating: "a", count: 64)
        var genCalls = 0
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            genCalls += 1
            return minted
        }

        _ = try controller.boot()

        XCTAssertEqual(genCalls, 1)
        XCTAssertEqual(controller.shellToken, minted)
        XCTAssertEqual(recording.lastSpec!.environment["VW_SHELL_TOKEN"], minted)
    }

    // (d) a second boot while a supervisor exists returns the cached URL
    // WITHOUT re-minting: the generator is still called once, the token
    // unchanged.
    func testSecondBootWhileRunningKeepsToken() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let minted = String(repeating: "a", count: 64)
        var genCalls = 0
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            genCalls += 1
            return minted
        }

        let first = try controller.boot()
        let second = try controller.boot()

        XCTAssertEqual(first, second)
        XCTAssertEqual(genCalls, 1)
        XCTAssertEqual(recording.supervisorCalls, 1)
        XCTAssertEqual(recording.launcher.launchCount, 1)
        XCTAssertEqual(controller.shellToken, minted)
        XCTAssertEqual(recording.lastSpec!.environment["VW_SHELL_TOKEN"], minted)
    }

    // (d) stop() clears the token (and the port) along with the supervisor.
    func testStopClearsShellToken() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let minted = String(repeating: "a", count: 64)
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            minted
        }

        _ = try controller.boot()
        XCTAssertEqual(controller.shellToken, minted)
        controller.stop()

        XCTAssertNil(controller.shellToken)
        XCTAssertNil(controller.port)
        XCTAssertNil(controller.supervisor)
    }

    // (d) a boot after stop() mints a NEW token: the generator's second value
    // (64×'b') reaches the second spec, and the controller exposes it.
    func testRebootAfterStopMintsNewToken() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let first = String(repeating: "a", count: 64)
        let second = String(repeating: "b", count: 64)
        var genCalls = 0
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            genCalls += 1
            return genCalls == 1 ? first : second
        }

        _ = try controller.boot()
        XCTAssertEqual(controller.shellToken, first)
        controller.stop()
        _ = try controller.boot()

        XCTAssertEqual(genCalls, 2)
        XCTAssertEqual(controller.shellToken, second)
        XCTAssertEqual(recording.lastSpec!.environment["VW_SHELL_TOKEN"], second)
        XCTAssertEqual(recording.launcher.launchCount, 2)
    }

    // (d) a throwing makeShellToken propagates before anything launches: boot
    // throws it, no supervisor, the token stays nil, and the launcher spawned
    // nothing.
    func testShellTokenGeneratorErrorPropagatesWithoutLaunch() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: port, recording: recording
        ) {
            throw ShellTokenError.generatedInvalid
        }

        XCTAssertThrowsError(try controller.boot()) { error in
            XCTAssertEqual(error as? ShellTokenError, ShellTokenError.generatedInvalid)
        }
        XCTAssertNil(controller.supervisor)
        XCTAssertNil(controller.shellToken)
        XCTAssertEqual(recording.supervisorCalls, 0)
        XCTAssertEqual(recording.launcher.launchCount, 0)
    }

    // (d) no free port: the port step fails before the token is minted, so the
    // generator is never called and boot throws .noFreePort.
    func testNoFreePortDoesNotMintToken() {
        let appPaths = try! AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        var genCalls = 0
        let controller = makeController(
            appPaths: appPaths, secret: secret, freePort: nil, recording: recording
        ) {
            genCalls += 1
            return String(repeating: "a", count: 64)
        }

        XCTAssertThrowsError(try controller.boot()) { error in
            XCTAssertEqual(error as? AppControllerError, AppControllerError.noFreePort)
        }
        XCTAssertEqual(genCalls, 0)
        XCTAssertNil(controller.shellToken)
        XCTAssertNil(controller.supervisor)
    }

    // (c) the DEFAULT minting closure (AppController(environment:), as
    // main.swift and ShellCoordinator use it) draws from ShellToken's secure
    // generator: each boot's token is 64 lowercase hex chars, passes the
    // shell_auth rule, reaches the spec, and a reboot after stop() gets a
    // different one. A constant or reused default would pass every injected-
    // closure test above; this leg pins the production path.
    func testDefaultGeneratorMintsFreshValidTokenPerBoot() throws {
        let appPaths = try AppPaths.resolve(applicationSupport: "/tmp/warden-app/support")
        let secret = String(repeating: "a", count: 32)
        let recording = AppRecording()
        let controller = AppController(
            environment: makeEnvironment(
                appPaths: appPaths, secret: secret, freePort: port, recording: recording
            )
        )
        let hex = CharacterSet(charactersIn: "0123456789abcdef")

        _ = try controller.boot()
        let first = try XCTUnwrap(controller.shellToken)
        XCTAssertEqual(first.count, 64)
        XCTAssertTrue(first.unicodeScalars.allSatisfy { hex.contains($0) })
        XCTAssertTrue(ShellToken.isValid(first))
        XCTAssertEqual(recording.lastSpec!.environment[ShellToken.environmentKey], first)

        controller.stop()
        _ = try controller.boot()
        let second = try XCTUnwrap(controller.shellToken)

        XCTAssertTrue(ShellToken.isValid(second))
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(recording.lastSpec!.environment[ShellToken.environmentKey], second)
    }
}

/// Records everything the injected seams do, so a test can assert that each
/// step ran (or did not) in the expected order. `launcher` is the single
/// shared `AppFakeLauncher` that `makeSupervisor` hands to the supervisor.
private final class AppRecording {
    let launcher = AppFakeLauncher()
    var prepareCalls: [AppPaths] = []
    var lastSpec: ServerLaunchSpec?
    var supervisorCalls = 0
}
