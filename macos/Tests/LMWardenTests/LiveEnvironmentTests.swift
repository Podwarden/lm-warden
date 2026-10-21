import XCTest
@testable import LMWarden

final class LiveEnvironmentTests: XCTestCase {
    /// live() must read parentPID and baseEnvironment from the injected
    /// ProcessInfo and pass pythonPath/applicationSupport through.
    func testLiveFieldsFromProcessInfo() {
        let processInfo = ProcessInfo()
        let env = AppEnvironment.live(
            pythonPath: "/opt/python",
            applicationSupport: "/tmp/app-support",
            processInfo: processInfo
        )
        XCTAssertEqual(env.pythonPath, "/opt/python")
        XCTAssertEqual(env.applicationSupport, "/tmp/app-support")
        XCTAssertEqual(env.parentPID, processInfo.processIdentifier)
        // Compared without XCTAssertEqual: a failure must not print the
        // process environment, which may hold tokens.
        XCTAssertTrue(
            env.baseEnvironment == processInfo.environment,
            "baseEnvironment must be the process environment (contents not printed)"
        )
    }

    /// The python path and application support string are passed through to
    /// the environment unchanged.
    func testLivePassesPaths() {
        let env = AppEnvironment.live(
            pythonPath: "/usr/bin/python3",
            applicationSupport: "/Users/test/Library/Application Support"
        )
        XCTAssertEqual(env.pythonPath, "/usr/bin/python3")
        XCTAssertEqual(env.applicationSupport, "/Users/test/Library/Application Support")
    }

    /// port(of:) returns the integer after the first "--port" argument.
    func testPortOfSpecReadsFlag() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/usr/bin/python3",
            port: 18080,
            dataDir: "/tmp/data",
            parentPID: 42,
            cookieSecret: String(repeating: "a", count: 32)
        )
        XCTAssertEqual(AppEnvironment.port(of: spec), 18080)
    }

    /// port(of:) returns nil when "--port" is absent from the arguments.
    func testPortOfSpecMissingFlagIsNil() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/usr/bin/python3",
            port: 8080,
            dataDir: "/tmp/data",
            parentPID: 42,
            cookieSecret: String(repeating: "a", count: 32)
        )
        let stripped = ServerLaunchSpec(
            executable: spec.executable,
            arguments: spec.arguments.filter { $0 != "--port" && $0 != "8080" },
            environment: spec.environment
        )
        XCTAssertNil(AppEnvironment.port(of: stripped))
    }

    /// port(of:) returns nil when "--port" is the last argument with no value
    /// following it.
    func testPortOfSpecTrailingFlagIsNil() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/usr/bin/python3"),
            arguments: ["-I", "-m", "uvicorn", "app.main:app", "--port"],
            environment: [:]
        )
        XCTAssertNil(AppEnvironment.port(of: spec))
    }

    /// port(of:) returns nil when the value after "--port" is not an integer.
    func testPortOfSpecNonIntegerIsNil() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/usr/bin/python3"),
            arguments: ["-I", "-m", "uvicorn", "app.main:app", "--port", "not-a-port"],
            environment: [:]
        )
        XCTAssertNil(AppEnvironment.port(of: spec))
    }

    /// livePrepare creates the data and models directories and drops the
    /// .metadata_never_index marker inside modelsDir.
    func testLivePrepareCreatesDirsAndMarker() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try AppEnvironment.livePrepare(paths)

        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.dataDir))
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.modelsDir))
        let marker = paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker), "expected marker at \(marker)")
    }

    /// makeSupervisor wires the injected schedule into the supervisor: after
    /// start() the first scheduled health poll reaches the injected recording
    /// closure with the default poll interval.
    func testMakeSupervisorUsesInjectedSchedule() throws {
        var scheduledDelays: [TimeInterval] = []
        let scheduleLock = NSLock()
        let schedule: (TimeInterval, @escaping () -> Void) -> Void = { delay, _ in
            scheduleLock.lock()
            scheduledDelays.append(delay)
            scheduleLock.unlock()
        }

        let env = AppEnvironment.live(
            pythonPath: "/usr/bin/false",
            applicationSupport: FileManager.default.temporaryDirectory.path,
            schedule: schedule
        )

        let spec = try ServerLaunchSpec.make(
            pythonPath: "/usr/bin/false",
            port: 18080,
            dataDir: FileManager.default.temporaryDirectory.path,
            parentPID: 42,
            cookieSecret: String(repeating: "a", count: 32)
        )
        let supervisor = env.makeSupervisor(spec)
        supervisor.start()

        // /usr/bin/false exits immediately; give the first scheduled poll a
        // moment to fire (default healthPollInterval is 0.5 s).
        let deadline = Date(timeIntervalSinceNow: 3)
        scheduleLock.lock()
        var hasDelay = !scheduledDelays.isEmpty
        scheduleLock.unlock()
        while !hasDelay, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
            scheduleLock.lock()
            hasDelay = !scheduledDelays.isEmpty
            scheduleLock.unlock()
        }

        scheduleLock.lock()
        let firstDelay = scheduledDelays.first
        scheduleLock.unlock()
        supervisor.stop()

        XCTAssertNotNil(firstDelay, "expected at least one scheduled delay to be recorded")
        // The first scheduled call must be the initial health poll (0.5 s
        // default), not the stop-grace kill (15 s) or any backoff: this proves
        // the injected schedule closure is the one the supervisor uses.
        guard let firstDelay else { return }
        XCTAssertEqual(firstDelay, 0.5, accuracy: 0.001)
    }

    /// port(of:) reads the FIRST "--port": a later duplicate flag is ignored.
    func testPortOfSpecFirstFlagWins() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/usr/bin/python3"),
            arguments: ["-m", "uvicorn", "--port", "18080", "--port", "9"],
            environment: [:]
        )
        XCTAssertEqual(AppEnvironment.port(of: spec), 18080)
    }

    /// live() wires livePrepare as `prepare`: the environment's own prepare
    /// creates both dirs and the models-dir marker.
    func testLivePrepareIsTheEnvironmentPrepare() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let env = AppEnvironment.live(pythonPath: "/usr/bin/false", applicationSupport: tempDir.path)
        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try env.prepare(paths)

        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.dataDir))
        let marker = paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker), "expected marker at \(marker)")
    }

    /// A marker failure is best-effort: with a DIRECTORY already sitting at
    /// the marker path (so the touch fails), livePrepare still succeeds.
    func testLivePrepareMarkerFailureDoesNotThrow() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        let markerPath = paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName
        try FileManager.default.createDirectory(atPath: markerPath, withIntermediateDirectories: true)
        XCTAssertThrowsError(try ModelDirMarkers.apply(modelsDir: paths.modelsDir), "precondition: the marker must fail")

        XCTAssertNoThrow(try AppEnvironment.livePrepare(paths))
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.dataDir))
    }

    /// An AppPaths.prepare failure propagates: a regular file where the data
    /// dir should be makes livePrepare throw .createFailed(dataDir).
    func testLivePrepareDirFailurePropagates() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        XCTAssertTrue(FileManager.default.createFile(atPath: paths.dataDir, contents: Data()))

        XCTAssertThrowsError(try AppEnvironment.livePrepare(paths)) { error in
            XCTAssertEqual(error as? AppPathsError, .createFailed(paths.dataDir))
        }
    }

    /// excludeFromBackup: true touches the marker then excludes the dir, in
    /// that order, each called once with the exact paths.
    func testLivePrepareExcludeTrueTouchesThenExcludes() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        var touched: [String] = []
        var excluded: [String] = []
        var flags: [Bool] = []
        var order: [String] = []

        try AppEnvironment.livePrepare(
            paths,
            excludeFromBackup: true,
            touch: { path in
                order.append("touch")
                touched.append(path)
            },
            exclude: { path, flag in
                order.append("exclude")
                excluded.append(path)
                flags.append(flag)
            }
        )

        XCTAssertEqual(touched, [paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName])
        XCTAssertEqual(excluded, [paths.modelsDir])
        XCTAssertEqual(flags, [true])
        XCTAssertEqual(order, ["touch", "exclude"], "touch must precede exclude")
    }

    /// excludeFromBackup: false still writes the Spotlight marker and calls
    /// the exclude closure once with false for the models dir (clearing any
    /// earlier exclusion), so a folder excluded on a prior launch is included
    /// again.
    func testLivePrepareExcludeFalseClearsExclusion() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        var touched: [String] = []
        var excluded: [String] = []
        var flags: [Bool] = []
        var order: [String] = []

        try AppEnvironment.livePrepare(
            paths,
            excludeFromBackup: false,
            touch: { path in
                order.append("touch")
                touched.append(path)
            },
            exclude: { path, flag in
                order.append("exclude")
                excluded.append(path)
                flags.append(flag)
            }
        )

        XCTAssertEqual(touched, [paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName])
        XCTAssertEqual(excluded, [paths.modelsDir])
        XCTAssertEqual(flags, [false])
        XCTAssertEqual(order, ["touch", "exclude"], "touch must precede exclude")
    }

    /// A marker failure is still best-effort with the flag false: a throwing
    /// touch never fails prepare and the data dir is still created.
    func testLivePrepareExcludeFalseTouchFailureDoesNotThrow() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        struct MarkerError: Error {}
        var excluded: [String] = []

        XCTAssertNoThrow(
            try AppEnvironment.livePrepare(
                paths,
                excludeFromBackup: false,
                touch: { _ in throw MarkerError() },
                exclude: { path, _ in excluded.append(path) }
            )
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.dataDir))
        XCTAssertTrue(excluded.isEmpty, "exclude must not run after a touch failure")
    }

    /// live(excludeModelsFromBackup: false) wires the flag into prepare: the
    /// real marker is still written but the dir's backup flag stays unset.
    func testLiveExcludeFalseLeavesBackupFlagUnset() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let env = AppEnvironment.live(
            pythonPath: "/usr/bin/false",
            applicationSupport: tempDir.path,
            excludeModelsFromBackup: false
        )
        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try env.prepare(paths)

        let marker = paths.modelsDir + "/" + ModelDirMarkers.neverIndexFileName
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker), "marker must be written even with the flag off")
        let values = try URL(fileURLWithPath: paths.modelsDir).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, false, "models dir must not be excluded from backup")
    }

    /// A models dir excluded from Time Machine on an earlier launch is
    /// included again when live(excludeModelsFromBackup: false) prepares it:
    /// the real system closure clears the key.
    func testLiveExcludeFalseClearsEarlierExclusion() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try AppEnvironment.livePrepare(paths, excludeFromBackup: true)
        let before = try URL(fileURLWithPath: paths.modelsDir).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(before.isExcludedFromBackup, true, "precondition: the earlier launch excluded the dir")

        let env = AppEnvironment.live(
            pythonPath: "/usr/bin/false",
            applicationSupport: tempDir.path,
            excludeModelsFromBackup: false
        )
        try env.prepare(paths)

        let after = try URL(fileURLWithPath: paths.modelsDir).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(after.isExcludedFromBackup, false, "unticking must clear an earlier exclusion")
    }

    /// live() default keeps today's behaviour: the flag is on, so the dir's
    /// backup exclusion is set.
    func testLiveDefaultSetsBackupFlag() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let env = AppEnvironment.live(
            pythonPath: "/usr/bin/false",
            applicationSupport: tempDir.path
        )
        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try env.prepare(paths)

        let values = try URL(fileURLWithPath: paths.modelsDir).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, true, "models dir must be excluded from backup by default")
    }

    /// The one-argument livePrepare wrapper keeps today's behaviour: the
    /// models dir's backup exclusion is set.
    func testLivePrepareWrapperSetsBackupFlag() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveEnvironmentTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let paths = try AppPaths.resolve(applicationSupport: tempDir.path)
        try AppEnvironment.livePrepare(paths)

        let values = try URL(fileURLWithPath: paths.modelsDir).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, true, "the one-argument livePrepare must exclude the models dir from backup")
    }

    /// The supervisor's HealthProbe targets the spec's "--port": a loopback
    /// listener answering {"ok":true} on that port moves the supervisor to
    /// .running when the recorded first poll is run by hand.
    func testMakeSupervisorProbesTheSpecPort() throws {
        let listener = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(listener, 0, "socket() should succeed")
        defer { close(listener) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(listener, $0, length) }
        }
        XCTAssertEqual(bound, 0, "bind() should succeed")
        XCTAssertEqual(listen(listener, 4), 0, "listen() should succeed")
        _ = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
        }
        let port = Int(UInt16(bigEndian: address.sin_port))

        // Serve one /healthz answer; poll() bounds the wait so a mutant that
        // never connects cannot hang the thread.
        DispatchQueue.global().async {
            var pfd = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
            guard poll(&pfd, 1, 5000) > 0 else { return }
            let conn = accept(listener, nil, nil)
            guard conn >= 0 else { return }
            var buffer = [UInt8](repeating: 0, count: 4096)
            _ = recv(conn, &buffer, buffer.count, 0)
            let body = "{\"ok\":true}"
            let reply = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + "Content-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n" + body
            _ = reply.withCString { send(conn, $0, strlen($0), 0) }
            close(conn)
        }

        let pendingLock = NSLock()
        var pending: [() -> Void] = []
        let env = AppEnvironment.live(
            pythonPath: "/bin/sh",
            applicationSupport: FileManager.default.temporaryDirectory.path,
            schedule: { _, work in
                pendingLock.lock()
                pending.append(work)
                pendingLock.unlock()
            }
        )
        // A long-lived child whose arguments carry the port, so the supervisor
        // stays .starting until the probe answers.
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "exec /bin/sleep 30", "--port", String(port)],
            environment: [:]
        )
        let supervisor = env.makeSupervisor(spec)
        let running = expectation(description: "supervisor reaches .running")
        supervisor.onStateChange = { state in
            if state == .running { running.fulfill() }
        }
        supervisor.start()
        defer { supervisor.stop() }

        pendingLock.lock()
        let firstPoll = pending.first
        pendingLock.unlock()
        XCTAssertNotNil(firstPoll, "start() must schedule the first health poll")
        firstPoll?()
        wait(for: [running], timeout: 5)
    }
}
