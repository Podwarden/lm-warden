// ShellCoordinatorTests — the non-UI brain of the menu bar app.
//
// Every test is fake-driven: a recording ShellViews, a FakeLauncher-backed
// AppController (in-memory keychain, closure port check, no-op prepare), a
// dictionary SettingsStore, a recording schedule and a settable clock. Nothing
// touches a window server, the real defaults, a socket, a process or a timer.
import XCTest
@testable import LMWarden

// MARK: - Fakes

private final class CoordDictStore: SettingsStore {
    var values: [String: Any] = [:]
    func object(forKey key: String) -> Any? { values[key] }
    func set(_ value: Any?, forKey key: String) { values[key] = value }
}

private final class CoordSecretBackend: SecretBackend {
    var items: [String: Data] = [:]
    func read(service: String, account: String) throws -> Data? { items["\(service)|\(account)"] }
    func write(_ data: Data, service: String, account: String) throws { items["\(service)|\(account)"] = data }
    func delete(service: String, account: String) throws { items["\(service)|\(account)"] = nil }
}

private final class CoordFakeProcess: LaunchedProcess {
    let pid: Int32 = 2000
    private(set) var terminateCount = 0
    func terminate() { terminateCount += 1 }
    func kill() {}
}

private final class CoordFakeLauncher: ProcessLaunching {
    private(set) var specs: [ServerLaunchSpec] = []
    private(set) var processes: [CoordFakeProcess] = []
    var outputs: [(Data) -> Void] = []

    func launch(
        _ spec: ServerLaunchSpec,
        onOutput: @escaping (Data) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws -> LaunchedProcess {
        specs.append(spec)
        outputs.append(onOutput)
        let process = CoordFakeProcess()
        processes.append(process)
        return process
    }
}

/// Everything a test inspects: the ordered event log of view calls and
/// controller-visible effects, the recorded coordinator schedule, the clock.
private final class CoordRecording {
    let launcher = CoordFakeLauncher()
    /// When true the fake health probe answers healthy at once, so a boot
    /// reaches `.running` synchronously; when false the server stays `.starting`.
    var healthy = true
    var freePort: Int? = 8080
    var prepareError: Error?

    var events: [String] = []
    var menus: [MenuBarModel] = []
    var dashboards: [URL] = []
    var prompts: [QuitPrompt] = []
    var promptReplies: [(QuitChoice) -> Void] = []
    var pasteboard: [String] = []
    var logs: [[String]] = []

    var scheduled: [(delay: TimeInterval, work: () -> Void)] = []
    var clock = Date(timeIntervalSince1970: 1_000_000)
    var status: ShellStatus?

    var terminateCount: Int { events.filter { $0 == "terminate" }.count }
    var lastMenu: MenuBarModel? { menus.last }

    /// Fire every closure scheduled so far (not ones they schedule in turn).
    func fireScheduled() {
        let pending = scheduled
        scheduled = []
        for item in pending { item.work() }
    }
}

private func statusWith(inFlight: Int) -> ShellStatus {
    ShellStatus(
        loadedModels: [],
        usedBytes: 0,
        usableBytes: 0,
        inFlight: inFlight > 0 ? [InFlightGroup(who: "alice", model: "qwen", requests: inFlight)] : []
    )
}

final class ShellCoordinatorTests: XCTestCase {
    private var store: CoordDictStore!
    private var recording: CoordRecording!
    private var controller: AppController!

    override func setUp() {
        super.setUp()
        store = CoordDictStore()
        recording = CoordRecording()
        let recording = self.recording!
        let ports = PortSelector(isAvailable: { candidate in
            recording.freePort == candidate
        })
        let environment = AppEnvironment(
            pythonPath: "/usr/bin/python3",
            applicationSupport: "/tmp/warden-coord/support",
            parentPID: 77,
            baseEnvironment: [:],
            store: KeychainStore(backend: CoordSecretBackend()),
            ports: ports,
            prepare: { _ in
                if let error = recording.prepareError { throw error }
            },
            makeSupervisor: { spec in
                ServerSupervisor(
                    spec: spec,
                    launcher: recording.launcher,
                    probe: { completion in
                        if recording.healthy { completion(.healthy) }
                    },
                    // Run the supervisor's own polls at once; the stop grace
                    // kill is a no-op on the fake process.
                    schedule: { _, work in work() },
                    now: { recording.clock }
                )
            }
        )
        controller = AppController(environment: environment)
    }

    override func tearDown() {
        controller?.stop()
        controller = nil
        recording = nil
        store = nil
        super.tearDown()
    }

    private func makeSettings() -> ShellSettings {
        ShellSettings(store: store)
    }

    private func makeViews() -> ShellViews {
        let recording = self.recording!
        return ShellViews(
            refreshMenu: { recording.menus.append($0); recording.events.append("menu") },
            presentFirstRun: { recording.events.append("firstRun") },
            showDashboard: { recording.dashboards.append($0); recording.events.append("dashboard") },
            presentQuitPrompt: { prompt, reply in
                recording.prompts.append(prompt)
                recording.promptReplies.append(reply)
                recording.events.append("prompt")
            },
            copyToPasteboard: { recording.pasteboard.append($0); recording.events.append("copy") },
            showLog: { recording.logs.append($0); recording.events.append("log") },
            showAbout: { recording.events.append("about") },
            terminate: { recording.events.append("terminate") }
        )
    }

    private func makeCoordinator(
        firstRunCompleted: Bool = true,
        quitWaitLimit: TimeInterval = ShellCoordinator.defaultQuitWaitLimit
    ) -> ShellCoordinator {
        let settings = makeSettings()
        settings.firstRunCompleted = firstRunCompleted
        let recording = self.recording!
        return ShellCoordinator(
            controller: controller,
            settings: settings,
            views: makeViews(),
            lanAddress: { "192.168.1.20" },
            status: { recording.status },
            canCheckForUpdates: true,
            schedule: { delay, work in recording.scheduled.append((delay, work)) },
            now: { recording.clock },
            quitWaitLimit: quitWaitLimit
        )
    }

    private var lastHost: String? {
        guard let args = recording.launcher.specs.last?.arguments,
              let index = args.firstIndex(of: "--host") else { return nil }
        return args[index + 1]
    }

    // MARK: - launch

    func testFirstLaunchShowsFirstRunAndDoesNotBoot() {
        let coordinator = makeCoordinator(firstRunCompleted: false)
        coordinator.launch()

        XCTAssertEqual(recording.events, ["firstRun"])
        XCTAssertEqual(recording.launcher.specs.count, 0)
        XCTAssertNil(controller.supervisor)
    }

    func testLaunchAfterFirstRunBootsWithSettingsAndRefreshesMenu() {
        store.values[ShellSettings.Key.shareOnLAN] = true
        store.values[ShellSettings.Key.modelsDir] = "/Volumes/Big/models"
        let coordinator = makeCoordinator()
        coordinator.launch()

        XCTAssertEqual(recording.launcher.specs.count, 1)
        XCTAssertEqual(lastHost, "0.0.0.0")
        XCTAssertEqual(recording.launcher.specs.last?.environment["VW_HF_CACHE_DIR"], "/Volumes/Big/models")
        XCTAssertEqual(recording.events.last, "menu")
        XCTAssertEqual(recording.lastMenu?.statusWord, "Running")
        XCTAssertEqual(recording.lastMenu?.address, "192.168.1.20:8080")
        XCTAssertFalse(recording.events.contains("firstRun"))
        XCTAssertFalse(recording.events.contains("dashboard"))
    }

    func testBootErrorLeavesMenuStoppedWithoutCrashing() {
        recording.freePort = nil
        let coordinator = makeCoordinator()
        coordinator.launch()

        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
        XCTAssertEqual(recording.events, ["menu"])
    }

    // MARK: - finishFirstRun

    func testFinishFirstRunSavesAllFourChoicesBootsAndShowsDashboard() {
        let coordinator = makeCoordinator(firstRunCompleted: false)
        coordinator.launch()
        coordinator.finishFirstRun(FirstRunResult(
            shareOnLAN: true,
            modelsDir: "/Volumes/Big/models",
            excludeModelsFromBackup: false,
            openAtLogin: false
        ))

        let settings = makeSettings()
        XCTAssertTrue(settings.firstRunCompleted)
        XCTAssertTrue(settings.shareOnLAN)
        XCTAssertEqual(settings.modelsDir, "/Volumes/Big/models")
        XCTAssertFalse(settings.excludeModelsFromBackup)
        XCTAssertFalse(settings.openAtLogin)
        XCTAssertEqual(recording.launcher.specs.count, 1)
        XCTAssertEqual(lastHost, "0.0.0.0")
        XCTAssertEqual(recording.dashboards, [URL(string: "http://127.0.0.1:8080/ui/")!])
        XCTAssertEqual(recording.events.last, "menu")
    }

    func testFinishFirstRunBootFailureShowsNoDashboard() {
        recording.prepareError = NSError(domain: "t", code: 1)
        let coordinator = makeCoordinator(firstRunCompleted: false)
        coordinator.finishFirstRun(FirstRunResult(
            shareOnLAN: false, modelsDir: "/tmp/m", excludeModelsFromBackup: true, openAtLogin: true
        ))

        XCTAssertTrue(makeSettings().firstRunCompleted)
        XCTAssertTrue(recording.dashboards.isEmpty)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
    }

    // MARK: - perform: start / stop

    func testStartAndStopServer() {
        let coordinator = makeCoordinator()
        coordinator.perform(.startServer)
        XCTAssertEqual(recording.launcher.specs.count, 1)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Running")
        XCTAssertEqual(recording.lastMenu?.sections[1][0].action, .stopServer)

        coordinator.perform(.stopServer)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.launcher.processes.last?.terminateCount, 1)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
        XCTAssertEqual(recording.lastMenu?.sections[1][0].action, .startServer)
        XCTAssertEqual(recording.events.last, "menu")
    }

    func testStartWhileRunningDoesNotRelaunch() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.startServer)
        XCTAssertEqual(recording.launcher.specs.count, 1)
    }

    func testStopTwiceIsIdempotent() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.stopServer)
        coordinator.perform(.stopServer)
        XCTAssertEqual(recording.launcher.processes.last?.terminateCount, 1)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
        XCTAssertTrue(recording.scheduled.isEmpty)
    }

    func testSupervisorStateChangeRefreshesMenu() {
        recording.healthy = false
        let coordinator = makeCoordinator()
        coordinator.launch()
        XCTAssertEqual(recording.lastMenu?.statusWord, "Starting")

        let menusBefore = recording.menus.count
        // The supervisor moves on its own (stop from outside the coordinator).
        controller.supervisor?.stop()
        XCTAssertGreaterThan(recording.menus.count, menusBefore)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
    }

    func testStartFromStoppedSupervisorRelaunches() {
        recording.healthy = false
        let coordinator = makeCoordinator()
        coordinator.launch()
        // The supervisor stopped behind the controller's back: Start must
        // launch afresh, not return the cached URL of the dead supervisor.
        controller.supervisor?.stop()
        recording.healthy = true
        coordinator.perform(.startServer)
        XCTAssertEqual(recording.launcher.specs.count, 2)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Running")
    }

    // MARK: - perform: share toggle

    func testShareToggleWhileRunningRestartsWithLANHost() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        XCTAssertEqual(lastHost, "127.0.0.1")

        coordinator.perform(.toggleShareOnLAN)

        XCTAssertTrue(makeSettings().shareOnLAN)
        XCTAssertEqual(recording.launcher.specs.count, 2)
        XCTAssertEqual(recording.launcher.processes.first?.terminateCount, 1)
        XCTAssertEqual(lastHost, "0.0.0.0")
        XCTAssertEqual(recording.lastMenu?.statusWord, "Running")
        XCTAssertEqual(recording.lastMenu?.sections[0][2].checked, true)
    }

    func testShareToggleRestartRefreshesMenuOnceWithFinalState() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        let menusBefore = recording.menus.count

        coordinator.perform(.toggleShareOnLAN)

        // The old supervisor is detached before it stops: no intermediate
        // "Stopped" menu flickers in mid-restart, only the final refresh.
        XCTAssertEqual(recording.menus.count - menusBefore, 1)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Running")
    }

    func testStopRefreshesMenuOnce() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        let menusBefore = recording.menus.count
        coordinator.perform(.stopServer)
        XCTAssertEqual(recording.menus.count - menusBefore, 1)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
    }

    func testShareToggleBackWhileRunningRestartsOnLoopback() {
        store.values[ShellSettings.Key.shareOnLAN] = true
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.toggleShareOnLAN)
        XCTAssertFalse(makeSettings().shareOnLAN)
        XCTAssertEqual(lastHost, "127.0.0.1")
        XCTAssertEqual(recording.launcher.specs.count, 2)
    }

    func testShareToggleWhileStartingRestarts() {
        recording.healthy = false
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.toggleShareOnLAN)
        XCTAssertEqual(recording.launcher.specs.count, 2)
        XCTAssertEqual(lastHost, "0.0.0.0")
    }

    func testShareToggleWhileStoppedDoesNotBoot() {
        let coordinator = makeCoordinator()
        coordinator.perform(.toggleShareOnLAN)

        XCTAssertTrue(makeSettings().shareOnLAN)
        XCTAssertEqual(recording.launcher.specs.count, 0)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
        XCTAssertEqual(recording.lastMenu?.sections[0][2].checked, true)
    }

    // MARK: - perform: dashboard, copy, log, about

    func testOpenDashboardIgnoredWhenStopped() {
        let coordinator = makeCoordinator()
        coordinator.perform(.openDashboard)
        XCTAssertTrue(recording.dashboards.isEmpty)
        XCTAssertEqual(recording.events, ["menu"])
    }

    func testOpenDashboardIgnoredWhileStarting() {
        recording.healthy = false
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.openDashboard)
        XCTAssertTrue(recording.dashboards.isEmpty)
    }

    func testOpenDashboardWhenRunning() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.openDashboard)
        XCTAssertEqual(recording.dashboards, [URL(string: "http://127.0.0.1:8080/ui/")!])
        XCTAssertEqual(recording.events.suffix(2), ["dashboard", "menu"])
    }

    func testCopyAPIAddressCopiesModelCopyText() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        coordinator.perform(.copyAPIAddress)
        XCTAssertEqual(recording.pasteboard, ["http://127.0.0.1:8080/v1"])
        XCTAssertEqual(recording.events.suffix(2), ["copy", "menu"])
    }

    func testShowLogPassesSupervisorLines() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.launcher.outputs.last?(Data("one\ntwo\n".utf8))
        coordinator.perform(.showLog)
        XCTAssertEqual(recording.logs, [["one", "two"]])
        XCTAssertEqual(recording.events.suffix(2), ["log", "menu"])
    }

    func testAboutShowsAbout() {
        let coordinator = makeCoordinator()
        coordinator.perform(.about)
        XCTAssertEqual(recording.events, ["about", "menu"])
    }

    // MARK: - Status feed

    func testStatusDidChangeRefreshesMenuOnce() {
        let coordinator = makeCoordinator()
        let menusBefore = recording.menus.count

        coordinator.statusDidChange()

        XCTAssertEqual(recording.menus.count - menusBefore, 1)
        XCTAssertEqual(recording.events.last, "menu")
    }

    func testStatusDidChangeUsesLatestStatus() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)

        coordinator.statusDidChange()

        XCTAssertNotNil(recording.lastMenu?.models)
        XCTAssertTrue(recording.lastMenu?.models?.memoryLine.contains("2 requests in flight") ?? false)
    }

    func testStatusDidChangeDoesNotLaunchOrStop() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        let specsBefore = recording.launcher.specs.count
        let terminateBefore = recording.terminateCount

        coordinator.statusDidChange()

        XCTAssertEqual(recording.launcher.specs.count, specsBefore)
        XCTAssertEqual(recording.terminateCount, terminateBefore)
        // The server is neither stopped nor restarted.
        XCTAssertNotNil(controller.supervisor)
        XCTAssertEqual(recording.launcher.processes.count, 1)
        XCTAssertEqual(recording.launcher.processes.last?.terminateCount, 0)
    }

    func testStatusDidChangeIsIdempotent() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)

        coordinator.statusDidChange()
        coordinator.statusDidChange()

        let last = recording.menus.suffix(2).map { $0.models?.memoryLine }
        XCTAssertEqual(last.count, 2)
        XCTAssertEqual(last.first, last.last)
        XCTAssertEqual(recording.launcher.specs.count, 1)
    }

    func testStatusDidChangeDuringQuitWaitKeepsWaiting() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)
        XCTAssertEqual(recording.scheduled.count, 1)

        // A feed update mid-wait refreshes the menu but neither quits, nor
        // re-prompts, nor starts a second poll chain, nor reopens Quit.
        recording.status = statusWith(inFlight: 1)
        coordinator.statusDidChange()
        XCTAssertEqual(recording.events.last, "menu")
        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertEqual(recording.prompts.count, 1)
        XCTAssertEqual(recording.scheduled.count, 1)
        XCTAssertNotNil(controller.supervisor)
        coordinator.requestQuit()
        XCTAssertEqual(recording.prompts.count, 1)

        // The original wait still finishes the quit.
        recording.status = statusWith(inFlight: 0)
        recording.clock += 2
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
        XCTAssertNil(controller.supervisor)
    }

    func testStatusDidChangeBeforeLaunchShowsStopped() {
        let coordinator = makeCoordinator()

        coordinator.statusDidChange()

        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.lastMenu?.statusWord, "Stopped")
    }

    // MARK: - quit

    func testQuitWithNilStatusTerminatesAtOnce() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = nil
        coordinator.perform(.quit)

        XCTAssertTrue(recording.prompts.isEmpty)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.launcher.processes.last?.terminateCount, 1)
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testQuitStopsBeforeTerminating() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        var stoppedAtTerminate: Bool?
        let recording = self.recording!
        let controller = self.controller!
        let views = makeViews()
        let coord2 = ShellCoordinator(
            controller: controller,
            settings: makeSettings(),
            views: ShellViews(
                refreshMenu: views.refreshMenu,
                presentFirstRun: views.presentFirstRun,
                showDashboard: views.showDashboard,
                presentQuitPrompt: views.presentQuitPrompt,
                copyToPasteboard: views.copyToPasteboard,
                showLog: views.showLog,
                showAbout: views.showAbout,
                terminate: {
                    stoppedAtTerminate = controller.supervisor == nil
                    recording.events.append("terminate")
                }
            ),
            lanAddress: { nil },
            status: { nil },
            canCheckForUpdates: false,
            schedule: { _, _ in }
        )
        _ = coordinator
        coord2.requestQuit()
        XCTAssertEqual(stoppedAtTerminate, true)
    }

    func testQuitWhenStoppedWithBusyStatusQuitsAtOnce() {
        let coordinator = makeCoordinator()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        XCTAssertTrue(recording.prompts.isEmpty)
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testQuitWithRequestsInFlightShowsPrompt() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.perform(.quit)

        XCTAssertEqual(recording.prompts.count, 1)
        XCTAssertEqual(recording.prompts.first?.title, "Quit LM Warden?")
        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertNotNil(controller.supervisor)
    }

    /// Install and Relaunch already asked: a confirmed quit with requests in
    /// flight stops the server and terminates without a second prompt.
    func testQuitConfirmedWithRequestsInFlightTerminatesWithoutPrompt() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.quitConfirmed()

        XCTAssertTrue(recording.prompts.isEmpty)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.terminateCount, 1)
        coordinator.quitConfirmed()
        coordinator.requestQuit()
        XCTAssertEqual(recording.terminateCount, 1, "terminates exactly once")
    }

    /// A confirmed quit while the Quit prompt is open supersedes it: the
    /// prompt's late answer is inert.
    func testQuitConfirmedSupersedesAnOpenPrompt() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        XCTAssertEqual(recording.prompts.count, 1)
        coordinator.quitConfirmed()
        XCTAssertEqual(recording.terminateCount, 1)
        recording.promptReplies.first?(.quitNow)
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testWaitThenQuitPollsEveryTwoSecondsUntilZeroThenTerminates() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)

        XCTAssertEqual(recording.scheduled.map(\.delay), [2])
        XCTAssertEqual(recording.terminateCount, 0)

        // Still busy: poll again in 2 s, server still up.
        recording.status = statusWith(inFlight: 1)
        recording.clock += 2
        recording.fireScheduled()
        XCTAssertEqual(recording.scheduled.map(\.delay), [2])
        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertNotNil(controller.supervisor)

        // Drained: stop and terminate, no further poll.
        recording.status = statusWith(inFlight: 0)
        recording.clock += 2
        recording.fireScheduled()
        XCTAssertTrue(recording.scheduled.isEmpty)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testWaitThenQuitStopsWhenFeedGoesAway() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 3)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)
        recording.status = nil
        recording.clock += 2
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testWaitThenQuitJustBeforeUpperBoundKeepsPolling() {
        let coordinator = makeCoordinator(quitWaitLimit: 10)
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)

        recording.clock += 9.999
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertEqual(recording.scheduled.count, 1)
        XCTAssertNotNil(controller.supervisor)
    }

    func testWaitThenQuitAtExactUpperBoundFallsThroughToQuitNow() {
        let coordinator = makeCoordinator(quitWaitLimit: 10)
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)

        recording.clock += 10
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
        XCTAssertNil(controller.supervisor)
        XCTAssertTrue(recording.scheduled.isEmpty)
    }

    func testWaitThenQuitPastUpperBoundFallsThroughToQuitNow() {
        let coordinator = makeCoordinator(quitWaitLimit: 10)
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)

        for _ in 0..<4 {
            recording.clock += 2
            recording.fireScheduled()
        }
        XCTAssertEqual(recording.terminateCount, 0)
        recording.clock += 2.5
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
        XCTAssertTrue(recording.scheduled.isEmpty)
    }

    func testDefaultQuitWaitLimitIsBounded() {
        XCTAssertGreaterThan(ShellCoordinator.defaultQuitWaitLimit, ShellCoordinator.quitPollInterval)
        XCTAssertLessThanOrEqual(ShellCoordinator.defaultQuitWaitLimit, 600)
        XCTAssertEqual(ShellCoordinator.quitPollInterval, 2)
    }

    func testQuitNowChoiceStopsAndTerminates() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.quitNow)

        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.terminateCount, 1)
        XCTAssertTrue(recording.scheduled.isEmpty)
    }

    func testCancelLeavesEverythingRunning() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.cancel)

        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertNotNil(controller.supervisor)
        XCTAssertEqual(controller.supervisor?.state, .running)
        XCTAssertTrue(recording.scheduled.isEmpty)

        // A later quit asks again.
        coordinator.requestQuit()
        XCTAssertEqual(recording.prompts.count, 2)
    }

    func testSecondQuitRequestWhilePromptOpenIsIgnored() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        coordinator.requestQuit()
        XCTAssertEqual(recording.prompts.count, 1)
    }

    func testSecondQuitRequestWhileWaitingIsIgnored() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        recording.promptReplies.first?(.waitThenQuit)
        XCTAssertEqual(recording.scheduled.count, 1)

        // A second Quit while Wait-Then-Quit runs neither re-prompts, nor
        // quits at once, nor starts a second poll chain.
        coordinator.requestQuit()
        XCTAssertEqual(recording.prompts.count, 1)
        XCTAssertEqual(recording.terminateCount, 0)
        XCTAssertEqual(recording.scheduled.count, 1)
        XCTAssertNotNil(controller.supervisor)

        // The original wait still finishes the quit.
        recording.status = statusWith(inFlight: 0)
        recording.clock += 2
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testQuitDuringFirstRunTerminatesWithoutBooting() {
        // First-run close quits the app (OVERSEER 2026-10-20, onCancel).
        let coordinator = makeCoordinator(firstRunCompleted: false)
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()

        XCTAssertTrue(recording.prompts.isEmpty)
        XCTAssertEqual(recording.launcher.specs.count, 0)
        XCTAssertNil(controller.supervisor)
        XCTAssertEqual(recording.terminateCount, 1)
        XCTAssertFalse(makeSettings().firstRunCompleted)
    }

    func testTerminateHappensOnceEvenIfQuitRequestedAgain() {
        let coordinator = makeCoordinator()
        coordinator.requestQuit()
        coordinator.requestQuit()
        XCTAssertEqual(recording.terminateCount, 1)
    }

    func testStalePollAfterQuitNowIsInert() {
        let coordinator = makeCoordinator()
        coordinator.launch()
        recording.status = statusWith(inFlight: 2)
        coordinator.requestQuit()
        let reply = recording.promptReplies.first!
        reply(.waitThenQuit)
        reply(.quitNow)
        XCTAssertEqual(recording.terminateCount, 1)
        recording.status = statusWith(inFlight: 0)
        recording.fireScheduled()
        XCTAssertEqual(recording.terminateCount, 1)
    }

    // MARK: - weak capture

    func testCallbacksDoNotRetainCoordinator() {
        weak var weakCoordinator: ShellCoordinator?
        var reply: ((QuitChoice) -> Void)?
        do {
            let coordinator = makeCoordinator()
            coordinator.launch()
            recording.status = statusWith(inFlight: 2)
            coordinator.requestQuit()
            reply = recording.promptReplies.first
            reply?(.waitThenQuit)
            weakCoordinator = coordinator
        }
        XCTAssertNil(weakCoordinator)
        // Firing the stale callbacks after release does nothing.
        recording.fireScheduled()
        reply?(.quitNow)
        controller.supervisor?.stop()
        XCTAssertEqual(recording.terminateCount, 0)
    }
}
