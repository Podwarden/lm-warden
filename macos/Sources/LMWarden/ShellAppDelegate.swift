// ShellAppDelegate — the NSApplication delegate of the GUI menu bar app
// (planner iter 42: LaunchMode `.gui` → NSApplication, `.accessory` policy).
//
// It composes every P7 piece into one running app and owns no decisions of
// its own: quit, boot and menu choices are the coordinator's, copy and
// layout the models', notifications the NotificationPolicy's.
//
// Threads. Two confinement domains:
// - The SHELL QUEUE (a private serial queue, `ShellCore`) owns the non-UI
//   core: ShellSettings, AppController, every ServerSupervisor it builds,
//   ShellStatusFeed and ShellCoordinator. Boot (Keychain, port, launch) runs
//   there, so a Keychain prompt never freezes the menu. Every injected
//   schedule, the feed's deliveries, the health probe's completions and the
//   child process's output and exit callbacks are hopped onto it.
// - The MAIN THREAD (this delegate, `@MainActor`) owns AppKit: the status
//   item, the windows, the quit alert, the notifier, the login item, the
//   system-signal observer, the SIGTERM source and the termination gate.
// Main → core calls go through `ShellCore.async`; core → main view effects
// go through `ShellCore.onMain`. Nothing crosses domains synchronously.
//
// Termination: every quit goes through the coordinator. A quit the
// coordinator starts (menu Quit, ⌘Q in the menu) reaches `views.terminate`,
// which approves and calls `NSApp.terminate`; `applicationShouldTerminate`
// then answers `.terminateNow`. A quit from elsewhere (Dock, logout, SIGTERM,
// closing the first-run window) arrives at `applicationShouldTerminate`
// first: it answers `.terminateLater` and asks the coordinator, which either
// terminates (reply true) or the user cancels the quit prompt (reply false).
// `TerminationGate` guarantees exactly one reply per `.terminateLater`.
//
// Quit watchdog (planner iter 58). Boot's Keychain read and the first-run
// facts read run on the shell queue, so a Keychain access prompt blocks it
// and a forwarded quit would wait for the user to answer the prompt (logout
// and SIGTERM hang). Every quit forwarded to the shell queue (the
// `.terminateLater` request and the menu Quit) therefore arms a main-thread
// deadline of `TerminationGate.unacknowledgedQuitLimit` (5 s); ShellCore
// acknowledges as the first statement of handling the quit. Unacknowledged
// at the deadline, a pending request is replied `true` (or, with none
// pending, `NSApp.terminate` answers `.terminateNow`) without the graceful
// coordinator path; acknowledged in time, the deadline is inert and the
// coordinator's quit prompt waits for the user as before. Race belt: if the
// Keychain read returns just after the forced exit decision and the boot
// launches a child in that instant, the child's VW_PARENT_PID watch
// (app/runtime/parent_watch.py, 2 s poll) and the process-group SIGTERM reap
// it; there is no extra code for it.
//
// Status feed: started once a boot has produced a shell token, stopped
// before Stop/Start/Share-on-LAN reach the coordinator and before terminate,
// and restarted against the new boot after every restart. Its `onChange`
// re-renders the menu through `ShellCoordinator.statusDidChange()`. The shell
// token is only ever read inside the feed's endpoint closure; it is never
// stored here or logged.
//
// Untested glue (needs a window server or real system services): this
// delegate, ShellCore's closure wiring, SystemMacFacts and the queue-hopping
// launcher. The decisions they lean on are the tested models plus the two
// small pure types TerminationGate and ShellEventTracker (own files, own tests).
import AppKit
import Foundation

// MARK: - Queue confinement helpers

/// Carries a value confined to one queue across a `@Sendable` boundary. Only
/// used for values that are created and consumed on the same queue.
private struct QueueConfined<Value>: @unchecked Sendable {
    let value: Value
}

/// A launcher that hops the child's output and exit callbacks onto the shell
/// queue, so the ServerSupervisor (not thread-safe) only ever runs there.
private struct QueueHoppingLauncher: ProcessLaunching {
    let base: ProcessLaunching
    let queue: DispatchQueue

    func launch(
        _ spec: ServerLaunchSpec,
        onOutput: @escaping (Data) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws -> LaunchedProcess {
        let output = QueueConfined(value: onOutput)
        let exit = QueueConfined(value: onExit)
        let queue = self.queue
        return try base.launch(
            spec,
            onOutput: { data in queue.async { output.value(data) } },
            onExit: { status in queue.async { exit.value(status) } }
        )
    }
}

// MARK: - Shell core

/// The non-UI core, confined to its serial queue (see the file header).
/// `@unchecked Sendable` because every member is only touched on `queue`;
/// the main thread reaches it only through `async`.
final class ShellCore: @unchecked Sendable {
    let queue = DispatchQueue(label: AppVersion.bundleIdentifier + ".shell", qos: .userInitiated)
    private let settings: ShellSettings
    private let controller: AppController
    private let feed: ShellStatusFeed
    private let keychain: KeychainStore
    private var coordinator: ShellCoordinator?
    private weak var ui: ShellAppDelegate?

    /// The supervisor the feed is currently polling (nil: feed stopped).
    private var feedBoot: ObjectIdentifier?
    /// True while an action that stops or boots is in progress: the menu
    /// refreshes inside it must not restart the feed half-way.
    private var holdFeed = false
    private var tracker = ShellEventTracker()

    private let canCheckForUpdates: Bool

    init(
        ui: ShellAppDelegate,
        pythonPath: String,
        applicationSupport: String,
        webuiDir: String?,
        canCheckForUpdates: Bool
    ) {
        self.ui = ui
        self.canCheckForUpdates = canCheckForUpdates
        let queue = self.queue
        let settings = ShellSettings(store: UserDefaults.standard)
        let schedule = Self.schedule(on: queue)
        let live = AppEnvironment.live(pythonPath: pythonPath, applicationSupport: applicationSupport, schedule: schedule)
        let environment = AppEnvironment(
            pythonPath: live.pythonPath,
            applicationSupport: live.applicationSupport,
            parentPID: live.parentPID,
            baseEnvironment: live.baseEnvironment,
            store: live.store,
            ports: live.ports,
            // Read the backup choice at boot time, so the first boot after
            // first run honours what the user just picked.
            prepare: { paths in
                try AppEnvironment.livePrepare(paths, excludeFromBackup: settings.excludeModelsFromBackup)
            },
            makeSupervisor: { spec in
                ServerSupervisor(
                    spec: spec,
                    launcher: QueueHoppingLauncher(base: SystemProcessLauncher(), queue: queue),
                    probe: { done in
                        let completion = QueueConfined(value: done)
                        HealthProbe(port: AppEnvironment.port(of: spec) ?? 0).check { result in
                            queue.async { completion.value(result) }
                        }
                    },
                    schedule: schedule
                )
            },
            webuiDir: webuiDir
        )
        let controller = AppController(environment: environment)
        self.settings = settings
        self.controller = controller
        self.keychain = live.store
        self.feed = ShellStatusFeed(
            endpoint: { [controller] in
                guard let port = controller.port, let token = controller.shellToken else {
                    return nil
                }
                return (port, token)
            },
            schedule: schedule,
            deliver: { work in
                let work = QueueConfined(value: work)
                queue.async { work.value() }
            }
        )
    }

    /// Run `work` on the shell queue.
    func async(_ work: @escaping @Sendable (ShellCore) -> Void) {
        queue.async { work(self) }
    }

    /// `delay`-seconds scheduling on `queue` (the supervisor, feed and
    /// coordinator all take this shape).
    private static func schedule(on queue: DispatchQueue) -> (TimeInterval, @escaping () -> Void) -> Void {
        { delay, work in
            let work = QueueConfined(value: work)
            queue.asyncAfter(deadline: .now() + delay) { work.value() }
        }
    }

    /// Run `body` on the main actor with the delegate, if it is still alive.
    private func onMain(_ body: @escaping @MainActor @Sendable (ShellAppDelegate) -> Void) {
        let ui = self.ui
        DispatchQueue.main.async { [weak ui] in
            MainActor.assumeIsolated {
                if let ui {
                    body(ui)
                }
            }
        }
    }

    // MARK: Shell-queue entry points

    /// Build the coordinator, show the initial menu, then launch (first run
    /// or boot). Shell queue only.
    func launch() {
        let coordinator = ShellCoordinator(
            controller: controller,
            settings: settings,
            views: makeViews(),
            lanAddress: { LANAddress.pick(LANAddress.systemInterfaces()) },
            status: { [feed] in feed.latest },
            canCheckForUpdates: canCheckForUpdates,
            schedule: Self.schedule(on: queue)
        )
        feed.onChange = { [weak coordinator] _ in
            coordinator?.statusDidChange()
        }
        self.coordinator = coordinator
        // A menu (with Quit) exists before the first boot, which may wait on
        // a Keychain prompt.
        coordinator.statusDidChange()
        coordinator.launch()
    }

    func perform(_ action: MenuAction) {
        guard let coordinator else {
            return
        }
        switch action {
        case .startServer, .stopServer, .toggleShareOnLAN:
            // The feed stops before the server stops or restarts, then
            // follows the new boot (if any) once the action is complete.
            holdFeed = true
            stopFeed()
            coordinator.perform(action)
            holdFeed = false
            syncFeed()
        default:
            coordinator.perform(action)
        }
    }

    func finishFirstRun(_ result: FirstRunResult) {
        coordinator?.finishFirstRun(result)
    }

    /// A quit request forwarded by `applicationShouldTerminate`. The
    /// acknowledgement comes first, before anything that could block.
    func requestQuit(acknowledging generation: Int) {
        acknowledgeQuit(generation)
        coordinator?.requestQuit()
    }

    /// A pre-confirmed quit forwarded by `applicationShouldTerminate` (the
    /// termination Sparkle requests after Install and Relaunch).
    func quitConfirmed(acknowledging generation: Int) {
        acknowledgeQuit(generation)
        coordinator?.quitConfirmed()
    }

    /// The menu Quit item, forwarded by the delegate.
    func performQuit(acknowledging generation: Int) {
        acknowledgeQuit(generation)
        perform(.quit)
    }

    /// Tell the delegate the shell queue picked up the quit forward of
    /// `generation`, so its watchdog deadline stays inert.
    private func acknowledgeQuit(_ generation: Int) {
        onMain { $0.quitAcknowledged(generation) }
    }

    /// Flip Open at Login in the settings, refresh the menu, and hand the new
    /// value to the login item on the main thread.
    func toggleOpenAtLogin() {
        guard let coordinator else {
            return
        }
        settings.openAtLogin.toggle()
        let on = settings.openAtLogin
        coordinator.perform(.toggleOpenAtLogin)
        onMain { $0.applyOpenAtLogin(on, fromToggle: true) }
    }

    // MARK: Feed lifecycle

    private func stopFeed() {
        feedBoot = nil
        feed.stop()
    }

    /// Follow the controller's current boot: stop on no boot, (re)start on a
    /// new one. The supervisor's identity marks a boot, so the token itself
    /// is never stored here.
    private func syncFeed() {
        guard !holdFeed else {
            return
        }
        let current = controller.shellToken == nil ? nil : controller.supervisor.map(ObjectIdentifier.init)
        guard current != feedBoot else {
            return
        }
        stopFeed()
        if let current {
            feedBoot = current
            feed.start()
        }
    }

    // MARK: Views

    private func makeViews() -> ShellViews {
        ShellViews(
            refreshMenu: { [weak self] model in
                self?.menuDidRefresh(model)
            },
            presentFirstRun: { [weak self] in
                self?.presentFirstRun()
            },
            showDashboard: { [weak self] url in
                self?.onMain { $0.showDashboard(url) }
            },
            reloadDashboardIfFailed: { [weak self] in
                self?.onMain { $0.reloadDashboardIfFailed() }
            },
            presentQuitPrompt: { [weak self] prompt, reply in
                let reply = QueueConfined(value: reply)
                guard let self else {
                    return
                }
                self.onMain { [self] ui in
                    ui.presentQuitPrompt(prompt) { choice in
                        self.async { _ in reply.value(choice) }
                    }
                }
            },
            copyToPasteboard: { [weak self] text in
                self?.onMain { $0.copyToPasteboard(text) }
            },
            showLog: { [weak self] lines in
                self?.onMain { $0.showLog(lines) }
            },
            showAbout: { [weak self] in
                self?.onMain { $0.showAbout() }
            },
            terminate: { [weak self] in
                guard let self else {
                    return
                }
                self.feed.onChange = nil
                self.stopFeed()
                self.onMain { $0.coordinatorDidTerminate() }
            }
        )
    }

    /// Every menu refresh: keep the feed on the current boot, derive the
    /// notification events, and render on the main thread.
    private func menuDidRefresh(_ model: MenuBarModel) {
        syncFeed()
        let state = controller.supervisor?.state ?? .stopped
        let status = feed.latest
        let events = tracker.events(state: state, status: status)
        let running = state == .running
        onMain { ui in
            ui.render(model, serverRunning: running, status: status, events: events)
        }
    }

    /// Gather the first-run facts off the main thread (the Keychain read may
    /// prompt), then show the window.
    private func presentFirstRun() {
        let home = NSHomeDirectory()
        let facts = SystemMacFacts.read(homeDir: home)
        let defaultModelsDir = (try? AppPaths.resolve(applicationSupport: AppPaths.defaultApplicationSupport()).modelsDir)
            ?? home
        let stored = (try? keychain.string(for: CookieSecret.account)) ?? nil
        let keychainHasSecret = stored.map(CookieSecret.isValid) ?? false
        let lan = LANAddress.pick(LANAddress.systemInterfaces())
        onMain { ui in
            ui.showFirstRun(
                FirstRunModel(
                    facts: facts,
                    defaultModelsDir: defaultModelsDir,
                    homeDir: home,
                    port: ShellCoordinator.defaultPort,
                    lanAddress: lan,
                    keychainHasSecret: keychainHasSecret,
                    freeSpace: SystemMacFacts.freeSpace,
                    volumeName: SystemMacFacts.volumeName
                )
            )
        }
    }
}

// MARK: - System facts

/// Reads the Mac facts the first-run Welcome step shows. Untested glue over
/// sysctl and URL resource values.
enum SystemMacFacts {
    static func read(homeDir: String) -> MacFacts {
        MacFacts(
            chipName: sysctlString("machdep.cpu.brand_string"),
            isAppleSilicon: sysctlInt("hw.optional.arm64") == 1,
            memoryBytes: ProcessInfo.processInfo.physicalMemory,
            freeDiskBytes: freeSpace(homeDir)
        )
    }

    /// Free space for important usage on the volume holding `path` (or its
    /// nearest existing ancestor: the models folder may not exist yet).
    static func freeSpace(_ path: String) -> Int64? {
        let values = try? nearestExisting(path).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    static func volumeName(_ path: String) -> String? {
        let values = try? nearestExisting(path).resourceValues(forKeys: [.volumeLocalizedNameKey])
        return values?.volumeLocalizedName
    }

    private static func nearestExisting(_ path: String) -> URL {
        var url = URL(fileURLWithPath: path)
        while !FileManager.default.fileExists(atPath: url.path), url.path != "/" {
            url.deleteLastPathComponent()
        }
        return url
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return nil
        }
        let value = buffer.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
            .trimmingCharacters(in: .whitespaces)
        return value.isEmpty ? nil : value
    }

    private static func sysctlInt(_ name: String) -> Int32? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else {
            return nil
        }
        return value
    }
}

// MARK: - App delegate

@MainActor
final class ShellAppDelegate: NSObject, NSApplicationDelegate {
    /// The embedded interpreter's file name. scripts/macos/bundle-python.sh
    /// renames python-build-standalone's bin/python3.12 to this (Metal design
    /// §3), so Activity Monitor, the Background Items list, the firewall
    /// dialog and `ps` show the product, not "python3". 16 bytes: exactly
    /// MAXCOMLEN, so the kernel's process name keeps all of it.
    nonisolated static let bundledInterpreterName = "LM Warden Server"

    /// The python interpreter inside the bundle (laid down by
    /// bundle-python.sh); a missing interpreter makes the boot fail, which the
    /// coordinator shows as the menu's Stopped/Failed state, never a crash.
    nonisolated static func bundledPythonPath(bundlePath: String = Bundle.main.bundlePath) -> String {
        bundlePath + "/Contents/Resources/python/bin/" + bundledInterpreterName
    }

    /// The web UI export inside the bundle (laid down by bundle-webui.sh).
    nonisolated static func bundledWebUIPath(bundlePath: String = Bundle.main.bundlePath) -> String {
        bundlePath + "/Contents/Resources/webui"
    }

    /// The bundled export to hand the server as VW_WEBUI_DIR, or nil when the
    /// bundle carries none (no `index.html`): the server refuses to start on
    /// a VW_WEBUI_DIR without one, so a bundle built without the UI (the GUI
    /// smoke's, a `make macos-app` dev build) still boots its API.
    nonisolated static func bundledWebUIDir(
        bundlePath: String = Bundle.main.bundlePath,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> String? {
        let dir = bundledWebUIPath(bundlePath: bundlePath)
        return fileExists(dir + "/index.html") ? dir : nil
    }

    private(set) var core: ShellCore?
    private var gate = TerminationGate()
    /// Arms the unacknowledged-quit deadline on the main thread. Injected so
    /// the deadline is a seam; the default is `DispatchQueue.main.asyncAfter`.
    var scheduleQuitDeadline: (TimeInterval, @escaping @MainActor () -> Void) -> Void = { delay, work in
        let work = QueueConfined(value: work)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
            MainActor.assumeIsolated {
                work.value()
            }
        }
    }

    private var statusItem: StatusItemController?
    private var firstRun: FirstRunWindowController?
    private let dashboard = DashboardWindowController()
    private let logWindow = LogWindowController()
    private var about: AboutWindowController?
    /// SparkleUpdater when the bundle carries SUFeedURL + SUPublicEDKey,
    /// else NoUpdater (a dev build). Built at launch.
    private var updater: UpdaterControlling = NoUpdater()
    private let notifier = UserNotifier()
    private let loginItem = LoginItem(backend: SystemLoginItemBackend())
    private var signalObserver: SystemSignalObserver?
    private var policy = NotificationPolicy()
    /// Held for the process lifetime so the SIGTERM source is never released.
    private var sigtermSource: DispatchSourceSignal?

    /// Main-thread mirror of the core's last menu refresh, for the About
    /// window's quit decision and the battery rule's "model loaded".
    private var serverRunning = false
    private var latestStatus: ShellStatus?

    // MARK: - NSApplicationDelegate

    func applicationWillFinishLaunching(_ notification: Notification) {
        installSIGTERM()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = StatusItemController(onAction: { [weak self] action in
            self?.menuAction(action)
        })
        let observer = SystemSignalObserver(
            onEvent: { [weak self] event in self?.handle(event) },
            modelLoaded: { [weak self] in self?.latestStatus?.loadedModels.isEmpty == false }
        )
        signalObserver = observer
        observer.start()

        let updater = SparkleUpdaterFactory.make(settings: ShellSettings(store: UserDefaults.standard))
        updater.onStateChange = { [weak self] in
            self?.about?.refresh()
        }
        updater.onInstallFailed = { [weak self] in
            self?.withdrawQuit()
        }
        self.updater = updater

        let core = ShellCore(
            ui: self,
            pythonPath: Self.bundledPythonPath(),
            applicationSupport: AppPaths.defaultApplicationSupport(),
            webuiDir: Self.bundledWebUIDir(),
            canCheckForUpdates: updater.canCheckForUpdates
        )
        self.core = core
        core.async { $0.launch() }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        switch gate.shouldTerminate(hasCoordinator: core != nil) {
        case .now:
            return .terminateNow
        case .cancel:
            return .terminateCancel
        case .laterAskCoordinator:
            let generation = forwardQuit()
            core?.async { $0.requestQuit(acknowledging: generation) }
            return .terminateLater
        case .laterQuitConfirmed:
            let generation = forwardQuit()
            core?.async { $0.quitConfirmed(acknowledging: generation) }
            return .terminateLater
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        signalObserver?.stop()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    // MARK: - SIGTERM

    /// GUI mode: ignore the default SIGTERM disposition and route the signal
    /// through the same graceful path as any other quit request.
    private func installSIGTERM() {
        guard sigtermSource == nil else {
            return
        }
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler {
            // Never terminate from inside this main-queue block: a
            // `.terminateLater` answer makes AppKit spin a nested run loop
            // until the reply, and the reply arrives as another main-queue
            // block (ShellCore.onMain), which libdispatch does not run while
            // this one is still executing — the app would hang for good. A
            // run-loop block in the common modes runs outside the queue drain.
            let main = CFRunLoopGetMain()
            CFRunLoopPerformBlock(main, CFRunLoopMode.commonModes.rawValue) {
                MainActor.assumeIsolated {
                    NSApp.terminate(nil)
                }
            }
            CFRunLoopWakeUp(main)
        }
        source.resume()
        sigtermSource = source
    }

    // MARK: - Menu

    private func menuAction(_ action: MenuAction) {
        switch action {
        case .toggleOpenAtLogin:
            core?.async { $0.toggleOpenAtLogin() }
        case .checkForUpdates:
            // No window of Sparkle's: the check runs headless and its result
            // shows only in the approved About window.
            updater.checkForUpdates()
            showAbout()
        case .quit:
            guard let core else {
                return
            }
            let generation = forwardQuit()
            core.async { $0.performQuit(acknowledging: generation) }
        default:
            core?.async { $0.perform(action) }
        }
    }

    // MARK: - Views (called by ShellCore on the main thread)

    func render(_ model: MenuBarModel, serverRunning: Bool, status: ShellStatus?, events: [ShellEvent]) {
        self.serverRunning = serverRunning
        latestStatus = status
        statusItem?.render(model)
        for event in events {
            handle(event)
        }
    }

    func showFirstRun(_ model: FirstRunModel) {
        guard firstRun == nil else {
            firstRun?.show()
            return
        }
        let window = FirstRunWindowController(
            model: model,
            onFinish: { [weak self] result in
                guard let self else {
                    return
                }
                self.firstRun = nil
                self.core?.async { $0.finishFirstRun(result) }
                self.applyOpenAtLogin(result.openAtLogin, fromToggle: false)
            },
            onCancel: { [weak self] in
                // Setup was not finished: quit through the normal path.
                self?.firstRun = nil
                NSApp.terminate(nil)
            }
        )
        firstRun = window
        window.show()
    }

    func showDashboard(_ url: URL) {
        dashboard.show(url: url)
    }

    func reloadDashboardIfFailed() {
        dashboard.reloadIfFailed()
    }

    func presentQuitPrompt(_ prompt: QuitPrompt, reply: @escaping (QuitChoice) -> Void) {
        QuitAlertPresenter.present(prompt) { [weak self] choice in
            reply(choice)
            if choice == .cancel, let self {
                self.perform(self.gate.quitCancelled())
            }
        }
    }

    func copyToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    func showLog(_ lines: [String]) {
        logWindow.show(lines: lines)
    }

    func showAbout() {
        if about == nil {
            about = AboutWindowController(
                updater: updater,
                quitDecision: { [weak self] in
                    QuitDecision.decide(serverRunning: self?.serverRunning ?? false, status: self?.latestStatus)
                },
                waitThenRun: { [weak self] work in
                    self?.waitThenRun(work, deadline: Date().addingTimeInterval(ShellCoordinator.defaultQuitWaitLimit))
                },
                confirmQuit: { [weak self] in
                    self?.preconfirmQuit()
                },
                withdrawQuit: { [weak self] in
                    self?.withdrawQuit()
                }
            )
        }
        about?.show()
    }

    /// Keep the login item in step with the Open at Login setting. From the
    /// menu toggle, a registration the user must still approve opens System
    /// Settings > Login Items.
    func applyOpenAtLogin(_ on: Bool, fromToggle: Bool) {
        loginItem.set(on)
        if fromToggle, on, loginItem.needsApproval {
            loginItem.openSystemSettings()
        }
    }

    /// The coordinator stopped the server and wants the process gone.
    func coordinatorDidTerminate() {
        perform(gate.coordinatorTerminated())
    }

    /// The shell queue picked up the quit forward of `generation`.
    func quitAcknowledged(_ generation: Int) {
        gate.quitAcknowledged(generation)
    }

    // MARK: - Helpers

    /// Record a quit forward to the shell queue and arm its watchdog
    /// deadline (unless one already covers it). Returns the generation the
    /// shell queue acknowledges.
    private func forwardQuit() -> Int {
        if let generation = gate.quitForwarded() {
            scheduleQuitDeadline(TerminationGate.unacknowledgedQuitLimit) { [weak self] in
                guard let self else {
                    return
                }
                self.perform(self.gate.acknowledgementDeadlinePassed(generation))
            }
        }
        return gate.forwardGeneration
    }

    /// The user confirmed the quit Install and Relaunch is about to cause:
    /// the next termination request does not prompt again (until it expires).
    private func preconfirmQuit() {
        let generation = gate.preconfirmQuit()
        scheduleQuitDeadline(TerminationGate.preconfirmedQuitLimit) { [weak self] in
            self?.gate.preconfirmationExpired(generation)
        }
    }

    /// The install that pre-confirmed the quit did not start or failed.
    private func withdrawQuit() {
        gate.withdrawPreconfirmation()
    }

    private func perform(_ effect: TerminationGate.Effect) {
        switch effect {
        case .reply(let value):
            NSApp.reply(toApplicationShouldTerminate: value)
        case .terminate:
            NSApp.terminate(nil)
        case .nothing:
            break
        }
    }

    private func handle(_ event: ShellEvent) {
        if let notification = policy.handle(event, now: Date()) {
            notifier.post(notification)
        }
    }

    /// The About window's Wait-Then-Quit: run `work` once the mirrored status
    /// has nothing in flight, or at the deadline.
    private func waitThenRun(_ work: @escaping () -> Void, deadline: Date) {
        if QuitDecision.shouldQuitAfterPoll(status: latestStatus) || Date() >= deadline {
            work()
            return
        }
        let work = QueueConfined(value: work)
        DispatchQueue.main.asyncAfter(deadline: .now() + ShellCoordinator.quitPollInterval) { [weak self] in
            MainActor.assumeIsolated {
                self?.waitThenRun(work.value, deadline: deadline)
            }
        }
    }
}
