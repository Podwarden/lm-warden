// ShellCoordinator — the non-UI brain of the menu bar app (Metal design §2,
// owner-approved mockups 2026-10-20).
//
// It owns the shell's flow: launch (first run or boot), Start/Stop Server,
// the Share-on-LAN toggle with its restart, the menu refresh, and the quit
// flow with its Wait-Then-Quit poll. It decides nothing about copy or layout
// (MenuBarModel, QuitDecision and FirstRunModel do) and touches no window:
// every view effect goes through the injected `ShellViews` closures, and time
// through the injected `schedule` and `now`, so tests drive it with fakes and
// never wait.
//
// Threading: like AppController and ServerSupervisor it is confined to one
// thread (the main thread in the app). It is the single owner of the shell's
// mutable state; the view layer (p7-wire-views) must call it, and hop the
// supervisor's `onStateChange` back, on that thread.
//
// Status feed seam: `status` returns the latest `ShellStatus` or nil. The
// feed is `ShellStatusFeed`, which polls `GET /api/shell/status` with the
// per-launch `VW_SHELL_TOKEN`. The app passes `status: { feed.latest }` and
// calls `statusDidChange()` from the feed's `onChange`, which rebuilds the
// menu from the current state and the latest status and refreshes it once.
// When the feed has no value yet the closure returns nil, so the menu shows
// only the supervisor state and Quit never prompts.
import Foundation

/// The view effects the coordinator drives. Each is a closure so the
/// coordinator stays free of AppKit; the app wires them to real windows.
struct ShellViews {
    var refreshMenu: (MenuBarModel) -> Void
    var presentFirstRun: () -> Void
    var showDashboard: (URL) -> Void
    /// The server reported running: reload the dashboard window if its last
    /// load failed (no-op otherwise; never brings the window forward).
    var reloadDashboardIfFailed: () -> Void = {}
    var presentQuitPrompt: (QuitPrompt, @escaping (QuitChoice) -> Void) -> Void
    var copyToPasteboard: (String) -> Void
    var showLog: ([String]) -> Void
    var showAbout: () -> Void
    var terminate: () -> Void
}

final class ShellCoordinator {
    /// Wait-Then-Quit re-reads the status feed this often.
    static let quitPollInterval: TimeInterval = 2
    /// Wait-Then-Quit gives up waiting after this long and falls through to
    /// Quit Now, so a stuck request (or a feed that never drains) cannot keep
    /// the app alive forever.
    static let defaultQuitWaitLimit: TimeInterval = 120
    /// The port the menu shows before the first boot and while stopped when
    /// no boot has happened yet (PortSelector's preferred port).
    static let defaultPort = 8080

    private let controller: AppController
    private let settings: ShellSettings
    private let views: ShellViews
    private let lanAddress: () -> String?
    private let status: () -> ShellStatus?
    private let canCheckForUpdates: Bool
    private let schedule: (TimeInterval, @escaping () -> Void) -> Void
    private let now: () -> Date
    private let quitWaitLimit: TimeInterval

    /// The UI URL of the current boot; nil while stopped.
    private var dashboardURL: URL?
    /// The port of the last successful boot, kept so the stopped menu still
    /// shows an address.
    private var lastPort: Int?
    /// The log of the last supervisor, kept across Stop so Show log still
    /// has something to show.
    private var lastLogLines: [String] = []

    private enum QuitPhase {
        case idle
        case prompting
        case waiting(deadline: Date)
        case terminated
    }

    private var quitPhase: QuitPhase = .idle
    /// Bumped whenever a quit attempt starts or ends; every prompt reply and
    /// scheduled poll carries the generation it was issued under and is inert
    /// once superseded (the injected schedule has no cancel).
    private var quitGeneration = 0

    init(
        controller: AppController,
        settings: ShellSettings,
        views: ShellViews,
        lanAddress: @escaping () -> String?,
        status: @escaping () -> ShellStatus?,
        canCheckForUpdates: Bool,
        schedule: @escaping (TimeInterval, @escaping () -> Void) -> Void,
        now: @escaping () -> Date = Date.init,
        quitWaitLimit: TimeInterval = ShellCoordinator.defaultQuitWaitLimit
    ) {
        self.controller = controller
        self.settings = settings
        self.views = views
        self.lanAddress = lanAddress
        self.status = status
        self.canCheckForUpdates = canCheckForUpdates
        self.schedule = schedule
        self.now = now
        self.quitWaitLimit = quitWaitLimit
    }

    // MARK: - Launch

    /// First launch shows the first-run window and boots nothing; later
    /// launches boot with the saved settings and refresh the menu.
    func launch() {
        guard settings.firstRunCompleted else {
            views.presentFirstRun()
            return
        }
        bootServer()
        refresh()
    }

    /// Save the four first-run choices, mark first run done, boot, and open
    /// the dashboard (only when the boot succeeded).
    func finishFirstRun(_ result: FirstRunResult) {
        settings.shareOnLAN = result.shareOnLAN
        settings.modelsDir = result.modelsDir
        settings.excludeModelsFromBackup = result.excludeModelsFromBackup
        settings.openAtLogin = result.openAtLogin
        settings.firstRunCompleted = true
        if let url = bootServer() {
            views.showDashboard(url)
        }
        refresh()
    }

    // MARK: - Status feed

    /// The status feed delivered a new value: rebuild the menu from the
    /// current state and the injected `status()` and refresh it once. It
    /// boots nothing, stops nothing and does not touch the quit flow.
    func statusDidChange() {
        refresh()
    }

    // MARK: - Menu actions

    func perform(_ action: MenuAction) {
        switch action {
        case .startServer:
            bootServer()
        case .stopServer:
            stopServer()
        case .toggleShareOnLAN:
            let restart = isActive
            settings.shareOnLAN.toggle()
            if restart {
                stopServer()
                bootServer()
            }
        case .openDashboard:
            if currentState == .running, let url = dashboardURL {
                views.showDashboard(url)
            }
        case .copyAPIAddress:
            views.copyToPasteboard(MenuBarModel.make(currentInput()).copyText)
        case .showLog:
            views.showLog(controller.supervisor?.log.lines ?? lastLogLines)
        case .about:
            views.showAbout()
        case .quit:
            requestQuit()
        case .toggleOpenAtLogin, .checkForUpdates:
            // Wired by the view layer (p7-wire-views), not here.
            break
        }
        refresh()
    }

    // MARK: - Quit

    /// Quit at once when nothing is in flight; otherwise ask. A request while
    /// a prompt is open, a wait is running, or after termination is ignored.
    func requestQuit() {
        guard case .idle = quitPhase else {
            return
        }
        switch QuitDecision.decide(serverRunning: currentState == .running, status: status()) {
        case .quitNow:
            quitNow()
        case .confirm(let prompt):
            quitGeneration += 1
            let generation = quitGeneration
            quitPhase = .prompting
            views.presentQuitPrompt(prompt) { [weak self] choice in
                self?.handleQuitChoice(choice, generation: generation)
            }
        }
    }

    /// Quit without asking: the user already confirmed (About's Install and
    /// Relaunch asked about in-flight requests first). Stops the server and
    /// terminates, exactly once, superseding any prompt or wait.
    func quitConfirmed() {
        quitNow()
    }

    private func handleQuitChoice(_ choice: QuitChoice, generation: Int) {
        guard generation == quitGeneration else {
            return
        }
        switch quitPhase {
        case .prompting, .waiting:
            break
        case .idle, .terminated:
            return
        }
        switch choice {
        case .cancel:
            quitGeneration += 1
            quitPhase = .idle
        case .quitNow:
            quitNow()
        case .waitThenQuit:
            guard case .prompting = quitPhase else {
                return
            }
            quitPhase = .waiting(deadline: now().addingTimeInterval(quitWaitLimit))
            scheduleQuitPoll(generation: generation)
        }
    }

    private func scheduleQuitPoll(generation: Int) {
        schedule(Self.quitPollInterval) { [weak self] in
            self?.pollQuit(generation: generation)
        }
    }

    private func pollQuit(generation: Int) {
        guard generation == quitGeneration, case .waiting(let deadline) = quitPhase else {
            return
        }
        if QuitDecision.shouldQuitAfterPoll(status: status()) || now() >= deadline {
            quitNow()
        } else {
            scheduleQuitPoll(generation: generation)
        }
    }

    /// Stop the server, then terminate — exactly once.
    private func quitNow() {
        if case .terminated = quitPhase {
            return
        }
        quitGeneration += 1
        quitPhase = .terminated
        stopServer()
        views.terminate()
    }

    // MARK: - Server

    /// The supervisor's state, or `.stopped` when there is none.
    private var currentState: SupervisorState {
        controller.supervisor?.state ?? .stopped
    }

    /// True while the server is up or on its way up (the states whose menu
    /// item reads Stop Server).
    private var isActive: Bool {
        switch currentState {
        case .running, .starting, .backingOff:
            return true
        case .stopped, .failed:
            return false
        }
    }

    /// Boot with the saved options and return the UI URL, or nil on failure
    /// (the menu then reads Stopped). A supervisor that has stopped or failed
    /// is discarded first, so Start launches afresh instead of returning the
    /// cached URL of a dead server.
    @discardableResult
    private func bootServer() -> URL? {
        if controller.supervisor != nil, !isActive {
            stopServer()
        }
        let options = BootOptions(shareOnLAN: settings.shareOnLAN, modelsDir: settings.modelsDir)
        do {
            let url = try controller.boot(options: options)
            dashboardURL = url
            lastPort = controller.port
            controller.supervisor?.onStateChange = { [weak self] state in
                if state == .running { self?.views.reloadDashboardIfFailed() }
                self?.refresh()
            }
            return url
        } catch {
            dashboardURL = nil
            return nil
        }
    }

    /// Stop and forget the supervisor. Idempotent.
    private func stopServer() {
        guard let supervisor = controller.supervisor else {
            return
        }
        // Detach first: the stop's own state change (and any late callback
        // from the dead supervisor) must not refresh the menu mid-restart;
        // the caller refreshes once the transition is complete.
        supervisor.onStateChange = nil
        lastLogLines = supervisor.log.lines
        controller.stop()
        dashboardURL = nil
    }

    // MARK: - Menu

    private func currentInput() -> MenuBarInput {
        MenuBarInput(
            state: currentState,
            port: controller.port ?? lastPort ?? Self.defaultPort,
            shareOnLAN: settings.shareOnLAN,
            lanAddress: lanAddress(),
            openAtLogin: settings.openAtLogin,
            canCheckForUpdates: canCheckForUpdates,
            status: status()
        )
    }

    private func refresh() {
        views.refreshMenu(MenuBarModel.make(currentInput()))
    }
}
