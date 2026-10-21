import Foundation
import XCTest
@testable import LMWarden

/// Tests for `ServerSupervisor`, the composition of `SupervisorStateMachine`
/// + health probe + `ProcessLaunching` + `LogRing` (Metal design §2). The
/// supervisor is not thread-safe; the tests call its methods sequentially on
/// one thread, so every injected seam (launcher, probe, scheduler, clock) can
/// be a plain fake that records calls for the test to drive by hand.
final class ServerSupervisorTests: XCTestCase {
    private func makeSpec() -> ServerLaunchSpec {
        ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/usr/bin/env"),
            arguments: ["python"],
            environment: [:]
        )
    }

    /// Build a supervisor over a fake launcher whose processes the test drives
    /// by hand, with a probe always returning `result` and a manual
    /// scheduler/clock.
    private func makeKit(
        result: HealthResult,
        nowValue: TimeInterval,
        onStateChange: ((SupervisorState) -> Void)? = nil
    ) -> (
        supervisor: ServerSupervisor,
        launcher: FakeLauncher,
        scheduler: FakeScheduler,
        clock: FakeClock
    ) {
        let clock = FakeClock(start: nowValue)
        let launcher = FakeLauncher()
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: launcher,
            probe: { completion in completion(result) },
            schedule: scheduler.schedule,
            now: clock.now
        )
        supervisor.onStateChange = onStateChange
        return (supervisor, launcher, scheduler, clock)
    }

    // (b) start() feeds .startRequested: state moves to .starting, the child
    // is launched (launchedAt recorded), and a health poll is scheduled.
    func testStartLaunchesAndPolls() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        XCTAssertEqual(supervisor.state, .starting)
        XCTAssertEqual(launcher.launchCount, 1)
        XCTAssertEqual(launcher.specs, [makeSpec()])
        XCTAssertEqual(scheduler.pairs.count, 1)
        XCTAssertEqual(scheduler.pairs[0].delay, 0.5, accuracy: 1e-9)
    }

    // (b) a healthy poll while .starting moves the supervisor to .running.
    func testHealthyMovesToRunning() {
        let (supervisor, _, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        XCTAssertEqual(supervisor.state, .starting)
        scheduler.runNext()
        XCTAssertEqual(supervisor.state, .running)
    }

    // (b) an unhealthy poll while .starting reschedules the next poll and
    // stays .starting (the child keeps running while we wait for health).
    func testUnhealthyPollsAgain() {
        let (supervisor, _, scheduler, _) = makeKit(result: .unhealthy("status 503"), nowValue: 0)
        supervisor.start()
        XCTAssertEqual(scheduler.pairs.count, 1)
        scheduler.runNext() // first poll -> unhealthy, reschedules
        XCTAssertEqual(supervisor.state, .starting)
        XCTAssertEqual(scheduler.pairs.count, 1)
        XCTAssertEqual(scheduler.pairs[0].delay, 0.5, accuracy: 1e-9)
    }

    // (c) two consecutive short exits back off with the policy's doubling
    // delay: first exit 1 s, second exit 2 s.
    func testExitBacksOffWithPolicyDelay() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        // First short exit (0.5 s uptime) -> backingOff(failures: 1, delay: 1).
        clock.advance(0.5)
        let first = launcher.launchCount
        launcher.exitStatus(first, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(scheduler.pairs.last!.delay, 1, accuracy: 1e-9)
        // The backoff fires and relaunches; the second short exit ->
        // backingOff(failures: 2, delay: 2).
        scheduler.runNext()
        XCTAssertEqual(supervisor.state, .starting)
        let second = launcher.launchCount
        XCTAssertEqual(second, first + 1)
        clock.advance(0.5)
        launcher.exitStatus(second, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 2, delay: 2))
        XCTAssertEqual(scheduler.pairs.last!.delay, 2, accuracy: 1e-9)
    }

    // (c) a backoffElapsed scheduled from .backingOff returns to .starting
    // and launches the child again.
    func testBackoffElapsedRelaunches() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        clock.advance(0.5)
        launcher.exitStatus(1, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(launcher.launchCount, 1)
        scheduler.runNext()
        XCTAssertEqual(supervisor.state, .starting)
        XCTAssertEqual(launcher.launchCount, 2)
    }

    // (c) the sixth consecutive short exit gives up: .failed, and the failed
    // state launches nothing.
    func testGiveUpAfterSixShortExits() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        for _ in 0..<6 {
            clock.advance(0.5)
            launcher.exitStatus(launcher.launchCount, 0)
            if case .failed = supervisor.state { break }
            scheduler.runNext()
        }
        XCTAssertEqual(supervisor.state, .failed(failures: 6))
        XCTAssertEqual(launcher.launchCount, 6)
    }

    // (c) an exit that ran for a full stable-uptime (60 s) starts a fresh
    // streak at failures 1, even right after four short exits.
    func testStableUptimeResetsStreak() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        for _ in 0..<4 {
            clock.advance(0.5)
            launcher.exitStatus(launcher.launchCount, 0)
            scheduler.runNext()
        }
        XCTAssertEqual(supervisor.state, .starting)
        // This process runs long (60 s) then exits: a fresh streak.
        clock.advance(60)
        launcher.exitStatus(launcher.launchCount, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
    }

    // (d) stop() terminates the current process, and the process's later
    // onExit changes nothing: state stays .stopped.
    func testStopTerminatesAndIgnoresLateExit() {
        let (supervisor, launcher, _, clock) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        XCTAssertEqual(launcher.launchCount, 1)
        supervisor.stop()
        XCTAssertEqual(supervisor.state, .stopped)
        let current = launcher.processes.last
        XCTAssertEqual(current?.terminateCount, 1)
        clock.advance(1)
        launcher.exitStatus(1, 0)
        XCTAssertEqual(supervisor.state, .stopped)
    }

    // (d) a backoff closure that was scheduled before stop() launches nothing
    // when it fires after the stop.
    func testBackoffAfterStopLaunchesNothing() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        clock.advance(0.5)
        launcher.exitStatus(1, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(launcher.launchCount, 1)
        supervisor.stop()
        XCTAssertEqual(supervisor.state, .stopped)
        scheduler.runNext() // fires the backoff scheduled before the stop
        XCTAssertEqual(supervisor.state, .stopped)
        XCTAssertEqual(launcher.launchCount, 1)
    }

    // (c) an onExit delivered for an already-replaced process is ignored: the
    // state machine leaves the state unchanged and nothing is launched.
    func testStaleExitIgnored() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        clock.advance(0.5)
        let firstExit = { launcher.exitStatus(1, 0) } // process 1's captured exit
        firstExit() // process 1 exits -> backingOff
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        scheduler.runNext() // relaunches process 2
        XCTAssertEqual(supervisor.state, .starting)
        clock.advance(0.5)
        // Process 2's current exit.
        launcher.exitStatus(2, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 2, delay: 2))
        scheduler.runNext() // relaunches process 3
        XCTAssertEqual(supervisor.state, .starting)
        clock.advance(0.5)
        // Now deliver the stale exit for process 1 (already replaced).
        firstExit()
        // Ignored: the machine is .starting (for process 3), so a non-ignored
        // .exited would have moved it to backingOff.
        XCTAssertEqual(supervisor.state, .starting)
        XCTAssertEqual(launcher.launchCount, 3)
    }

    // (c) a launch() that throws is treated as an exit with uptime 0: it folds
    // into the failure streak and backs off.
    func testLaunchThrowsCountsAsExit() {
        let clock = FakeClock(start: 0)
        let launcher = FakeLauncher()
        launcher.throwsOnLaunch = true
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: launcher,
            probe: { _ in },
            schedule: scheduler.schedule,
            now: clock.now
        )
        supervisor.start()
        // The launch threw, so this is an immediate exit with uptime 0.
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(launcher.specs.count, 1)
        XCTAssertEqual(launcher.processes.count, 0)
    }

    // (b) output chunks from the child are captured into the log ring: a
    // newline-terminated line appears as a line, and a trailing unterminated
    // chunk is held pending until the next chunk completes it.
    func testOutputCapturedInLog() {
        let (supervisor, launcher, _, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        let output = launcher.currentOutput
        output(Data("hello\n".utf8))
        XCTAssertEqual(supervisor.log.lines, ["hello"])
        // A chunk with no trailing newline is pending, not yet a line.
        output(Data("wor".utf8))
        XCTAssertEqual(supervisor.log.lines, ["hello"])
        // The next chunk completes the held line.
        output(Data("ld\n".utf8))
        XCTAssertEqual(supervisor.log.lines, ["hello", "world"])
    }

    // (e) onStateChange fires once per actual state change and not for
    // ignored events (a stale exit and an ignored start).
    func testOnStateChangeOncePerChange() {
        var seen: [SupervisorState] = []
        let (supervisor, launcher, scheduler, clock) = makeKit(
            result: .unhealthy("down"),
            nowValue: 0,
            onStateChange: { seen.append($0) }
        )
        supervisor.start() // -> .starting (1)
        clock.advance(0.5)
        let firstExit = { launcher.exitStatus(1, 0) }
        firstExit() // -> .backingOff(1, 1) (2)
        scheduler.runNext() // -> .starting (3)
        clock.advance(0.5)
        launcher.exitStatus(2, 0) // -> .backingOff(2, 2) (4)
        scheduler.runNext() // -> .starting (5)
        // A stale exit for process 1: ignored, no callback.
        firstExit()
        XCTAssertEqual(seen.count, 5)
        // A stray start while already starting/running must not re-callback.
        supervisor.start()
        XCTAssertEqual(seen.count, 5)
        XCTAssertEqual(seen, [
            .starting,
            .backingOff(failures: 1, delay: 1),
            .starting,
            .backingOff(failures: 2, delay: 2),
            .starting,
        ])
    }

    // (b) the injected healthPollInterval paces both the first poll and the
    // unhealthy re-poll (not a hard-coded 0.5).
    func testHealthPollIntervalHonoured() {
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: FakeLauncher(),
            probe: { completion in completion(.unhealthy("down")) },
            schedule: scheduler.schedule,
            now: FakeClock(start: 0).now,
            healthPollInterval: 2
        )
        supervisor.start()
        XCTAssertEqual(scheduler.pairs.map(\.delay), [2])
        scheduler.runNext()
        XCTAssertEqual(scheduler.pairs.map(\.delay), [2])
    }

    // (c) on a real exit the log is flushed, so the child's final
    // unterminated line is kept.
    func testExitFlushesPendingLine() {
        let (supervisor, launcher, _, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        launcher.currentOutput(Data("Traceback: boom".utf8))
        XCTAssertEqual(supervisor.log.lines, [])
        clock.advance(0.5)
        launcher.exitStatus(1, 1)
        XCTAssertEqual(supervisor.log.lines, ["Traceback: boom"])
    }

    // (d) a poll scheduled before stop() does not survive a stop/start as a
    // second poll chain: firing it after the restart probes nothing.
    func testStalePollAfterStopStartProbesNothing() {
        var probes = 0
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: FakeLauncher(),
            probe: { completion in probes += 1; completion(.unhealthy("down")) },
            schedule: scheduler.schedule,
            now: FakeClock(start: 0).now
        )
        supervisor.start() // poll A
        supervisor.stop()  // schedules grace (15 s)
        supervisor.start() // poll B
        // poll A + grace + poll B
        XCTAssertEqual(scheduler.pairs.count, 3)
        scheduler.run(at: 0) // stale poll A
        XCTAssertEqual(probes, 0)
        XCTAssertEqual(scheduler.pairs.count, 2)
        scheduler.runNext() // poll B (last pair); probes unhealthy, reschedules
        XCTAssertEqual(probes, 1)
        // grace + poll C (rescheduled by poll B)
        XCTAssertEqual(scheduler.pairs.count, 2)
    }

    // (d) a backoff scheduled before stop() that fires after a stop/start and
    // a fresh exit does not cut the new backoff short.
    func testStaleBackoffAfterStopStartIgnored() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .unhealthy("down"), nowValue: 0)
        supervisor.start()
        clock.advance(0.5)
        launcher.exitStatus(1, 0) // backoff B1 pending
        let staleIndex = scheduler.pairs.count - 1
        supervisor.stop()
        supervisor.start() // process 2
        clock.advance(0.5)
        launcher.exitStatus(2, 0)
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        scheduler.run(at: staleIndex) // stale B1
        XCTAssertEqual(supervisor.state, .backingOff(failures: 1, delay: 1))
        XCTAssertEqual(launcher.launchCount, 2)
    }

    // (b) a probe answer that arrives after its process was replaced does not
    // mark the new process healthy.
    func testStaleProbeResultIgnored() {
        var pending: [(HealthResult) -> Void] = []
        let clock = FakeClock(start: 0)
        let launcher = FakeLauncher()
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: launcher,
            probe: { pending.append($0) },
            schedule: scheduler.schedule,
            now: clock.now
        )
        supervisor.start()
        scheduler.runNext() // probe for process 1 in flight
        XCTAssertEqual(pending.count, 1)
        clock.advance(0.5)
        launcher.exitStatus(1, 0)
        scheduler.runNext() // backoff -> process 2
        XCTAssertEqual(supervisor.state, .starting)
        pending[0](.healthy) // late answer about process 1
        XCTAssertEqual(supervisor.state, .starting)
    }

    // (d) stop() schedules a hard kill on the default grace (15 s); running
    // the closure kills the process.
    func testStopSchedulesHardKillAfterGrace() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        supervisor.stop()
        XCTAssertEqual(launcher.processes.last?.terminateCount, 1)
        // stop() schedules one extra closure at the default grace (15 s).
        XCTAssertEqual(scheduler.pairs.count, 2)
        XCTAssertEqual(ServerSupervisor.defaultStopGracePeriod, 15)
        XCTAssertEqual(scheduler.pairs[scheduler.pairs.count - 1].delay, 15, accuracy: 1e-9)
        // Running the grace closure kills the process.
        scheduler.runNext()
        XCTAssertEqual(launcher.processes.last?.killCount, 1)
    }

    // (a) a custom stopGracePeriod is honoured: the grace closure is
    // scheduled at the custom delay, not the default 15 s.
    func testCustomStopGraceHonoured() {
        let clock = FakeClock(start: 0)
        let launcher = FakeLauncher()
        let scheduler = FakeScheduler()
        let supervisor = ServerSupervisor(
            spec: makeSpec(),
            launcher: launcher,
            probe: { completion in completion(.healthy) },
            schedule: scheduler.schedule,
            now: clock.now,
            stopGracePeriod: 3
        )
        supervisor.start()
        supervisor.stop()
        // One poll at 0.5 (from start) plus one grace closure at 3.
        XCTAssertEqual(scheduler.pairs.count, 2)
        XCTAssertEqual(scheduler.pairs[scheduler.pairs.count - 1].delay, 3, accuracy: 1e-9)
        scheduler.runNext()
        XCTAssertEqual(launcher.processes.last?.killCount, 1)
    }

    // (b) if the process exits during the grace window, the pending kill
    // closure does nothing: the process is already gone.
    func testHardKillSkippedWhenExitedDuringGrace() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        supervisor.stop()
        // The process exits cleanly during the grace window.
        launcher.exitStatus(1, 0)
        // The grace closure fires but must not kill (already exited).
        scheduler.runNext()
        XCTAssertEqual(launcher.processes.last?.killCount, 0)
    }

    /// Start and stop a supervisor whose ONLY strong reference dies inside
    /// this helper (as AppController.stop() drops its supervisor), and prove
    /// it is gone. The scheduler then holds [poll, grace]; the grace is last.
    private func makeReleasedStoppedSupervisor() -> (launcher: FakeLauncher, scheduler: FakeScheduler) {
        let launcher = FakeLauncher()
        let scheduler = FakeScheduler()
        weak var weakSupervisor: ServerSupervisor?
        do {
            let supervisor = ServerSupervisor(
                spec: makeSpec(),
                launcher: launcher,
                probe: { completion in completion(.healthy) },
                schedule: scheduler.schedule,
                now: FakeClock(start: 0).now
            )
            supervisor.start()
            supervisor.stop()
            weakSupervisor = supervisor
        }
        XCTAssertNil(weakSupervisor, "the supervisor must really be released")
        XCTAssertEqual(scheduler.pairs.count, 2)
        return (launcher, scheduler)
    }

    // (b) the grace closure captures the process strongly, so it still kills
    // even after the supervisor itself has been released (AppController.stop()
    // drops its supervisor reference).
    func testHardKillFiresAfterSupervisorReleased() {
        let (launcher, scheduler) = makeReleasedStoppedSupervisor()
        scheduler.runNext()
        XCTAssertEqual(launcher.processes[0].killCount, 1)
    }

    // (b) a process that exits during the grace window after its supervisor
    // was released is NOT killed: the exit is recorded without the supervisor.
    func testHardKillSkippedWhenExitedAfterSupervisorReleased() {
        let (launcher, scheduler) = makeReleasedStoppedSupervisor()
        launcher.exitStatus(1, 0)
        scheduler.runNext()
        XCTAssertEqual(launcher.processes[0].killCount, 0)
    }

    // (d) the old process exits after a stop/start: its pending grace closure
    // must not SIGKILL it (its pid may already be reused).
    func testHardKillSkippedWhenOldProcessExitsAfterRestart() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        supervisor.stop() // grace A at index 1
        supervisor.start()
        launcher.exitStatus(1, 0)
        scheduler.run(at: 1) // grace A
        XCTAssertEqual(launcher.processes[0].killCount, 0)
        XCTAssertEqual(launcher.processes[1].killCount, 0)
    }

    // (d) a late exit of the OLD process does not mark the NEW one as exited:
    // the new process's own grace still kills it.
    func testLateOldExitDoesNotSpareNewProcess() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        supervisor.stop() // grace A
        supervisor.start()
        supervisor.stop() // grace B (last)
        XCTAssertEqual(launcher.processes[1].terminateCount, 1)
        launcher.exitStatus(1, 0) // A exits late
        scheduler.runNext() // grace B
        XCTAssertEqual(launcher.processes[1].killCount, 1)
        XCTAssertEqual(launcher.processes[0].killCount, 0)
    }

    // (d) after stop() then start(), the pending grace closure kills only the
    // OLD process (if it has not exited), never the new one.
    func testHardKillTargetsOldProcessAfterRestart() {
        let (supervisor, launcher, scheduler, clock) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        let oldProcess = launcher.processes[0]
        supervisor.stop()
        // The grace closure is the second pair scheduled.
        let graceIndex = scheduler.pairs.count - 1
        supervisor.start()
        let newProcess = launcher.processes[1]
        // Fire the stale grace closure: it kills the old process, not the new one.
        scheduler.run(at: graceIndex)
        XCTAssertEqual(oldProcess.killCount, 1)
        XCTAssertEqual(newProcess.killCount, 0)
    }

    // (c) stop() without a current process (never started) schedules nothing
    // and kills nothing.
    func testStopWithoutProcessSchedulesNothing() {
        let (supervisor, _, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.stop()
        XCTAssertEqual(scheduler.pairs.count, 0)
    }

    // (c) stop() after a crash cleared the process (backing off) schedules
    // nothing and signals nothing.
    func testStopAfterBackoffClearedProcessSchedulesNothing() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        launcher.exitStatus(1, 1) // crash -> backing off, process cleared
        let scheduled = scheduler.pairs.count
        supervisor.stop()
        XCTAssertEqual(scheduler.pairs.count, scheduled)
        XCTAssertEqual(launcher.processes[0].terminateCount, 0)
        XCTAssertEqual(launcher.processes[0].killCount, 0)
    }

    // (c) a second stop() while the first grace is pending sends no second
    // SIGTERM and schedules no second kill.
    func testSecondStopDoesNotDoubleSignal() {
        let (supervisor, launcher, scheduler, _) = makeKit(result: .healthy, nowValue: 0)
        supervisor.start()
        supervisor.stop()
        supervisor.stop()
        // Only one SIGTERM (from the first stop) and one grace closure.
        XCTAssertEqual(launcher.processes.last?.terminateCount, 1)
        XCTAssertEqual(scheduler.pairs.count, 2)
    }

    // MARK: - Fakes

    /// Records launches and exposes the captured closures. `throwsOnLaunch`
    /// makes `launch` throw (and return no process) to exercise the
    /// launch-failure path.
    private final class FakeLauncher: ProcessLaunching {
        var throwsOnLaunch = false
        private(set) var specs: [ServerLaunchSpec] = []
        private(set) var processes: [FakeProcess] = []

        var launchCount: Int { processes.count }
        var currentOutput: (Data) -> Void { processes.last?.onOutput ?? { _ in } }

        /// Invoke the onExit closure of the process launched as `number`
        /// (1-based) with `status`.
        func exitStatus(_ number: Int, _ status: Int32) {
            processes[number - 1].invokeExit(status)
        }

        func launch(
            _ spec: ServerLaunchSpec,
            onOutput: @escaping (Data) -> Void,
            onExit: @escaping (Int32) -> Void
        ) throws -> LaunchedProcess {
            specs.append(spec)
            guard !throwsOnLaunch else {
                throw LaunchSpecError.emptyPythonPath
            }
            let process = FakeProcess(onOutput: onOutput, onExit: onExit)
            processes.append(process)
            return process
        }
    }

    /// A `LaunchedProcess` the test can drive by hand: `terminate()` and
    /// `kill()` are recorded and the captured `onExit` can be invoked with
    /// any status.
    private final class FakeProcess: LaunchedProcess {
        let pid: Int32
        let onOutput: (Data) -> Void
        private let onExit: (Int32) -> Void
        private(set) var terminateCount: Int = 0
        private(set) var killCount: Int = 0

        init(pid: Int32 = 1000, onOutput: @escaping (Data) -> Void, onExit: @escaping (Int32) -> Void) {
            self.pid = pid
            self.onOutput = onOutput
            self.onExit = onExit
        }

        var invokeExit: (Int32) -> Void { onExit }

        func terminate() {
            terminateCount += 1
        }

        func kill() {
            killCount += 1
        }
    }

    /// Stores `(delay, closure)` pairs in the order they were scheduled.
    /// `runNext()` runs the most recently scheduled pending closure: the test
    /// schedules exactly one closure per action it drives (a poll at launch, a
    /// backoff at exit), so the newest one is the one it wants to fire. A
    /// stale earlier poll, if it ever fires, is a no-op (the supervisor guards
    /// `state == .starting` before probing).
    private final class FakeScheduler {
        struct Pair {
            let delay: TimeInterval
            let closure: () -> Void
        }
        private(set) var pairs: [Pair] = []

        func schedule(_ delay: TimeInterval, _ closure: @escaping () -> Void) {
            pairs.append(Pair(delay: delay, closure: closure))
        }

        /// Run the closure scheduled `index`-th (0-based, in schedule order)
        /// out of order, to fire a stale closure after newer ones.
        func run(at index: Int) {
            let pair = pairs.remove(at: index)
            pair.closure()
        }

        func runNext() {
            guard !pairs.isEmpty else { return }
            let pair = pairs.removeLast()
            pair.closure()
        }
    }

    /// A mutable clock the test advances by hand.
    private final class FakeClock {
        private(set) var value: TimeInterval
        init(start: TimeInterval) {
            value = start
        }
        func advance(_ delta: TimeInterval) {
            value += delta
        }
        var now: () -> Date {
            { [weak self] in Date(timeIntervalSince1970: self?.value ?? 0) }
        }
    }
}
