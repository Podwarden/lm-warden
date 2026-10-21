import XCTest
@testable import LMWarden

final class ProcessLauncherTests: XCTestCase {
    /// /bin/sh -c 'echo hi; echo err 1>&2; exit 3' produces stdout "hi" and
    /// stderr "err", then exits with status 3. Both streams share one pipe,
    /// so the concatenated output must contain both strings and onExit must
    /// be called with 3.
    func testOutputAndExitStatus() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo hi; echo err 1>&2; exit 3"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let collected = OutputCollector()
        collected.expect(["hi", "err"], expectation(description: "output seen"))
        let exitExpectation = expectation(description: "onExit called")
        exitExpectation.assertForOverFulfill = false

        do {
            _ = try launcher.launch(spec, onOutput: { collected.append($0) }, onExit: { status in
                XCTAssertEqual(status, 3)
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }
        waitForExpectations(timeout: 5)
        let text = collected.text
        XCTAssertTrue(text.contains("hi"), "output missing 'hi': \(text)")
        XCTAssertTrue(text.contains("err"), "output missing 'err': \(text)")
    }

    /// The spec's environment is passed to the child: LMW_T=x is visible in
    /// the child's environment and echoed to stdout.
    func testEnvironmentPassed() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo \"$LMW_T[$HOME]\""],
            environment: ["LMW_T": "x"]
        )
        let launcher = SystemProcessLauncher()
        let collected = OutputCollector()
        collected.expect(["\n"], expectation(description: "output seen"))
        let exitExpectation = expectation(description: "exit")
        exitExpectation.assertForOverFulfill = false

        do {
            _ = try launcher.launch(spec, onOutput: { collected.append($0) }, onExit: { _ in
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }
        waitForExpectations(timeout: 5)
        let text = collected.text
        XCTAssertTrue(text.contains("x"), "output should contain 'x', got: \(text)")
        // The child sees exactly the spec's environment: nothing inherited
        // from the parent (the test runner always has HOME set).
        XCTAssertTrue(text.contains("x[]"), "parent environment leaked into the child: \(text)")
    }

    /// /bin/sleep 30 is terminated with SIGTERM (terminate()) and onExit fires
    /// within 5 s (it would not exit on its own for 30 s); a SIGKILL cleanup
    /// runs whenever no exit was reaped.
    func testTerminateStopsSleep() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sleep"),
            arguments: ["30"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let exitStatus = StatusBox()
        let exitExpectation = expectation(description: "exit after terminate")
        exitExpectation.assertForOverFulfill = false

        let process: LaunchedProcess
        do {
            process = try launcher.launch(spec, onOutput: { _ in }, onExit: { status in
                exitStatus.set(status)
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }
        let pid = process.pid
        // SIGKILL fallback only while no exit has been reaped: once onExit
        // has a status the pid is free and may already name someone else.
        defer {
            if exitStatus.value == -1 {
                Darwin.killpg(pid, SIGKILL)
                Darwin.kill(pid, SIGKILL)
                exitStatus.waitForExit(timeout: 5)
            }
        }
        // Give the process a moment to actually start sleeping.
        Thread.sleep(forTimeInterval: 0.3)
        process.terminate()
        let result = XCTWaiter.wait(for: [exitExpectation], timeout: 5)
        XCTAssertEqual(result, .completed, "child did not exit after terminate()")
        if result != .completed {
            process.kill()
            exitStatus.waitForExit(timeout: 5)
            return
        }
        XCTAssertEqual(exitStatus.value, SIGTERM)
    }

    /// A sleep that ignores SIGTERM (the ignored disposition survives exec) is
    /// hard-killed with SIGKILL (kill()) and onExit fires within 5 s with
    /// status SIGKILL; a kill() that sent SIGTERM would leave it sleeping.
    func testKillStopsSleep() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap '' TERM; exec /bin/sleep 30"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let exitExpectation = expectation(description: "exit after kill")
        exitExpectation.assertForOverFulfill = false

        let process: LaunchedProcess
        do {
            process = try launcher.launch(spec, onOutput: { _ in }, onExit: { status in
                XCTAssertEqual(status, SIGKILL)
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }
        Thread.sleep(forTimeInterval: 0.3)
        process.kill()
        waitForExpectations(timeout: 5)
    }

    /// A spec whose executable does not exists must make launch throw (the
    /// Process.run() error is rethrown), not silently succeed.
    func testMissingExecutableThrows() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/nonexistent/bin"),
            arguments: [],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        XCTAssertThrowsError(
            try launcher.launch(spec, onOutput: { _ in }, onExit: { _ in })
        )
    }

    /// The child is spawned as the leader of a new process group (posix_spawn
    /// with POSIX_SPAWN_SETPGROUP): getpgid(child) == the child's pid and, for
    /// a child that outlives launch(), != the test runner's own process group
    /// (getpgrp()). On the base, Foundation `Process` leaves the child in the
    /// parent's group, so both legs fail here.
    func testChildLeadsOwnProcessGroup() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap '' TERM; /bin/sleep 30"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let process: LaunchedProcess
        do {
            process = try launcher.launch(spec, onOutput: { _ in }, onExit: { _ in })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }
        defer { process.kill() }
        Thread.sleep(forTimeInterval: 0.3)
        let pid = process.pid
        XCTAssertEqual(Darwin.getpgid(pid), pid, "child must lead its own group")
        XCTAssertNotEqual(Darwin.getpgid(pid), Darwin.getpgrp(), "child group must differ from the test runner's")
    }

    /// kill() signals the WHOLE process group (killpg): a grandchild in the
    /// child's group that ignores SIGTERM is still hard-killed. The shell
    /// echoes the grandchild's pid to stdout; after kill() that pid must be
    /// gone (kill(pid, 0) == -1, errno == ESRCH).
    func testKillReachesGrandchild() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap '' TERM; /bin/sleep 60 & echo $!; wait"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let collected = OutputCollector()
        let exitExpectation = expectation(description: "exit")
        exitExpectation.assertForOverFulfill = false

        let process: LaunchedProcess
        do {
            process = try launcher.launch(spec, onOutput: { collected.append($0) }, onExit: { _ in
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }

        let grandchild = waitForChildPid(collected)
        guard grandchild != -1 else {
            process.kill()
            waitForExpectations(timeout: 5)
            XCTFail("could not read the grandchild pid from output: \(collected.text)")
            return
        }
        // Safety: if this assertion is red on the base, kill the grandchild so
        // no stray `/bin/sleep 60` survives the failed run.
        defer { Darwin.kill(grandchild, SIGKILL) }
        process.kill()
        let gone = waitForGrandchildGone(grandchild)
        XCTAssertTrue(gone, "grandchild \(grandchild) still alive after kill()")
        waitForExpectations(timeout: 5)
    }

    /// terminate() signals the WHOLE process group (killpg, SIGTERM): a
    /// grandchild with the default TERM disposition dies to the group signal.
    func testTerminateReachesGrandchild() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "/bin/sleep 60 & echo $!; wait"],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        let collected = OutputCollector()
        let exitExpectation = expectation(description: "exit")
        exitExpectation.assertForOverFulfill = false

        let process: LaunchedProcess
        do {
            process = try launcher.launch(spec, onOutput: { collected.append($0) }, onExit: { _ in
                exitExpectation.fulfill()
            })
        } catch {
            XCTFail("launch failed: \(error)")
            return
        }

        let grandchild = waitForChildPid(collected)
        guard grandchild != -1 else {
            process.kill()
            waitForExpectations(timeout: 5)
            XCTFail("could not read the grandchild pid from output: \(collected.text)")
            return
        }
        defer { Darwin.kill(grandchild, SIGKILL) }
        process.terminate()
        let gone = waitForGrandchildGone(grandchild)
        XCTAssertTrue(gone, "grandchild \(grandchild) still alive after terminate()")
        waitForExpectations(timeout: 5)
    }

    /// The child must start with an EMPTY signal mask even when launch() runs
    /// on a thread that has SIGTERM blocked (dispatch worker threads often
    /// do). posix_spawn copies the calling thread's mask into the child unless
    /// POSIX_SPAWN_SETSIGMASK is set, and a blocked SIGTERM survives exec, so
    /// without it terminate()'s killpg(SIGTERM) never arrives and uvicorn
    /// outlives the app. Here the child must die to terminate() within 2 s
    /// with status SIGTERM; a SIGKILL cleanup runs whatever happens.
    func testChildUnblocksSigtermSpawnedFromBlockedThread() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sleep"),
            arguments: ["30"],
            environment: [:]
        )
        let exitStatus = StatusBox()
        let exited = expectation(description: "exit after terminate")
        exited.assertForOverFulfill = false
        let process = launchWithBlockedSignals([SIGTERM, SIGINT, SIGHUP], spec: spec) { status in
            exitStatus.set(status)
            exited.fulfill()
        }
        guard let process else {
            XCTFail("launch from the blocked thread failed")
            return
        }
        let pid = process.pid
        // SIGKILL fallback only while no exit has been reaped: once onExit
        // has a status the pid is free and may already name someone else.
        defer {
            if exitStatus.value == -1 {
                Darwin.killpg(pid, SIGKILL)
                Darwin.kill(pid, SIGKILL)
                exitStatus.waitForExit(timeout: 5)
            }
        }
        Thread.sleep(forTimeInterval: 0.3)
        process.terminate()
        let result = XCTWaiter.wait(for: [exited], timeout: 2)
        XCTAssertEqual(result, .completed, "child ignored SIGTERM: it inherited the spawning thread's blocked mask")
        if result != .completed {
            process.kill()
            exitStatus.waitForExit(timeout: 5)
            return
        }
        XCTAssertEqual(exitStatus.value, SIGTERM)
    }

    /// The child must start with the DEFAULT disposition for SIGHUP even when
    /// the parent ignores it (an ignored disposition survives exec unless
    /// POSIX_SPAWN_SETSIGDEF resets it). The parent ignores SIGHUP only for
    /// the launch; then killpg(SIGHUP) must end the child within 2 s.
    func testChildResetsIgnoredSighupToDefault() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sleep"),
            arguments: ["30"],
            environment: [:]
        )
        let exitStatus = StatusBox()
        let exited = expectation(description: "exit after SIGHUP")
        exited.assertForOverFulfill = false
        let previous = Darwin.signal(SIGHUP, SIG_IGN)
        let process: LaunchedProcess?
        do {
            process = try SystemProcessLauncher().launch(spec, onOutput: { _ in }, onExit: { status in
                exitStatus.set(status)
                exited.fulfill()
            })
        } catch {
            process = nil
        }
        Darwin.signal(SIGHUP, previous)
        guard let process else {
            XCTFail("launch failed")
            return
        }
        let pid = process.pid
        // SIGKILL fallback only while no exit has been reaped: once onExit
        // has a status the pid is free and may already name someone else.
        defer {
            if exitStatus.value == -1 {
                Darwin.killpg(pid, SIGKILL)
                Darwin.kill(pid, SIGKILL)
                exitStatus.waitForExit(timeout: 5)
            }
        }
        Thread.sleep(forTimeInterval: 0.3)
        Darwin.killpg(pid, SIGHUP)
        let result = XCTWaiter.wait(for: [exited], timeout: 2)
        XCTAssertEqual(result, .completed, "child kept the parent's ignored SIGHUP disposition")
        if result != .completed {
            process.kill()
            exitStatus.waitForExit(timeout: 5)
            return
        }
        XCTAssertEqual(exitStatus.value, SIGHUP)
    }

    /// Runs launch() on a fresh thread whose signal mask blocks `signals`,
    /// joins it, and returns the launched process (nil if launch threw).
    private func launchWithBlockedSignals(
        _ signals: [Int32],
        spec: ServerLaunchSpec,
        onExit: @escaping (Int32) -> Void
    ) -> LaunchedProcess? {
        let box = ProcessBox()
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            var set = sigset_t()
            sigemptyset(&set)
            for signal in signals {
                sigaddset(&set, signal)
            }
            var old = sigset_t()
            guard pthread_sigmask(SIG_BLOCK, &set, &old) == 0 else {
                done.signal()
                return
            }
            box.set(try? SystemProcessLauncher().launch(spec, onOutput: { _ in }, onExit: onExit))
            pthread_sigmask(SIG_SETMASK, &old, nil)
            done.signal()
        }
        thread.start()
        done.wait()
        return box.value
    }

    /// A missing executable makes launch throw `ProcessLaunchError.spawnFailed`
    /// carrying the errno from posix_spawn (ENOENT).
    func testSpawnFailedCarriesErrno() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/nonexistent/bin"),
            arguments: [],
            environment: [:]
        )
        let launcher = SystemProcessLauncher()
        do {
            _ = try launcher.launch(spec, onOutput: { _ in }, onExit: { _ in })
            XCTFail("launch should have thrown")
        } catch let error as ProcessLaunchError {
            guard case .spawnFailed(let errno) = error else {
                XCTFail("expected .spawnFailed, got \(error)")
                return
            }
            XCTAssertEqual(errno, ENOENT)
        } catch {
            XCTFail("expected ProcessLaunchError, got \(error)")
        }
    }

    /// The parent closes its copy of the pipe's write end after the spawn, so
    /// the reader sees EOF when the child exits and, once the caller drops the
    /// LaunchedProcess, no fd of the output pipe stays open in the parent.
    /// (Holding the write end would keep the readabilityHandler waiting forever
    /// and leak both pipe fds per launch.) A warm-up launch first, so lazily
    /// created dispatch fds are counted in the baseline.
    func testPipeFdsReleasedAfterExit() {
        runEchoAndDrop()
        let baseline = openFdCount()
        runEchoAndDrop()
        let deadline = Date().addingTimeInterval(5)
        while openFdCount() > baseline, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertEqual(openFdCount(), baseline, "pipe fds leaked in the parent after the child exited")
    }

    private func runEchoAndDrop() {
        let spec = ServerLaunchSpec(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo done"],
            environment: [:]
        )
        let collected = OutputCollector()
        collected.expect(["done"], expectation(description: "output"))
        let exitExpectation = expectation(description: "exit")
        exitExpectation.assertForOverFulfill = false
        do {
            var process: LaunchedProcess? = try SystemProcessLauncher().launch(
                spec, onOutput: { collected.append($0) }, onExit: { _ in exitExpectation.fulfill() }
            )
            waitForExpectations(timeout: 5)
            XCTAssertNotNil(process)
            process = nil
        } catch {
            XCTFail("launch failed: \(error)")
        }
    }

    private func openFdCount() -> Int {
        var count = 0
        for fd in 0..<Int32(1024) where fcntl(fd, F_GETFD) != -1 {
            count += 1
        }
        return count
    }

    /// Polls the collector until a child pid line has arrived, or gives up
    /// after 5 s (returns -1).
    private func waitForChildPid(_ collected: OutputCollector) -> Int32 {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let pid = parseChildPid(collected.text) {
                return pid
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        return -1
    }

    /// The grandchild pid is the first integer on its own line (the `echo $!`
    /// output).
    private func parseChildPid(_ text: String) -> Int32? {
        for line in text.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let pid = Int32(trimmed), pid > 0 {
                return pid
            }
        }
        return nil
    }

    /// Polls (Thread.sleep 0.05, max 5 s) until the grandchild is gone:
    /// kill(pid, 0) == -1 with errno == ESRCH. No timers.
    private func waitForGrandchildGone(_ pid: Int32) -> Bool {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if Darwin.kill(pid, 0) == -1, errno == ESRCH {
                return true
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        return Darwin.kill(pid, 0) == -1 && errno == ESRCH
    }

    // MARK: - Thread-safe collectors

    /// Holds an exit status written from the process-source queue.
    private final class StatusBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Int32 = -1

        func set(_ status: Int32) {
            lock.withLock { stored = status }
        }

        var value: Int32 {
            lock.withLock { stored }
        }

        /// Polls (0.05 s) until a status has been recorded or `timeout` passes.
        func waitForExit(timeout: TimeInterval) {
            let deadline = Date().addingTimeInterval(timeout)
            while value == -1, Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
    }

    /// Hands the launched process from the launching thread to the test.
    private final class ProcessBox: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: LaunchedProcess?

        func set(_ process: LaunchedProcess?) {
            lock.withLock { stored = process }
        }

        var value: LaunchedProcess? {
            lock.withLock { stored }
        }
    }

    /// Accumulates output chunks from the (concurrent) readabilityHandler.
    private final class OutputCollector: @unchecked Sendable {
        private let queue = DispatchQueue(label: "output.collector")
        private var buffer = Data()
        private var needles: [String] = []
        private var seen: XCTestExpectation?

        /// Fulfils `exp` once the accumulated output contains every needle:
        /// onExit can run before the readabilityHandler delivers the last chunk.
        func expect(_ needles: [String], _ exp: XCTestExpectation) {
            queue.sync { self.needles = needles; seen = exp }
        }

        func append(_ chunk: Data) {
            queue.sync {
                buffer.append(chunk)
                let text = String(data: buffer, encoding: .utf8) ?? ""
                if let exp = seen, needles.allSatisfy({ text.contains($0) }) {
                    exp.fulfill()
                    seen = nil
                }
            }
        }

        var text: String {
            queue.sync { String(data: buffer, encoding: .utf8) ?? "" }
        }
    }
}
