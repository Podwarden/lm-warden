// ProcessLauncher — the seam between ServerSupervisor and the spawned child.
// (Metal design s2, non-UI: the supervisor owns one child.)
//
// s2 hazard: a Swift child is NOT killed when the parent dies. A force-quit
// would leave a headless uvicorn holding ~12 GB. Two belts cover it: the
// `VW_PARENT_PID` env var (already injected by `ServerLaunchSpec.make`,
// polled by the FastAPI lifespan) and — this task — spawning the child as the
// leader of its OWN process group so the whole group (child plus any
// grandchildren, e.g. uvicorn re-execs) can be killed on exit with one
// killpg.
//
// Quit semantics (design s2): Stop Server sends SIGTERM to the group via
// `terminate()`; ⌘Q does a graceful SIGTERM with a hard kill after ~15 s via
// `kill()`.
import Darwin
import Foundation

/// A launched child process: its pid plus the two signals the supervisor may
/// send. `terminate()` is the graceful stop (SIGTERM); `kill()` is the hard
/// stop (SIGKILL). Both signal the child's whole process group.
protocol LaunchedProcess: AnyObject {
    var pid: Int32 { get }
    func terminate()
    func kill()
}

/// Launches one child from a `ServerLaunchSpec`, streaming its merged
/// stdout/stderr and reporting its exit status through closures. A protocol
/// so the supervisor can be tested without spawning real processes.
protocol ProcessLaunching {
    func launch(
        _ spec: ServerLaunchSpec,
        onOutput: @escaping (Data) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws -> LaunchedProcess
}

/// A real-process spawn failure carrying the `errno` posix_spawn returned.
enum ProcessLaunchError: Error, Equatable {
    case spawnFailed(errno: Int32)
}

/// A `LaunchedProcess` backed by a real child spawned with posix_spawn.
///
/// `processSource` and `stdoutPipe` are kept for the child's lifetime so the
/// exit observer and the pipe's readabilityHandler closure cannot be
/// deallocated while the child is still running.
final class SystemLaunchedProcess: LaunchedProcess {
    private let pidValue: Int32
    private let processSource: any DispatchSourceProcess
    private let stdoutPipe: Pipe

    init(pid: Int32, processSource: any DispatchSourceProcess, stdoutPipe: Pipe) {
        self.pidValue = pid
        self.processSource = processSource
        self.stdoutPipe = stdoutPipe
    }

    var pid: Int32 {
        pidValue
    }

    func terminate() {
        Darwin.killpg(pidValue, SIGTERM)
    }

    func kill() {
        Darwin.killpg(pidValue, SIGKILL)
    }
}

/// Launches real processes with posix_spawn.
final class SystemProcessLauncher: ProcessLaunching {
    func launch(
        _ spec: ServerLaunchSpec,
        onOutput: @escaping (Data) -> Void,
        onExit: @escaping (Int32) -> Void
    ) throws -> LaunchedProcess {
        // One pipe carries both stdout and stderr so the caller sees a single
        // merged stream; chunks arrive in (nondeterministic) order.
        let pipe = Pipe()
        let writeFD = pipe.fileHandleForWriting.fileDescriptor

        let executable = spec.executable.path
        let arguments = spec.arguments
        let environment = spec.environment

        // argv[0] is the executable path, then the spec's arguments, NUL-terminated.
        var argv: [UnsafeMutablePointer<CChar>?] = []
        argv.reserveCapacity(arguments.count + 2)
        let executableArg = strdup(executable)
        guard executableArg != nil else { throw ProcessLaunchError.spawnFailed(errno: ENOMEM) }
        argv.append(executableArg)
        for argument in arguments {
            let a = strdup(argument)
            guard a != nil else {
                freeArgArray(&argv)
                throw ProcessLaunchError.spawnFailed(errno: ENOMEM)
            }
            argv.append(a)
        }
        argv.append(nil)

        // envp is exactly spec.environment as KEY=value, NUL-terminated.
        var envp: [UnsafeMutablePointer<CChar>?] = []
        envp.reserveCapacity(environment.count + 1)
        for entry in environment {
            let line = strdup("\(entry.key)=\(entry.value)")
            guard line != nil else {
                freeArgArray(&argv)
                freeArgArray(&envp)
                throw ProcessLaunchError.spawnFailed(errno: ENOMEM)
            }
            envp.append(line)
        }
        envp.append(nil)

        // Optional so &attr is the Optional-pointee pointer form the Darwin C
        // API expects (posix_spawnattr_t is an opaque void*).
        var attr: posix_spawnattr_t?
        var actions: posix_spawn_file_actions_t?

        var result: Int32 = 0
        var pid: pid_t = 0
        withUnsafeMutablePointer(to: &pid) { pidPtr in
            argv.withUnsafeMutableBufferPointer { argvBuf in
                envp.withUnsafeMutableBufferPointer { envpBuf in
                    result = spawnChild(
                        pidPtr: pidPtr,
                        executable: executable,
                        argvPtr: argvBuf.baseAddress!,
                        envpPtr: envpBuf.baseAddress!,
                        attr: &attr,
                        actions: &actions,
                        writeFD: writeFD
                    )
                }
            }
        }
        // The strdups are copied into the child by posix_spawn; free them now.
        freeArgArray(&argv)
        freeArgArray(&envp)
        _ = Darwin.posix_spawnattr_destroy(&attr)
        _ = Darwin.posix_spawn_file_actions_destroy(&actions)

        if result != 0 {
            throw ProcessLaunchError.spawnFailed(errno: result)
        }

        // The child has its own copy of the write end on fds 1/2; close the
        // parent's so the reader sees EOF when the child (and every holder of
        // the pipe in its group) exits. Keeping it open would leave the
        // readabilityHandler (which retains `pipe`) waiting forever and leak
        // both pipe fds per launch.
        try? pipe.fileHandleForWriting.close()

        // Stream the merged output; EOF stops the handler.
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                pipe.fileHandleForReading.readabilityHandler = nil
                return
            }
            onOutput(data)
        }

        // Observe the child's exit, then reap it. waitpid is blocking but the
        // source only fires on .exit, so the child is already gone and this
        // returns immediately. Retained by the returned SystemLaunchedProcess.
        let processSource: any DispatchSourceProcess = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit)
        processSource.setEventHandler {
            var status: Int32 = 0
            while Darwin.waitpid(pid, &status, 0) == -1, errno == EINTR {}
            onExit(Self.statusCode(from: status))
            processSource.cancel()
        }
        processSource.resume()

        return SystemLaunchedProcess(pid: pid, processSource: processSource, stdoutPipe: pipe)
    }

    /// Configures the spawn attributes (new process group, cloexec default,
    /// empty signal mask, default SIGTERM/SIGINT/SIGHUP)
    /// and file actions (/dev/null on 0, the pipe on 1 and 2), then calls
    /// posix_spawn. Returns posix_spawn's status (0 on success, the errno on
    /// failure); the caller frees the C arrays and destroys the attributes.
    private func spawnChild(
        pidPtr: UnsafeMutablePointer<pid_t>,
        executable: String,
        argvPtr: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
        envpPtr: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
        attr: UnsafeMutablePointer<posix_spawnattr_t?>,
        actions: UnsafeMutablePointer<posix_spawn_file_actions_t?>,
        writeFD: CInt
    ) -> Int32 {
        var rc = Darwin.posix_spawnattr_init(attr)
        if rc != 0 { return rc }
        rc = Darwin.posix_spawn_file_actions_init(actions)
        if rc != 0 { return rc }

        // POSIX_SPAWN_CLOEXEC_DEFAULT: every fd the child inherits is
        // auto-closed on exec (we close the pipe's write end on 1/2 anyway).
        // POSIX_SPAWN_SETPGROUP + setpgroup(0): the child becomes the leader
        // of a new process group whose id is its own pid.
        // POSIX_SPAWN_SETSIGMASK + an empty mask: posix_spawn otherwise copies
        // the CALLING THREAD's signal mask into the child, and launch() runs on
        // dispatch threads that may block SIGTERM; a blocked mask survives
        // exec, so terminate()'s killpg(SIGTERM) and the VW_PARENT_PID
        // self-SIGTERM would never arrive and uvicorn would outlive the app.
        // POSIX_SPAWN_SETSIGDEF: SIGTERM, SIGINT and SIGHUP start at SIG_DFL
        // even if this process ignores them (SIG_IGN also survives exec).
        // Each call returns 0 or an errno; the first failure is returned as-is
        // so spawnFailed carries a real errno, never -1.
        let flags = POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF
        rc = Darwin.posix_spawnattr_setflags(attr, Int16(flags))
        if rc == 0 { rc = Darwin.posix_spawnattr_setpgroup(attr, 0) }
        var emptyMask = sigset_t()
        if rc == 0 { rc = sigemptyset(&emptyMask) == 0 ? 0 : errno }
        if rc == 0 { rc = Darwin.posix_spawnattr_setsigmask(attr, &emptyMask) }
        var defaultSignals = sigset_t()
        if rc == 0 { rc = sigemptyset(&defaultSignals) == 0 ? 0 : errno }
        for signal in [SIGTERM, SIGINT, SIGHUP] where rc == 0 {
            rc = sigaddset(&defaultSignals, signal) == 0 ? 0 : errno
        }
        if rc == 0 { rc = Darwin.posix_spawnattr_setsigdefault(attr, &defaultSignals) }
        if rc == 0 { rc = Darwin.posix_spawn_file_actions_addopen(actions, 0, "/dev/null", O_RDONLY, 0) }
        if rc == 0 { rc = Darwin.posix_spawn_file_actions_adddup2(actions, writeFD, 1) }
        if rc == 0 { rc = Darwin.posix_spawn_file_actions_adddup2(actions, writeFD, 2) }
        if rc != 0 {
            return rc
        }

        return Darwin.posix_spawn(pidPtr, executable, actions, attr, argvPtr, envpPtr)
    }

    /// Decodes the wait status into the value the old Foundation
    /// `terminationStatus` would have reported: the exit code for a normal
    /// exit, the terminating signal for a signalled exit. (The WIF*/W* macros
    /// are function-like and not Swift-importable, so the bit math is inline.)
    private static func statusCode(from status: Int32) -> Int32 {
        let low = status & 0o177
        let coreDumpFlag = status & WCOREFLAG
        if low == 0 {
            return (status >> 8) & 0xff
        } else {
            return low & ~coreDumpFlag
        }
    }
}

private func freeArgArray(_ array: inout [UnsafeMutablePointer<CChar>?]) {
    for pointer in array where pointer != nil {
        free(pointer)
    }
    array.removeAll()
}
