// ServerLaunchSpec — pure argv/env builder for the embedded uvicorn child
// process (Metal design §3 and §4).
//
// The ServerSupervisor spawns one `Process` running the bundled interpreter
// with `python -I -B -m uvicorn app.main:app`. This type owns only the argv and
// environment it hands to that `Process` — it does not spawn, wait, or kill.
//
// The design pins three things here: bind 127.0.0.1 and pass `--host`/`--port`
// (the app's `bind_host`/`bind_port` config is dead), run the interpreter with
// `-I` (isolated; never `pip install` at runtime) and `-B` (never write a .pyc
// into the signed bundle -- PYTHONDONTWRITEBYTECODE would be ignored under -I),
// and inject VW_DATA_DIR,
// VW_HF_CACHE_DIR, VW_COOKIE_SECRET (from Keychain), VW_SHELL_TOKEN (minted
// per-launch; stripped when absent), VW_PARENT_PID (the belt against a
// headless child outliving the parent) and VW_WEBUI_DIR (the bundled web UI
// export the server mounts at /ui; stripped when absent).
import Foundation

/// Failure to build a `ServerLaunchSpec` from the given launch inputs.
enum LaunchSpecError: Error, Equatable {
    case emptyPythonPath
    case invalidPort(Int)
    case relativeDataDir(String)
    case badCookieSecret
    /// The token never rides along with the error, so it can never be logged.
    case badShellToken
    case relativeWebUIDir(String)
}

/// Pure argv/environment for the embedded uvicorn child.
///
/// Built by `ServerLaunchSpec.make`. The server supervisor reads `executable`
/// and `arguments` for the `Process` and merges `environment` over the child
/// process's environment.
struct ServerLaunchSpec: Equatable {
    let executable: URL
    let arguments: [String]
    let environment: [String: String]

    // planner-22 DEFAULT_BACKEND decision: the Swift launcher sets
    // LMW_PLATFORM_PROFILE=apple so the app's platform profile is "apple".
    static let platformProfile = "apple"

    /// The variable app/main.py reads to mount the UI export at /ui.
    static let webuiEnvironmentKey = "VW_WEBUI_DIR"

    /// Build a launch spec, validating inputs in a fixed order (the first
    /// failure wins): empty `pythonPath` → `emptyPythonPath`; `port` outside
    /// 1...65535 → `invalidPort`; `dataDir` not absolute → `relativeDataDir`;
    /// `cookieSecret` shorter than 32 characters or containing whitespace →
    /// `badCookieSecret`; a non-nil `shellToken` failing `ShellToken.isValid`
    /// → `badShellToken`; a non-nil `webuiDir` that is not absolute →
    /// `relativeWebUIDir`.
    ///
    /// `webuiDir` is the static UI export (bundle-webui.sh lays it at
    /// Contents/Resources/webui); the server mounts it at /ui when
    /// VW_WEBUI_DIR names it. nil leaves the variable out, so the server's
    /// route table is the Linux one.
    static func make(
        pythonPath: String,
        port: Int,
        dataDir: String,
        parentPID: Int32,
        cookieSecret: String,
        host: String = "127.0.0.1",
        hfCacheDir: String? = nil,
        baseEnvironment: [String: String] = [:],
        shellToken: String? = nil,
        webuiDir: String? = nil
    ) throws -> ServerLaunchSpec {
        if pythonPath.isEmpty {
            throw LaunchSpecError.emptyPythonPath
        }
        guard (1...65535).contains(port) else {
            throw LaunchSpecError.invalidPort(port)
        }
        guard dataDir.hasPrefix("/") else {
            throw LaunchSpecError.relativeDataDir(dataDir)
        }
        guard cookieSecret.count >= 32, !cookieSecret.contains(where: { $0.isWhitespace }) else {
            throw LaunchSpecError.badCookieSecret
        }
        if let shellToken, !ShellToken.isValid(shellToken) {
            throw LaunchSpecError.badShellToken
        }
        if let webuiDir, !webuiDir.hasPrefix("/") {
            throw LaunchSpecError.relativeWebUIDir(webuiDir)
        }

        var environment = baseEnvironment
        environment["LMW_PLATFORM_PROFILE"] = ServerLaunchSpec.platformProfile
        environment["VW_DATA_DIR"] = dataDir
        environment["VW_PARENT_PID"] = String(parentPID)
        environment["VW_COOKIE_SECRET"] = cookieSecret
        environment["PYTHONUNBUFFERED"] = "1"
        if let hfCacheDir {
            environment["VW_HF_CACHE_DIR"] = hfCacheDir
        }
        if let shellToken {
            environment[ShellToken.environmentKey] = shellToken
        } else {
            // nil means "no shell": an inherited VW_SHELL_TOKEN (e.g. the
            // app's own shell env) must not reach the child.
            environment.removeValue(forKey: ShellToken.environmentKey)
        }
        if let webuiDir {
            environment[ServerLaunchSpec.webuiEnvironmentKey] = webuiDir
        } else {
            // nil means "no bundled UI": an inherited VW_WEBUI_DIR must not
            // make the child mount a directory the app did not choose.
            environment.removeValue(forKey: ServerLaunchSpec.webuiEnvironmentKey)
        }
        // Isolated interpreter (§3): PYTHONPATH/PYTHONHOME would reach into a
        // foreign site-packages, defeating the -I isolation.
        environment.removeValue(forKey: "PYTHONPATH")
        environment.removeValue(forKey: "PYTHONHOME")

        let arguments = [
            "-I", "-B", "-m", "uvicorn", "app.main:app",
            "--host", host,
            "--port", String(port),
        ]
        return ServerLaunchSpec(
            executable: URL(fileURLWithPath: pythonPath),
            arguments: arguments,
            environment: environment
        )
    }

    /// A human-readable description that never exposes the cookie secret or
    /// the shell token: the executable path, the space-joined arguments, and
    /// the environment keys sorted ascending as `KEY=value` lines with
    /// VW_COOKIE_SECRET's and VW_SHELL_TOKEN's values printed as `<redacted>`.
    var redactedDescription: String {
        var lines = [executable.path, arguments.joined(separator: " ")]
        let redacted = redactedEnvironment
        for key in redacted.keys.sorted() {
            lines.append("\(key)=\(redacted[key] ?? "")")
        }
        return lines.joined(separator: "\n")
    }

    /// `environment` with VW_COOKIE_SECRET's and VW_SHELL_TOKEN's values
    /// replaced by `<redacted>` (exact, case-sensitive keys, as the backend
    /// reads them).
    private var redactedEnvironment: [String: String] {
        var redacted = environment
        for key in ["VW_COOKIE_SECRET", ShellToken.environmentKey] where redacted[key] != nil {
            redacted[key] = "<redacted>"
        }
        return redacted
    }
}

// The cookie secret and the shell token are secrets: a spec printed,
// interpolated or dumped (String(describing:), String(reflecting:), dump, an
// XCTest failure message such as XCTAssertEqual on two specs) shows them
// redacted, never the values.
extension ServerLaunchSpec: CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    var description: String { redactedDescription }

    var debugDescription: String { redactedDescription }

    var customMirror: Mirror {
        Mirror(self, children: [
            "executable": executable,
            "arguments": arguments,
            "environment": redactedEnvironment,
        ])
    }
}
