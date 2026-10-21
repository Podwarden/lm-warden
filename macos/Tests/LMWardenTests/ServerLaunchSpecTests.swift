import XCTest
@testable import LMWarden

final class ServerLaunchSpecTests: XCTestCase {
    /// A valid 32-character secret with no whitespace.
    private var validSecret: String { String(repeating: "a", count: 32) }

    /// A valid 64-hex shell token, mirroring `ShellToken.generate`'s output.
    private var validShellToken: String { String(repeating: "ab", count: 32) }

    /// `arguments` is exactly `-I -B -m uvicorn app.main:app --host <host>
    /// --port <port>`, in that order, for both the default and a custom
    /// host/port. `-B` is on the argv, not PYTHONDONTWRITEBYTECODE: `-I`
    /// ignores every PYTHON* variable.
    func testArgumentsExact() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 4242,
            cookieSecret: validSecret
        )
        XCTAssertEqual(
            spec.arguments,
            ["-I", "-B", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
        )

        let custom = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 9123,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            host: "0.0.0.0"
        )
        XCTAssertEqual(
            custom.arguments,
            ["-I", "-B", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9123"]
        )
    }

    /// The shell launches the renamed interpreter bundle-python.sh lays down:
    /// `<bundle>/Contents/Resources/python/bin/LM Warden Server` (Metal
    /// design §3), so the process shows under the product's name, never
    /// "python3". It carries the product name, "LM Warden", and fits the
    /// kernel's 16-byte MAXCOMLEN process name whole.
    func testBundledPythonPathIsTheRenamedInterpreter() {
        XCTAssertEqual(ShellAppDelegate.bundledInterpreterName, "LM Warden Server")
        XCTAssertEqual(ShellAppDelegate.bundledInterpreterName.utf8.count, 16)
        XCTAssertEqual(
            ShellAppDelegate.bundledPythonPath(bundlePath: "/Applications/LM Warden.app"),
            "/Applications/LM Warden.app/Contents/Resources/python/bin/LM Warden Server"
        )
        let spec = try? ServerLaunchSpec.make(
            pythonPath: ShellAppDelegate.bundledPythonPath(bundlePath: "/Applications/LM Warden.app"),
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret
        )
        XCTAssertEqual(spec?.executable.lastPathComponent, "LM Warden Server")
    }

    /// `executable` is the python path as a file URL.
    func testExecutableURL() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret
        )
        XCTAssertEqual(spec.executable, URL(fileURLWithPath: "/opt/python/bin/python3"))
    }

    /// An empty `pythonPath` is rejected with `.emptyPythonPath`.
    func testEmptyPythonPathThrows() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .emptyPythonPath)
        }
    }

    /// `port` is valid only on 1...65535: 0 and 65536 throw `.invalidPort`,
    /// while the exact bounds 1 and 65535 succeed.
    func testPortBounds() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 0,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .invalidPort(0))
        }
        XCTAssertNoThrow(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 1,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        )
        XCTAssertNoThrow(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 65535,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        )
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 65536,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .invalidPort(65536))
        }
    }

    /// `dataDir` must be absolute: "" and "data" throw `.relativeDataDir`.
    func testRelativeDataDirThrows() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "",
                parentPID: 1,
                cookieSecret: validSecret
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .relativeDataDir(""))
        }
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "data",
                parentPID: 1,
                cookieSecret: validSecret
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .relativeDataDir("data"))
        }
    }

    /// `cookieSecret` must be >= 32 characters: 31 throws, 32 succeeds.
    func testCookieSecretLengthBound() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: String(repeating: "a", count: 31)
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badCookieSecret)
        }
        XCTAssertNoThrow(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: String(repeating: "a", count: 32)
            )
        )
    }

    /// A `cookieSecret` containing any whitespace or newline is rejected.
    func testCookieSecretWhitespaceThrows() {
        let spaced = String(repeating: "a", count: 15) + " " + String(repeating: "a", count: 16)
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: spaced
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badCookieSecret)
        }
        let newline = String(repeating: "a", count: 31) + "\n"
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: newline
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badCookieSecret)
        }
    }

    /// Validation is ordered: an empty python path AND a bad port report
    /// `.emptyPythonPath` (first failure wins).
    func testValidationOrder() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "",
                port: 70000,
                dataDir: "relative",
                parentPID: 1,
                cookieSecret: "short"
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .emptyPythonPath)
        }
        // Bad port AND relative dataDir AND bad secret -> invalidPort.
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 70000,
                dataDir: "relative",
                parentPID: 1,
                cookieSecret: "short"
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .invalidPort(70000))
        }
        // Relative dataDir AND bad secret -> relativeDataDir.
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "relative",
                parentPID: 1,
                cookieSecret: "short"
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .relativeDataDir("relative"))
        }
    }

    /// The overlay wins over a conflicting `baseEnvironment` key.
    func testOverlayWinsOverBase() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 4242,
            cookieSecret: validSecret,
            baseEnvironment: ["VW_DATA_DIR": "/x"]
        )
        XCTAssertEqual(spec.environment["VW_DATA_DIR"], "/data")
        XCTAssertEqual(spec.environment["VW_PARENT_PID"], "4242")
        XCTAssertEqual(spec.environment["VW_COOKIE_SECRET"], validSecret)
        XCTAssertEqual(spec.environment["PYTHONUNBUFFERED"], "1")
    }

    /// `VW_HF_CACHE_DIR` is set only when `hfCacheDir` is non-nil; with it
    /// nil a base `VW_HF_CACHE_DIR` survives untouched.
    func testHfCacheDirOptional() throws {
        let withCache = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            hfCacheDir: "/models"
        )
        XCTAssertEqual(withCache.environment["VW_HF_CACHE_DIR"], "/models")

        let nilCache = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            hfCacheDir: nil,
            baseEnvironment: ["VW_HF_CACHE_DIR": "/base-cache"]
        )
        XCTAssertEqual(nilCache.environment["VW_HF_CACHE_DIR"], "/base-cache")

        let noBase = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            hfCacheDir: nil
        )
        XCTAssertNil(noBase.environment["VW_HF_CACHE_DIR"])
    }

    /// Unrelated base keys survive; PYTHONPATH and PYTHONHOME are removed.
    func testPythonPathAndHomeRemoved() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: ["HOME": "/Users/ip", "PYTHONPATH": "/nowhere", "PYTHONHOME": "/nowhere2"]
        )
        XCTAssertEqual(spec.environment["HOME"], "/Users/ip")
        XCTAssertNil(spec.environment["PYTHONPATH"])
        XCTAssertNil(spec.environment["PYTHONHOME"])
    }

    /// `redactedDescription` never contains the secret value, shows
    /// `<redacted>` for VW_COOKIE_SECRET, and lists keys sorted ascending.
    func testRedactedDescriptionHidesSecret() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: ["HOME": "/Users/ip"]
        )
        let desc = spec.redactedDescription
        XCTAssertFalse(desc.contains(validSecret), "secret must not appear: \(desc)")
        XCTAssertTrue(desc.contains("<redacted>"))
        XCTAssertTrue(desc.contains("VW_COOKIE_SECRET=<redacted>"))

        // Keys must appear in sorted ascending order.
        let expectedOrder = [
            "HOME=",
            "PYTHONUNBUFFERED=",
            "VW_COOKIE_SECRET=",
            "VW_DATA_DIR=",
            "VW_PARENT_PID=",
        ]
        var lastIndex = -1
        for key in expectedOrder {
            guard let range = desc.range(of: key) else {
                XCTFail("missing key \(key) in \(desc)")
                continue
            }
            let pos = desc.distance(from: desc.startIndex, to: range.lowerBound)
            XCTAssertGreaterThan(pos, lastIndex, "out-of-order key \(key)")
            lastIndex = pos
        }
    }

    /// The overlay sets `LMW_PLATFORM_PROFILE` to `ServerLaunchSpec.platformProfile`
    /// ("apple") even when the base environment is empty.
    func testPlatformProfileIsApple() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret
        )
        XCTAssertEqual(spec.environment["LMW_PLATFORM_PROFILE"], "apple")
    }

    /// `LMW_PLATFORM_PROFILE` overrides any conflicting `baseEnvironment`
    /// value, including the exact "linux" value and an empty string.
    func testPlatformProfileOverridesBase() throws {
        let linuxBase = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: ["LMW_PLATFORM_PROFILE": "linux"]
        )
        XCTAssertEqual(linuxBase.environment["LMW_PLATFORM_PROFILE"], "apple")

        let emptyBase = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: ["LMW_PLATFORM_PROFILE": ""]
        )
        XCTAssertEqual(emptyBase.environment["LMW_PLATFORM_PROFILE"], "apple")
    }

    /// `ServerLaunchSpec.platformProfile` is exactly "apple" — one of the
    /// values app/platform_profile.py accepts (linux, apple).
    func testPlatformProfileConstant() {
        XCTAssertEqual(ServerLaunchSpec.platformProfile, "apple")
    }

    /// `redactedDescription` includes the `LMW_PLATFORM_PROFILE=apple` line,
    /// and it sorts BEFORE `PYTHONUNBUFFERED=`.
    func testRedactedDescriptionShowsProfile() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret
        )
        let desc = spec.redactedDescription
        XCTAssertTrue(desc.contains("LMW_PLATFORM_PROFILE=apple"), "missing profile line: \(desc)")

        let profileRange = desc.range(of: "LMW_PLATFORM_PROFILE=apple")
        let pyRange = desc.range(of: "PYTHONUNBUFFERED=")
        guard let profileRange, let pyRange else {
            XCTFail("missing line in \(desc)")
            return
        }
        let profilePos = desc.distance(from: desc.startIndex, to: profileRange.lowerBound)
        let pyPos = desc.distance(from: desc.startIndex, to: pyRange.lowerBound)
        XCTAssertLessThan(profilePos, pyPos, "LMW_PLATFORM_PROFILE must sort before PYTHONUNBUFFERED")
    }

    /// A valid `shellToken` is injected under `VW_SHELL_TOKEN`.
    func testShellTokenSetInEnvironment() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            shellToken: validShellToken
        )
        XCTAssertEqual(spec.environment[ShellToken.environmentKey], validShellToken)
    }

    /// With `shellToken` nil the key is absent from the environment.
    func testShellTokenNilLeavesKeyAbsent() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret
        )
        XCTAssertNil(spec.environment[ShellToken.environmentKey])
    }

    /// With `shellToken` nil a `VW_SHELL_TOKEN` carried by `baseEnvironment`
    /// is stripped so the child never inherits it.
    func testShellTokenNilStripsInheritedBaseValue() throws {
        let inherited = String(repeating: "f", count: 32)
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: [ShellToken.environmentKey: inherited],
            shellToken: nil
        )
        XCTAssertNil(spec.environment[ShellToken.environmentKey])
    }

    /// A valid `shellToken` wins over a conflicting `baseEnvironment` value.
    func testShellTokenOverridesBaseValue() throws {
        let baseValue = String(repeating: "c", count: 64)
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            baseEnvironment: [ShellToken.environmentKey: baseValue],
            shellToken: validShellToken
        )
        XCTAssertEqual(spec.environment[ShellToken.environmentKey], validShellToken)
    }

    /// `shellToken` is valid only on 32...256 characters: 31 and 257 throw
    /// `.badShellToken`, while the exact bounds 32 and 256 succeed.
    func testShellTokenLengthBound() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret,
                shellToken: String(repeating: "a", count: 31)
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badShellToken)
        }
        XCTAssertNoThrow(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret,
                shellToken: String(repeating: "a", count: 32)
            )
        )
        XCTAssertNoThrow(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret,
                shellToken: String(repeating: "a", count: 256)
            )
        )
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret,
                shellToken: String(repeating: "a", count: 257)
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badShellToken)
        }
    }

    /// A `shellToken` containing a space, '=' or a newline is rejected with
    /// `.badShellToken` (each is prefixed to 31 'a' chars to keep the length
    /// valid at 32).
    func testShellTokenInvalidCharThrows() {
        for bad in [" ", "=", "\n"] {
            XCTAssertThrowsError(
                try ServerLaunchSpec.make(
                    pythonPath: "/opt/python/bin/python3",
                    port: 8000,
                    dataDir: "/data",
                    parentPID: 1,
                    cookieSecret: validSecret,
                    shellToken: bad + String(repeating: "a", count: 31)
                )
            ) { error in
                XCTAssertEqual(error as? LaunchSpecError, .badShellToken)
            }
        }
    }

    /// An empty `shellToken` is not nil and throws `.badShellToken`.
    func testShellTokenEmptyStringThrows() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: validSecret,
                shellToken: ""
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badShellToken)
        }
    }

    /// Validation is ordered: a bad `cookieSecret` AND a bad `shellToken`
    /// report `.badCookieSecret` (the cookie check runs first).
    func testShellTokenValidatedAfterCookieSecret() {
        XCTAssertThrowsError(
            try ServerLaunchSpec.make(
                pythonPath: "/opt/python/bin/python3",
                port: 8000,
                dataDir: "/data",
                parentPID: 1,
                cookieSecret: String(repeating: "a", count: 31),
                shellToken: String(repeating: "a", count: 31)
            )
        ) { error in
            XCTAssertEqual(error as? LaunchSpecError, .badCookieSecret)
        }
    }

    /// `redactedDescription` never contains the token value and prints
    /// `VW_SHELL_TOKEN=<redacted>`.
    func testRedactedDescriptionHidesShellToken() throws {
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: validSecret,
            shellToken: validShellToken
        )
        let desc = spec.redactedDescription
        XCTAssertFalse(desc.contains(validShellToken), "token must not appear: \(desc)")
        XCTAssertTrue(desc.contains("VW_SHELL_TOKEN=<redacted>"))
    }

    /// Printing, debug-printing, interpolating or dumping a spec never shows
    /// the shell token or the cookie secret (an XCTest failure message on a
    /// spec is one of these), and still shows the non-secret environment.
    func testPrintableFormsHideShellTokenAndCookieSecret() throws {
        let secret = String(repeating: "s", count: 40)
        let spec = try ServerLaunchSpec.make(
            pythonPath: "/opt/python/bin/python3",
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: secret,
            shellToken: validShellToken
        )
        var dumped = ""
        dump(spec, to: &dumped)
        for text in [String(describing: spec), String(reflecting: spec), "\(spec)", dumped] {
            XCTAssertFalse(text.contains(validShellToken), "shell token leaked in a printable form")
            XCTAssertFalse(text.contains(secret), "cookie secret leaked in a printable form")
            XCTAssertTrue(text.contains("<redacted>"))
            XCTAssertTrue(text.contains("/data"))
        }
    }
}
