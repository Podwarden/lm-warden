import XCTest
@testable import LMWarden

final class HeadlessOptionsTests: XCTestCase {
    /// No arguments (excluding argv[0]) is the version mode: `parse` returns
    /// nil so `main` prints the version.
    func testEmptyIsVersionMode() throws {
        let options = try HeadlessOptions.parse([])
        XCTAssertNil(options)
    }

    /// `--headless --python <abs>` is the minimal headless form; the optional
    /// `--app-support` stays nil.
    func testPythonOnly() throws {
        let options = try HeadlessOptions.parse(
            ["--headless", "--python", "/usr/bin/python3"]
        )
        XCTAssertEqual(
            options,
            HeadlessOptions(pythonPath: "/usr/bin/python3", applicationSupport: nil)
        )
    }

    /// `--app-support` may come before or after `--python`.
    func testPythonAndAppSupportAnyOrder() throws {
        let first = try HeadlessOptions.parse(
            ["--headless", "--python", "/opt/venv/bin/python", "--app-support", "/Users/u/Library/Application Support/LMWarden"]
        )
        XCTAssertEqual(
            first,
            HeadlessOptions(
                pythonPath: "/opt/venv/bin/python",
                applicationSupport: "/Users/u/Library/Application Support/LMWarden"
            )
        )

        let second = try HeadlessOptions.parse(
            ["--headless", "--app-support", "/Users/u/Library/Application Support/LMWarden", "--python", "/opt/venv/bin/python"]
        )
        XCTAssertEqual(
            second,
            HeadlessOptions(
                pythonPath: "/opt/venv/bin/python",
                applicationSupport: "/Users/u/Library/Application Support/LMWarden"
            )
        )
    }

    /// The first argument must be `--headless`; anything else is unknown.
    func testFirstArgMustBeHeadless() throws {
        do {
            _ = try HeadlessOptions.parse(["--python", "/usr/bin/python3"])
            XCTFail("expected .unknownArgument for a leading non-headless token")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .unknownArgument("--python"))
        }
    }

    /// A flag with no following value is `.missingValue(flag)`.
    func testMissingValue() throws {
        do {
            _ = try HeadlessOptions.parse(["--headless", "--python"])
            XCTFail("expected .missingValue(--python)")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .missingValue("--python"))
        }

        do {
            _ = try HeadlessOptions.parse(["--headless", "--python", "/usr/bin/python3", "--app-support"])
            XCTFail("expected .missingValue(--app-support)")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .missingValue("--app-support"))
        }
    }

    /// A `--python` value that is not absolute is rejected.
    func testRelativePythonRejected() throws {
        do {
            _ = try HeadlessOptions.parse(["--headless", "--python", "usr/bin/python3"])
            XCTFail("expected .relativePath for a relative --python value")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .relativePath("usr/bin/python3"))
        }
    }

    /// A `--app-support` value that is not absolute is rejected too.
    func testRelativeAppSupportRejected() throws {
        do {
            _ = try HeadlessOptions.parse(
                ["--headless", "--python", "/usr/bin/python3", "--app-support", "Library/Application Support/LMWarden"]
            )
            XCTFail("expected .relativePath for a relative --app-support value")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .relativePath("Library/Application Support/LMWarden"))
        }
    }

    /// Any token that is not one of the known flags is `.unknownArgument`.
    func testUnknownToken() throws {
        do {
            _ = try HeadlessOptions.parse(["--headless", "--python", "/usr/bin/python3", "--bogus"])
            XCTFail("expected .unknownArgument(--bogus)")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .unknownArgument("--bogus"))
        }
    }

    /// `--headless` with no `--python` at all is `.missingPython`.
    func testMissingPython() throws {
        do {
            _ = try HeadlessOptions.parse(["--headless"])
            XCTFail("expected .missingPython")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .missingPython)
        }

        do {
            _ = try HeadlessOptions.parse(
                ["--headless", "--app-support", "/Users/u/Library/Application Support/LMWarden"]
            )
            XCTFail("expected .missingPython when only --app-support is given")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .missingPython)
        }
    }

    /// A repeated flag: the last occurrence wins.
    func testLastRepeatWins() throws {
        let options = try HeadlessOptions.parse(
            [
                "--headless",
                "--python", "/old/python",
                "--app-support", "/old/support",
                "--python", "/new/python",
                "--app-support", "/new/support",
            ]
        )
        XCTAssertEqual(
            options,
            HeadlessOptions(pythonPath: "/new/python", applicationSupport: "/new/support")
        )
    }
}
