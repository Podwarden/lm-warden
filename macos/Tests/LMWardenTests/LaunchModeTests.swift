import XCTest
@testable import LMWarden

final class LaunchModeTests: XCTestCase {
    /// An empty argument list inside an app bundle opens the GUI menu bar
    /// app.
    func testEmptyArgsInBundleIsGui() throws {
        let mode = try LaunchMode.resolve(
            arguments: [],
            bundlePath: "/Applications/LMWarden.app"
        )
        XCTAssertEqual(mode, .gui)
    }

    /// A trailing "/" on the bundle path is removed before the ".app" check,
    /// so the GUI is still chosen.
    func testEmptyArgsInBundleWithTrailingSlashIsGui() throws {
        let mode = try LaunchMode.resolve(
            arguments: [],
            bundlePath: "/Applications/LMWarden.app/"
        )
        XCTAssertEqual(mode, .gui)
    }

    /// An empty argument list outside a bundle prints the version.
    func testEmptyArgsOutsideBundleIsVersion() throws {
        let mode = try LaunchMode.resolve(
            arguments: [],
            bundlePath: "/usr/local/bin"
        )
        XCTAssertEqual(mode, .version)
    }

    func testUpdateProbeFlag() throws {
        XCTAssertEqual(try LaunchMode.resolve(arguments: ["--update-probe"], bundlePath: "/x/LM Warden.app"), .updateProbe)
        XCTAssertEqual(try LaunchMode.resolve(arguments: ["-psn_0_1", "--update-probe"], bundlePath: "/usr/local/bin"), .updateProbe)
        XCTAssertThrowsError(try LaunchMode.resolve(arguments: ["--update-probe", "x"], bundlePath: "/x/LM Warden.app"))
    }

    /// The ".app" suffix is case-sensitive: a different case is not a bundle.
    func testAppSuffixIsCaseSensitive() throws {
        let mode = try LaunchMode.resolve(
            arguments: [],
            bundlePath: "/x/LMWarden.APP"
        )
        XCTAssertEqual(mode, .version)
    }

    /// Only a bare ".app" counts as a bundle; ".application" does not.
    func testApplicationSuffixIsNotABundle() throws {
        let mode = try LaunchMode.resolve(
            arguments: [],
            bundlePath: "/x/LMWarden.application"
        )
        XCTAssertEqual(mode, .version)
    }

    /// A lone LaunchServices "-psn_" argument is dropped, leaving the empty
    /// list, which is the GUI inside a bundle.
    func testPsnDroppedInBundleIsGui() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["-psn_0_12345"],
            bundlePath: "/Applications/LMWarden.app"
        )
        XCTAssertEqual(mode, .gui)
    }

    /// An explicit "--version" prints the version even inside a bundle.
    func testVersionFlagInBundleIsVersion() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["--version"],
            bundlePath: "/Applications/LMWarden.app"
        )
        XCTAssertEqual(mode, .version)
    }

    /// "--version" takes no following value: a trailing argument is rejected
    /// by the headless parser, so it surfaces as an unknown argument.
    func testVersionFlagWithTrailingArgThrows() {
        do {
            _ = try LaunchMode.resolve(
                arguments: ["--version", "x"],
                bundlePath: "/Applications/LMWarden.app"
            )
            XCTFail("expected .unknownArgument for a value after --version")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .unknownArgument("--version"))
        } catch {
            XCTFail("expected a HeadlessOptionsError, got \(error)")
        }
    }

    /// A full headless argument list resolves to .headless carrying the
    /// parsed options.
    func testFullHeadlessArgvParses() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["--headless", "--python", "/usr/bin/python3", "--app-support", "/Users/u/Library/Application Support/LMWarden"],
            bundlePath: "/usr/local/bin"
        )
        XCTAssertEqual(
            mode,
            .headless(
                HeadlessOptions(
                    pythonPath: "/usr/bin/python3",
                    applicationSupport: "/Users/u/Library/Application Support/LMWarden"
                )
            )
        )
    }

    /// The headless parser's own errors propagate unchanged: a bare unknown
    /// token is reported as an unknown argument.
    func testBogusArgumentPropagates() {
        do {
            _ = try LaunchMode.resolve(
                arguments: ["--bogus"],
                bundlePath: "/usr/local/bin"
            )
            XCTFail("expected .unknownArgument(--bogus)")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .unknownArgument("--bogus"))
        } catch {
            XCTFail("expected a HeadlessOptionsError, got \(error)")
        }
    }

    /// A "-psn_" argument is dropped before the headless list is parsed, so a
    /// real headless invocation following it still resolves to .headless.
    func testPsnDroppedThenHeadlessParses() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["-psn_1", "--headless", "--python", "/p"],
            bundlePath: "/usr/local/bin"
        )
        XCTAssertEqual(
            mode,
            .headless(HeadlessOptions(pythonPath: "/p", applicationSupport: nil))
        )
    }

    /// Xcode's user-default arguments inside a bundle are a GUI launch, not
    /// an unknown headless flag.
    func testUserDefaultPairsInBundleIsGui() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["-NSDocumentRevisionsDebugMode", "YES", "-AppleLanguages", "(de)"],
            bundlePath: "/Applications/LMWarden.app"
        )
        XCTAssertEqual(mode, .gui)
    }

    /// The App Translocation path of a quarantined download still ends in
    /// ".app", and "-psn_" plus a user-default pair still opens the GUI.
    func testTranslocatedBundleWithPsnAndUserDefaultIsGui() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["-psn_0_12345", "-AppleLanguages", "(en)"],
            bundlePath: "/private/var/folders/ab/xyz/T/AppTranslocation/0A1B-2C3D/d/LMWarden.app"
        )
        XCTAssertEqual(mode, .gui)
    }

    /// Outside a bundle the same user-default pair is not swallowed: the
    /// headless parser still reports it, exactly as before.
    func testUserDefaultPairsOutsideBundleStillThrows() {
        do {
            _ = try LaunchMode.resolve(
                arguments: ["-AppleLanguages", "(de)"],
                bundlePath: "/usr/local/bin"
            )
            XCTFail("expected .unknownArgument(-AppleLanguages)")
        } catch let error as HeadlessOptionsError {
            XCTAssertEqual(error, .unknownArgument("-AppleLanguages"))
        } catch {
            XCTFail("expected a HeadlessOptionsError, got \(error)")
        }
    }

    /// A key with no value (odd count) or a double-dash flag is not a
    /// user-default pair, even inside a bundle.
    func testIncompletePairOrDoubleDashInBundleThrows() {
        for arguments in [["-AppleLanguages"], ["--bogus", "x"]] {
            do {
                _ = try LaunchMode.resolve(
                    arguments: arguments,
                    bundlePath: "/Applications/LMWarden.app"
                )
                XCTFail("expected .unknownArgument for \(arguments)")
            } catch let error as HeadlessOptionsError {
                XCTAssertEqual(error, .unknownArgument(arguments[0]))
            } catch {
                XCTFail("expected a HeadlessOptionsError, got \(error)")
            }
        }
    }

    /// A headless launch from inside the bundle (a LaunchAgent) is unchanged.
    func testHeadlessInsideBundleParses() throws {
        let mode = try LaunchMode.resolve(
            arguments: ["--headless", "--python", "/p"],
            bundlePath: "/Applications/LMWarden.app"
        )
        XCTAssertEqual(
            mode,
            .headless(HeadlessOptions(pythonPath: "/p", applicationSupport: nil))
        )
    }
}
