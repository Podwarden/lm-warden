import XCTest
@testable import LMWarden

final class AppVersionTests: XCTestCase {
    /// Under `swift test` the executable runs without a bundled app wrapper,
    /// so CFBundleShortVersionString is absent and `AppVersion.current`
    /// must fall back to "dev".
    func testFallsBackToDev() {
        XCTAssertEqual(AppVersion.current, "dev")
    }

    /// When `LMW_VERSION` is set and non-empty in the environment, it wins
    /// over any bundle lookup — the override is returned (trimmed) before
    /// `Bundle.main` is consulted. The first assertion in this test fails
    /// on the unmodified tree because `current` ignores the environment.
    func testEnvironmentOverrideWins() {
        let version = AppVersion.version(
            environment: ["LMW_VERSION": "1.2.3"],
            bundle: Bundle.main
        )
        XCTAssertEqual(version, "1.2.3")
    }

    /// An empty `LMW_VERSION` is treated as unset: behaviour falls through
    /// to the bundle lookup, which under `swift test` (no real LMWarden
    /// bundle) returns "dev".
    func testEmptyOverrideIgnored() {
        let version = AppVersion.version(
            environment: ["LMW_VERSION": ""],
            bundle: Bundle.main
        )
        XCTAssertEqual(version, "dev")
    }

    /// A whitespace-only `LMW_VERSION` is also treated as unset after
    /// trimming, so it must not leak through as an empty/blank string.
    func testWhitespaceOverrideIgnored() {
        let version = AppVersion.version(
            environment: ["LMW_VERSION": "   "],
            bundle: Bundle.main
        )
        XCTAssertEqual(version, "dev")
    }

    /// The override is trimmed of surrounding whitespace before being
    /// returned, so `"  1.2.3  "` surfaces as `"1.2.3"`.
    func testOverrideIsTrimmed() {
        let version = AppVersion.version(
            environment: ["LMW_VERSION": "  1.2.3  "],
            bundle: Bundle.main
        )
        XCTAssertEqual(version, "1.2.3")
    }

    /// When `LMW_VERSION` is absent entirely, behaviour is unchanged:
    /// the bundle is consulted, and under `swift test` that resolves to
    /// "dev".
    func testNoOverrideFallsBackToDev() {
        let version = AppVersion.version(
            environment: [:],
            bundle: Bundle.main
        )
        XCTAssertEqual(version, "dev")
    }

    /// The bundle path is unchanged: a bundle identified as
    /// `com.podwarden.lmwarden` yields its CFBundleShortVersionString when
    /// no override is set. Pins that `version(environment:bundle:)` reads
    /// the injected `bundle`, not `Bundle.main`.
    func testMatchingBundleVersionUsed() throws {
        let bundle = try makeBundle(identifier: AppVersion.bundleIdentifier, version: "9.8.7")
        XCTAssertEqual(AppVersion.version(environment: [:], bundle: bundle), "9.8.7")
    }

    /// The override wins even over a real LMWarden bundle: the environment
    /// is consulted before any bundle lookup.
    func testOverrideBeatsMatchingBundle() throws {
        let bundle = try makeBundle(identifier: AppVersion.bundleIdentifier, version: "9.8.7")
        XCTAssertEqual(
            AppVersion.version(environment: ["LMW_VERSION": "1.2.3"], bundle: bundle),
            "1.2.3"
        )
    }

    /// A versioned bundle with any other identifier still falls back to "dev".
    func testForeignBundleFallsBackToDev() throws {
        let bundle = try makeBundle(identifier: "com.example.other", version: "9.8.7")
        XCTAssertEqual(AppVersion.version(environment: [:], bundle: bundle), "dev")
    }

    /// `current` wires the real process environment into the lookup.
    func testCurrentReadsProcessEnvironment() {
        let previous = ProcessInfo.processInfo.environment["LMW_VERSION"]
        setenv("LMW_VERSION", "4.5.6", 1)
        defer {
            if let previous {
                setenv("LMW_VERSION", previous, 1)
            } else {
                unsetenv("LMW_VERSION")
            }
        }
        XCTAssertEqual(AppVersion.current, "4.5.6")
    }

    /// A throwaway on-disk bundle with the given identifier and short
    /// version, removed at teardown.
    private func makeBundle(identifier: String, version: String) throws -> Bundle {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppVersionTests-\(UUID().uuidString).bundle")
        let contents = root.appendingPathComponent("Contents")
        try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
        let info: [String: Any] = [
            "CFBundleIdentifier": identifier,
            "CFBundleShortVersionString": version,
            "CFBundlePackageType": "BNDL",
        ]
        let data = try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
        try data.write(to: contents.appendingPathComponent("Info.plist"))
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return try XCTUnwrap(Bundle(url: root))
    }
}
