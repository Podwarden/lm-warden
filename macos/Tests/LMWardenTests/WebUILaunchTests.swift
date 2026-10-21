import XCTest
@testable import LMWarden

/// BACKLOG-macos P8: the bundled web UI export reaches the server as
/// VW_WEBUI_DIR (app/main.py mounts it at /ui), and only when the bundle
/// carries one.
final class WebUILaunchTests: XCTestCase {
    private let secret = String(repeating: "a", count: 32)
    private let bundle = "/Applications/LM Warden.app"

    private func spec(
        webuiDir: String?,
        base: [String: String] = [:]
    ) throws -> ServerLaunchSpec {
        try ServerLaunchSpec.make(
            pythonPath: ShellAppDelegate.bundledPythonPath(bundlePath: bundle),
            port: 8000,
            dataDir: "/data",
            parentPID: 1,
            cookieSecret: secret,
            baseEnvironment: base,
            webuiDir: webuiDir
        )
    }

    /// The export lives at Contents/Resources/webui, beside the embedded
    /// python (bundle-webui.sh's `dest`).
    func testBundledWebUIPathIsResourcesWebui() {
        XCTAssertEqual(ServerLaunchSpec.webuiEnvironmentKey, "VW_WEBUI_DIR")
        XCTAssertEqual(
            ShellAppDelegate.bundledWebUIPath(bundlePath: bundle),
            "/Applications/LM Warden.app/Contents/Resources/webui"
        )
    }

    /// Present means `<webui>/index.html` exists (what app/webui/mount.py
    /// requires); a bundle without it gets nil, never a path the server
    /// would refuse to mount.
    func testBundledWebUIDirIsGatedOnIndexHtml() {
        var asked: [String] = []
        let present = ShellAppDelegate.bundledWebUIDir(bundlePath: bundle) { path in
            asked.append(path)
            return true
        }
        XCTAssertEqual(present, "/Applications/LM Warden.app/Contents/Resources/webui")
        XCTAssertEqual(asked, ["/Applications/LM Warden.app/Contents/Resources/webui/index.html"])
        XCTAssertNil(ShellAppDelegate.bundledWebUIDir(bundlePath: bundle) { _ in false })
    }

    /// A webuiDir is passed through verbatim under VW_WEBUI_DIR and changes
    /// nothing else: same argv, same other keys.
    func testWebuiDirSetsTheVariable() throws {
        let dir = ShellAppDelegate.bundledWebUIPath(bundlePath: bundle)
        let with = try spec(webuiDir: dir)
        let without = try spec(webuiDir: nil)
        XCTAssertEqual(with.environment["VW_WEBUI_DIR"], dir)
        XCTAssertNil(without.environment["VW_WEBUI_DIR"])
        XCTAssertEqual(with.arguments, without.arguments)
        var rest = with.environment
        rest.removeValue(forKey: "VW_WEBUI_DIR")
        XCTAssertEqual(rest, without.environment)
        XCTAssertTrue(with.redactedDescription.contains("VW_WEBUI_DIR=\(dir)"))
    }

    /// nil strips an inherited VW_WEBUI_DIR; a given one wins over it.
    func testInheritedVariableNeverLeaks() throws {
        let base = ["VW_WEBUI_DIR": "/somewhere/else"]
        XCTAssertNil(try spec(webuiDir: nil, base: base).environment["VW_WEBUI_DIR"])
        XCTAssertEqual(try spec(webuiDir: "/x/webui", base: base).environment["VW_WEBUI_DIR"], "/x/webui")
    }

    /// A relative directory is refused (the child's cwd is not the bundle).
    func testRelativeWebuiDirThrows() {
        XCTAssertThrowsError(try spec(webuiDir: "Contents/Resources/webui")) { error in
            XCTAssertEqual(error as? LaunchSpecError, .relativeWebUIDir("Contents/Resources/webui"))
        }
        XCTAssertThrowsError(try spec(webuiDir: "")) { error in
            XCTAssertEqual(error as? LaunchSpecError, .relativeWebUIDir(""))
        }
    }
}
