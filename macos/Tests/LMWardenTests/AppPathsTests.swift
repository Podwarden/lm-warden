import XCTest
@testable import LMWarden

final class AppPathsTests: XCTestCase {
    /// The production folder name is the single source of truth for the
    /// Application Support subdirectory (Metal design s4: "LLM Warden" in
    /// the spec; the product is LM Warden, so the folder is `LM Warden` —
    /// planner 24).
    func testFolderNameConstant() {
        XCTAssertEqual(AppPaths.folderName, "LM Warden")
    }

    /// `resolve` builds `<applicationSupport>/LM Warden` and `<dataDir>/models`
    /// when no override is given.
    func testResolveDefault() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        XCTAssertEqual(paths.dataDir, "/Users/x/Library/Application Support/LM Warden")
        XCTAssertEqual(paths.modelsDir, "/Users/x/Library/Application Support/LM Warden/models")
    }

    /// A trailing '/' on applicationSupport is not doubled:
    /// `/Users/x/support/` + `/` + folderName would produce a `//` in the
    /// result, which is the bug this test pins.
    func testResolveTrailingSlash() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support/")
        XCTAssertEqual(paths.dataDir, "/Users/x/Library/Application Support/LM Warden")
        XCTAssertEqual(paths.modelsDir, "/Users/x/Library/Application Support/LM Warden/models")
    }

    /// `modelsOverride` replaces the derived `<dataDir>/models` but leaves
    /// `dataDir` untouched.
    func testModelsOverride() throws {
        let paths = try AppPaths.resolve(
            applicationSupport: "/Users/x/Library/Application Support",
            modelsOverride: "/Volumes/disk/models"
        )
        XCTAssertEqual(paths.dataDir, "/Users/x/Library/Application Support/LM Warden")
        XCTAssertEqual(paths.modelsDir, "/Volumes/disk/models")
    }

    /// A non-absolute applicationSupport throws .notAbsolute(value) — the
    /// value itself is in the payload so a caller can surface it.
    func testRelativeApplicationSupportThrows() {
        do {
            _ = try AppPaths.resolve(applicationSupport: "relative/path")
            XCTFail("expected notAbsolute for relative applicationSupport")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .notAbsolute("relative/path"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    /// A non-absolute modelsOverride throws .notAbsolute(modelsOverride).
    /// applicationSupport is checked first, so a valid applicationSupport
    /// with a relative override still throws for the override value.
    func testRelativeModelsOverrideThrows() {
        do {
            _ = try AppPaths.resolve(applicationSupport: "/Users/x", modelsOverride: "relative/models")
            XCTFail("expected notAbsolute for relative modelsOverride")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .notAbsolute("relative/models"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    /// A relative applicationSupport wins over a relative modelsOverride:
    /// applicationSupport is checked first, so .notAbsolute carries the
    /// applicationSupport value, not the modelsOverride value.
    func testApplicationSupportCheckedBeforeModelsOverride() {
        do {
            _ = try AppPaths.resolve(applicationSupport: "rel/support", modelsOverride: "rel/models")
            XCTFail("expected notAbsolute")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .notAbsolute("rel/support"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    /// prepare creates dataDir first, then modelsDir: the recording closure
    /// must see dataDir before modelsDir in the call log.
    func testCreateOrder() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        var created: [String] = []
        try AppPaths.prepare(
            paths,
            isLocalVolume: { _ in true },
            createDirectory: { path in created.append(path) }
        )
        XCTAssertEqual(created, [paths.dataDir, paths.modelsDir])
    }

    /// When createDirectory throws a non-AppPathsError for dataDir, the
    /// failure is mapped to .createFailed(dataDir) (a CocoaError is NOT
    /// rethrown as-is), and modelsDir is never attempted.
    func testCreateFailureStops() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        var created: [String] = []
        do {
            try AppPaths.prepare(
                paths,
                isLocalVolume: { _ in true },
                createDirectory: { path in
                    if path == paths.dataDir {
                        throw NSError(domain: "AppPathsTests", code: 42)
                    }
                    created.append(path)
                }
            )
            XCTFail("expected createFailed for dataDir")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .createFailed(paths.dataDir))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
        // modelsDir must never be attempted.
        XCTAssertTrue(created.isEmpty)
    }

    /// dataDir on a non-local (network) volume is refused with
    /// .networkVolume(dataDir) — WAL on SMB/NFS corrupts. The check runs
    /// AFTER both creates (the closure log shows both paths were created).
    func testNetworkDataDirRefused() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        var created: [String] = []
        do {
            try AppPaths.prepare(
                paths,
                isLocalVolume: { path in path == paths.modelsDir },
                createDirectory: { path in created.append(path) }
            )
            XCTFail("expected networkVolume for dataDir")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .networkVolume(paths.dataDir))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
        XCTAssertEqual(created, [paths.dataDir, paths.modelsDir])
    }

    /// modelsDir on a network volume is ALLOWED (only dataDir is refused):
    /// isLocalVolume false for modelsDir, true for dataDir, succeeds.
    func testNetworkModelsDirAllowed() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        var created: [String] = []
        try AppPaths.prepare(
            paths,
            isLocalVolume: { path in path != paths.modelsDir },
            createDirectory: { path in created.append(path) }
        )
        XCTAssertEqual(created, [paths.dataDir, paths.modelsDir])
    }

    /// The production defaults work against a real (local) temp directory:
    /// both dirs get created on disk, and are removed at teardown.
    func testSystemDefaultsOnTempDir() throws {
        let support = NSTemporaryDirectory() + "AppPathsTests-\(UUID().uuidString)"
        let paths = try AppPaths.resolve(applicationSupport: support)
        defer { try? FileManager.default.removeItem(atPath: support) }
        try AppPaths.prepare(paths)
        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.dataDir, isDirectory: &isDir),
                      "dataDir was not created: \(paths.dataDir)")
        XCTAssertTrue(isDir.boolValue)
        isDir = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.modelsDir, isDirectory: &isDir),
                      "modelsDir was not created: \(paths.modelsDir)")
        XCTAssertTrue(isDir.boolValue)
        // dataDir holds the DB and hf-token: owner-only.
        let attrs = try FileManager.default.attributesOfItem(atPath: paths.dataDir)
        let mode = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? -1
        XCTAssertEqual(mode & 0o777, 0o700)
    }

    /// A modelsDir create error maps to .createFailed(modelsDir), not dataDir,
    /// and the volume check never runs.
    func testModelsDirCreateFailureNamesModelsDir() throws {
        let paths = try AppPaths.resolve(applicationSupport: "/Users/x/Library/Application Support")
        var created: [String] = []
        var checked: [String] = []
        do {
            try AppPaths.prepare(
                paths,
                isLocalVolume: { path in checked.append(path); return true },
                createDirectory: { path in
                    if path == paths.modelsDir {
                        throw NSError(domain: "AppPathsTests", code: 43)
                    }
                    created.append(path)
                }
            )
            XCTFail("expected createFailed for modelsDir")
        } catch let error as AppPathsError {
            XCTAssertEqual(error, .createFailed(paths.modelsDir))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
        XCTAssertEqual(created, [paths.dataDir])
        XCTAssertTrue(checked.isEmpty)
    }

    /// An unknown volume (lookup fails: the path does not exist) reads as
    /// local, so a transient lookup failure never refuses a healthy volume.
    func testSystemIsLocalVolumeUnknownIsLocal() {
        let missing = NSTemporaryDirectory() + "AppPathsTests-missing-\(UUID().uuidString)/x"
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing))
        XCTAssertTrue(AppPaths.systemIsLocalVolume(missing))
    }

    /// `defaultApplicationSupport` returns FileManager's
    /// .applicationSupportDirectory in .userDomainMask — under `swift test`
    /// that is `~/Library/Application Support`.
    func testDefaultApplicationSupport() throws {
        let expected = try FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .path
        XCTAssertEqual(AppPaths.defaultApplicationSupport(), expected)
    }
    // MARK: - bundle-id-scoped application support

    func testApplicationSupportForExactAppIdIsBase() {
        XCTAssertEqual(
            AppPaths.applicationSupport(base: "/Users/u/Library/Application Support", bundleIdentifier: "com.podwarden.lmwarden"),
            "/Users/u/Library/Application Support"
        )
    }

    func testApplicationSupportForSmokeVariantIsScoped() {
        XCTAssertEqual(
            AppPaths.applicationSupport(base: "/Users/u/Library/Application Support", bundleIdentifier: "com.podwarden.lmwarden.smoke"),
            "/Users/u/Library/Application Support/com.podwarden.lmwarden.smoke"
        )
    }

    func testApplicationSupportForSmokeVariantTrimsTrailingSlash() {
        XCTAssertEqual(
            AppPaths.applicationSupport(base: "/as/", bundleIdentifier: "com.podwarden.lmwarden.smoke"),
            "/as/com.podwarden.lmwarden.smoke"
        )
    }

    /// The scoped base still resolves to `<base>/<id>/LM Warden`, never the
    /// real `<base>/LM Warden`.
    func testSmokeVariantResolvesOutsideRealDataDir() throws {
        let base = AppPaths.applicationSupport(base: "/as", bundleIdentifier: "com.podwarden.lmwarden.smoke")
        let paths = try AppPaths.resolve(applicationSupport: base)
        XCTAssertEqual(paths.dataDir, "/as/com.podwarden.lmwarden.smoke/LM Warden")
    }

    func testApplicationSupportForPrefixWithDotOnlyIsBase() {
        XCTAssertEqual(AppPaths.applicationSupport(base: "/as", bundleIdentifier: "com.podwarden.lmwarden."), "/as")
    }

    func testApplicationSupportForBarePrefixWithoutDotIsBase() {
        XCTAssertEqual(AppPaths.applicationSupport(base: "/as", bundleIdentifier: "com.podwarden.lmwardenx"), "/as")
    }

    func testApplicationSupportForNilIsBase() {
        XCTAssertEqual(AppPaths.applicationSupport(base: "/as", bundleIdentifier: nil), "/as")
    }

    func testApplicationSupportForForeignIdIsBase() {
        XCTAssertEqual(AppPaths.applicationSupport(base: "/as", bundleIdentifier: "com.apple.dt.xctest.tool"), "/as")
    }
}
