import XCTest
@testable import LMWarden

final class ModelDirMarkersTests: XCTestCase {
    /// A modelsDir not starting with '/' throws .notAbsolute(modelsDir)
    /// BEFORE any touch/exclude closure is called — the recording log must
    /// stay empty.
    func testRelativeDirThrowsBeforeAnyCall() {
        var touched: [String] = []
        var excluded: [String] = []
        do {
            try ModelDirMarkers.apply(
                modelsDir: "relative/models",
                touch: { path in touched.append(path) },
                excludeFromBackup: { path in excluded.append(path) }
            )
            XCTFail("expected notAbsolute for relative modelsDir")
        } catch let error as ModelDirMarkersError {
            XCTAssertEqual(error, .notAbsolute("relative/models"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
        XCTAssertTrue(touched.isEmpty, "touch must not run for a relative dir")
        XCTAssertTrue(excluded.isEmpty, "exclude must not run for a relative dir")
    }

    /// apply touches the marker file first, then excludes the dir — the
    /// call log must show the marker path before the dir, in that order.
    func testOrderTouchThenExclude() throws {
        var log: [String] = []
        try ModelDirMarkers.apply(
            modelsDir: "/Volumes/disk/models",
            touch: { path in log.append("touch:" + path) },
            excludeFromBackup: { path in log.append("exclude:" + path) }
        )
        XCTAssertEqual(log, ["touch:/Volumes/disk/models/.metadata_never_index", "exclude:/Volumes/disk/models"])
    }

    /// A trailing '/' on modelsDir is not doubled: the marker path is
    /// `modelsDir + "/" + name` with the duplicate slash collapsed, so the
    /// recording closure sees a single '/' between the dir and the name.
    func testTrailingSlashNoDoubleSlash() throws {
        var touched: [String] = []
        try ModelDirMarkers.apply(
            modelsDir: "/Volumes/disk/models/",
            touch: { path in touched.append(path) },
            excludeFromBackup: { _ in }
        )
        XCTAssertEqual(touched, ["/Volumes/disk/models/.metadata_never_index"])
    }

    /// A touch failure maps to .markerFailed(<that file path>) and stops:
    /// excludeFromBackup is never called.
    func testTouchFailureStopsBeforeExclude() {
        var excluded: [String] = []
        do {
            try ModelDirMarkers.apply(
                modelsDir: "/Volumes/disk/models",
                touch: { _ in throw NSError(domain: "ModelDirMarkersTests", code: 1) },
                excludeFromBackup: { path in excluded.append(path) }
            )
            XCTFail("expected markerFailed for the touch")
        } catch let error as ModelDirMarkersError {
            XCTAssertEqual(error, .markerFailed("/Volumes/disk/models/.metadata_never_index"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
        XCTAssertTrue(excluded.isEmpty, "exclude must not run after a touch failure")
    }

    /// An excludeFromBackup failure maps to .excludeFailed(modelsDir).
    func testExcludeFailureMapsToExcludeFailed() {
        do {
            try ModelDirMarkers.apply(
                modelsDir: "/Volumes/disk/models",
                touch: { _ in },
                excludeFromBackup: { _ in throw NSError(domain: "ModelDirMarkersTests", code: 2) }
            )
            XCTFail("expected excludeFailed for the dir")
        } catch let error as ModelDirMarkersError {
            XCTAssertEqual(error, .excludeFailed("/Volumes/disk/models"))
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    /// The production defaults work against a real temp directory: the
    /// marker file exists on disk and the dir's isExcludedFromBackup reads
    /// true. Removed at teardown.
    func testSystemApplyOnTempDir() throws {
        let models = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)/models"
        try FileManager.default.createDirectory(atPath: models, withIntermediateDirectories: true)
        defer {
            let root = (models as NSString).deletingLastPathComponent
            try? FileManager.default.removeItem(atPath: root)
        }
        try ModelDirMarkers.apply(modelsDir: models)
        let marker = models + "/.metadata_never_index"
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker), "marker file was not created: \(marker)")
        let values = try URL(fileURLWithPath: models).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, true, "modelsDir must be excluded from Time Machine")
    }

    /// systemTouch creates an empty file if absent and leaves an existing
    /// file's contents untouched (never truncates).
    func testSystemTouchKeepsExistingContents() throws {
        let dir = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let fresh = dir + "/.metadata_never_index"
        try ModelDirMarkers.systemTouch(fresh)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fresh), "systemTouch did not create the file")
        let freshAttrs = try FileManager.default.attributesOfItem(atPath: fresh)
        let freshSize = (freshAttrs[.size] as? NSNumber)?.intValue ?? -1
        XCTAssertEqual(freshSize, 0, "a freshly created marker must be empty")

        let existing = dir + "/existing"
        try "keep me".write(toFile: existing, atomically: true, encoding: .utf8)
        try ModelDirMarkers.systemTouch(existing)
        let kept = try String(contentsOf: URL(fileURLWithPath: existing), encoding: .utf8)
        XCTAssertEqual(kept, "keep me", "systemTouch must never truncate an existing file")
    }

    /// apply is idempotent: a second run against the same temp dir succeeds
    /// again and leaves the marker and the exclusion flag intact.
    func testApplyTwiceIsIdempotent() throws {
        let models = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)/models"
        try FileManager.default.createDirectory(atPath: models, withIntermediateDirectories: true)
        defer {
            let root = (models as NSString).deletingLastPathComponent
            try? FileManager.default.removeItem(atPath: root)
        }
        try ModelDirMarkers.apply(modelsDir: models)
        try ModelDirMarkers.apply(modelsDir: models)
        let marker = models + "/.metadata_never_index"
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker), "marker missing after the second apply")
        let values = try URL(fileURLWithPath: models).resourceValues(forKeys: [.isExcludedFromBackupKey])
        XCTAssertEqual(values.isExcludedFromBackup, true, "exclusion flag must still be set")
    }

    /// systemTouch surfaces a missing parent dir as a thrown error (it must
    /// not swallow FileManager.createFile's false).
    func testSystemTouchMissingDirThrows() {
        let path = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)/absent/.metadata_never_index"
        XCTAssertThrowsError(try ModelDirMarkers.systemTouch(path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
    }

    /// apply on a models dir that does not exist fails at the marker step with
    /// .markerFailed(<marker path>), not later at the exclude step.
    func testSystemApplyOnMissingDirThrowsMarkerFailed() {
        let models = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)/models"
        XCTAssertThrowsError(try ModelDirMarkers.apply(modelsDir: models)) { error in
            XCTAssertEqual(error as? ModelDirMarkersError, .markerFailed(models + "/.metadata_never_index"))
        }
    }

    /// apply on a read-only (0500) models dir fails with .markerFailed: the
    /// marker could not be written, so the call must not report success.
    func testSystemApplyOnUnwritableDirThrowsMarkerFailed() throws {
        try XCTSkipIf(getuid() == 0, "root ignores directory permissions")
        let root = NSTemporaryDirectory() + "ModelDirMarkersTests-\(UUID().uuidString)"
        let models = root + "/models"
        try FileManager.default.createDirectory(atPath: models, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o500], ofItemAtPath: models)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: models)
            try? FileManager.default.removeItem(atPath: root)
        }
        XCTAssertThrowsError(try ModelDirMarkers.apply(modelsDir: models)) { error in
            XCTAssertEqual(error as? ModelDirMarkersError, .markerFailed(models + "/.metadata_never_index"))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: models + "/.metadata_never_index"))
    }
}
