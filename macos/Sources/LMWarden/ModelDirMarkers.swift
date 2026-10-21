// ModelDirMarkers — Time Machine exclusion + .metadata_never_index for the
// models dir (Metal design s4: "Set NSURLIsExcludedFromBackupKey +
// .metadata_never_index on the model dir").
//
// Downloaded weights are large and reproducible, so they must not bloat Time
// Machine backups (isExcludedFromBackup) and must not be crawled by Spotlight
// (.metadata_never_index sentinel file inside the dir).
import Foundation

enum ModelDirMarkersError: Error, Equatable {
    case notAbsolute(String)
    case markerFailed(String)
    case excludeFailed(String)
}

enum ModelDirMarkers {
    /// The Spotlight exclusion sentinel, dropped inside the models dir.
    static let neverIndexFileName = ".metadata_never_index"

    /// Touches the marker file then excludes the dir (in that order). A
    /// non-absolute modelsDir throws .notAbsolute(modelsDir) before any call;
    /// a touch error becomes .markerFailed(<that file path>) and stops; an
    /// exclude error becomes .excludeFailed(modelsDir).
    static func apply(
        modelsDir: String,
        touch: (String) throws -> Void = ModelDirMarkers.systemTouch,
        excludeFromBackup: (String) throws -> Void = ModelDirMarkers.systemExcludeFromBackup
    ) throws {
        guard modelsDir.hasPrefix("/") else { throw ModelDirMarkersError.notAbsolute(modelsDir) }
        // No doubled slash when modelsDir ends in '/'.
        let markerPath = modelsDir.hasSuffix("/")
            ? modelsDir + neverIndexFileName
            : modelsDir + "/" + neverIndexFileName
        do {
            try touch(markerPath)
        } catch {
            throw ModelDirMarkersError.markerFailed(markerPath)
        }
        do {
            try excludeFromBackup(modelsDir)
        } catch {
            throw ModelDirMarkersError.excludeFailed(modelsDir)
        }
    }

    /// Creates an empty marker file if absent; an existing file's contents are
    /// left untouched (never truncates). One open(2) with O_CREAT and no
    /// O_TRUNC: create-or-keep in a single syscall (no exists-then-create
    /// race), and a missing or unwritable dir throws the POSIX error instead of
    /// being swallowed (FileManager.createFile only returns false).
    static func systemTouch(_ path: String) throws {
        let fd = open(path, O_WRONLY | O_CREAT | O_CLOEXEC, 0o644)
        guard fd >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        close(fd)
    }

    /// Sets NSURLIsExcludedFromBackupKey (isExcludedFromBackup = true) on the
    /// directory URL so Time Machine skips it.
    static func systemExcludeFromBackup(_ path: String) throws {
        var url = URL(fileURLWithPath: path)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }
}
