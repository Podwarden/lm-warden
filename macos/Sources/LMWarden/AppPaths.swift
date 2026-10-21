// AppPaths — VW_DATA_DIR and VW_HF_CACHE_DIR resolution for the shell
// (Metal design s4 Filesystem & networking).
//
// `VW_DATA_DIR` = `~/Library/Application Support/LM Warden/` (SQLite + WAL,
// logs, crashes, hf-token, chat2 attachments) — NOT `~/Library/Caches`,
// which the OS purges under disk pressure. `VW_HF_CACHE_DIR` defaults to
// `<dataDir>/models`. `prepare` creates both and refuses a network
// `dataDir` because WAL on SMB/NFS corrupts; `modelsDir` on a network
// volume is allowed.
import Foundation

enum AppPathsError: Error, Equatable {
    case notAbsolute(String)
    case networkVolume(String)
    case createFailed(String)
}

struct AppPaths: Equatable {
    let dataDir: String
    let modelsDir: String

    static let folderName = "LM Warden"

    /// dataDir = applicationSupport + "/" + folderName (no doubled slash),
    /// modelsDir = modelsOverride ?? dataDir + "/models". A non-absolute
    /// applicationSupport or modelsOverride throws .notAbsolute(value);
    /// applicationSupport is checked first.
    static func resolve(applicationSupport: String, modelsOverride: String? = nil) throws -> AppPaths {
        guard applicationSupport.hasPrefix("/") else { throw AppPathsError.notAbsolute(applicationSupport) }
        if let modelsOverride, !modelsOverride.hasPrefix("/") {
            throw AppPathsError.notAbsolute(modelsOverride)
        }
        let base = applicationSupport.replacingOccurrences(of: "/", with: "", options: [.anchored, .backwards])
        let dataDir = base + "/" + folderName
        let modelsDir = modelsOverride ?? dataDir + "/models"
        return AppPaths(dataDir: dataDir, modelsDir: modelsDir)
    }

    /// Creates dataDir then modelsDir (in that order; a create error becomes
    /// .createFailed(path) and stops), then refuses with .networkVolume(dataDir)
    /// if isLocalVolume is false for dataDir — WAL on SMB/NFS corrupts.
    /// modelsDir on a network volume is allowed.
    static func prepare(
        _ paths: AppPaths,
        isLocalVolume: (String) -> Bool = AppPaths.systemIsLocalVolume,
        createDirectory: (String) throws -> Void = AppPaths.systemCreateDirectory
    ) throws {
        do {
            try createDirectory(paths.dataDir)
        } catch {
            throw AppPathsError.createFailed(paths.dataDir)
        }
        do {
            try createDirectory(paths.modelsDir)
        } catch {
            throw AppPathsError.createFailed(paths.modelsDir)
        }
        guard isLocalVolume(paths.dataDir) else { throw AppPathsError.networkVolume(paths.dataDir) }
    }

    /// FileManager's .applicationSupportDirectory in .userDomainMask, e.g.
    /// `~/Library/Application Support`, scoped by the running bundle id via
    /// `applicationSupport(base:bundleIdentifier:)`: a shipped bundle keeps
    /// the plain directory; a test variant gets its own subdirectory.
    static func defaultApplicationSupport() -> String {
        applicationSupport(
            base: (try? FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            ).path) ?? "~/Library/Application Support",
            bundleIdentifier: Bundle.main.bundleIdentifier
        )
    }

    /// `base` unchanged unless `bundleIdentifier` is a test variant of the app
    /// (`AppVersion.bundleIdentifier` + "." + a non-empty suffix, e.g.
    /// `com.podwarden.lmwarden.smoke`); then `base` + "/" + that id, so the
    /// variant's `<folderName>` data dir never lands in the real one. The
    /// exact app id, nil, a foreign id and the bare prefix without the dot
    /// (`com.podwarden.lmwardenx`) all keep `base`.
    static func applicationSupport(base: String, bundleIdentifier: String?) -> String {
        let prefix = AppVersion.bundleIdentifier + "."
        guard let bundleIdentifier,
              bundleIdentifier.hasPrefix(prefix),
              bundleIdentifier.count > prefix.count
        else {
            return base
        }
        let trimmed = base.replacingOccurrences(of: "/", with: "", options: [.anchored, .backwards])
        return trimmed + "/" + bundleIdentifier
    }

    /// volumeIsLocal on the volume containing `path`; unknown -> true so
    /// a transient lookup failure does not refuse a healthy local volume.
    static func systemIsLocalVolume(_ path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        let values = try? url.resourceValues(forKeys: [.volumeIsLocalKey])
        return values?.volumeIsLocal ?? true
    }

    /// Creates `path` (and intermediates) owner-only (0700): dataDir holds
    /// the SQLite DB and hf-token. An already-existing directory keeps its mode.
    static func systemCreateDirectory(_ path: String) throws {
        try FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
    }
}
