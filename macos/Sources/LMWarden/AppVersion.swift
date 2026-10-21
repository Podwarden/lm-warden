// AppVersion — the app version string surfaced to LM Warden's macOS shell.
//
// `AppVersion.current` reads CFBundleShortVersionString from the main
// bundle's Info dictionary, falling back to "dev" when there is no real app
// bundle. That covers two unbundled contexts the tests and `swift run` hit:
//
//   - `swift run LMWarden`: Bundle.main is the executable's build output
//     directory, which has no Info dictionary at all (the key is absent).
//   - `swift test`: Bundle.main is the XCTest runner tool
//     (com.apple.dt.xctest.tool, at .../Xcode.app/Contents/Developer/usr/bin),
//     which DOES carry a CFBundleShortVersionString (the tool's own version,
//     e.g. "16.0") and a real bundleIdentifier — but it is not the LMWarden
//     app, so it must not leak through.
//
// The discriminator is therefore the bundle identifier: only a real
// `LMWarden.app` wrapper (Task 7's DMG) sets
// `Bundle.main.bundleIdentifier == "com.podwarden.lmwarden"` and carries our
// own version. Anything else — the XCTest runner, a bare `swift run`
// executable, another app that links this code — falls back to "dev".
//
// The SwiftUI shell and Task 7's DMG script both consume this.
import Foundation

enum AppVersion {
    /// Our own bundle id, written into the generated Info.plist by
    /// scripts/macos/make-dmg.sh (Task 7).
    static let bundleIdentifier = "com.podwarden.lmwarden"

    /// The app's short version string, or `"dev"` when not running inside a
    /// real `LMWarden.app` bundle identified as `com.podwarden.lmwarden`.
    ///
    /// `LMW_VERSION` in the environment overrides any bundle lookup: when
    /// set and non-empty after trimming, the trimmed value is returned. This
    /// lets CI and ops pin a version string for an unbundled `swift run`
    /// without rebuilding. The lookup is delegated to
    /// `version(environment:bundle:)` so tests can inject both inputs.
    static var current: String {
        version(
            environment: ProcessInfo.processInfo.environment,
            bundle: Bundle.main
        )
    }

    /// Injectable core of `current`: returns a non-empty (after trimming)
    /// `LMW_VERSION` from `environment` before consulting `bundle`'s
    /// `CFBundleShortVersionString` (only when `bundle` is identified as
    /// `com.podwarden.lmwarden`), else `"dev"`.
    static func version(environment: [String: String], bundle: Bundle) -> String {
        if let override = environment["LMW_VERSION"]?.trimmingCharacters(
            in: .whitespacesAndNewlines
        ), !override.isEmpty {
            return override
        }
        let key = "CFBundleShortVersionString"
        guard bundle.bundleIdentifier == AppVersion.bundleIdentifier,
              let version = bundle.object(forInfoDictionaryKey: key) as? String,
              !version.isEmpty
        else {
            return "dev"
        }
        return version
    }
}
