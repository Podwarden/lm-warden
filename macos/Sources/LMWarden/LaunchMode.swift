// LaunchMode — decides what a given launch of the LMWarden binary means,
// based on its command line and whether it lives inside a .app bundle
// (Metal design, non-UI).
//
// A binary inside LMWarden.app with no arguments opens the menu bar app;
// the same binary run directly (e.g. `swift run LMWarden`) with no
// arguments prints the version. `--version` always prints the version,
// `--update-probe` runs one informational Sparkle check (UpdateProbe), and
// anything else is handed to the headless parser.
import Foundation

/// What a launch of the LMWarden binary should do.
enum LaunchMode: Equatable {
    /// Print the app version and exit.
    case version
    /// Boot the headless server with the parsed options.
    case headless(HeadlessOptions)
    /// Open the GUI menu bar app.
    case gui
    /// One informational update check against the bundle's feed, then exit.
    case updateProbe
}

extension LaunchMode {
    /// Decide the launch mode from `arguments` (excluding argv[0]) and
    /// `bundlePath` (the path to the running binary's container).
    ///
    /// - Any argument starting with "-psn_" (the LaunchServices process
    ///   serial number some macOS versions pass) is dropped first.
    /// - With nothing left: `.gui` if `bundlePath`, with any trailing "/"
    ///   removed, ends with ".app" (case-sensitive); otherwise `.version`.
    /// - With exactly `["--version"]`: `.version`.
    /// - With exactly `["--update-probe"]`: `.updateProbe`.
    /// - Anything else: `.headless(try HeadlessOptions.parse(rest))`, with
    ///   `HeadlessOptions`' errors propagating unchanged.
    static func resolve(arguments: [String], bundlePath: String) throws -> LaunchMode {
        let rest = arguments.filter { !$0.hasPrefix("-psn_") }

        var path = bundlePath
        while path.hasSuffix("/") {
            path.removeLast()
        }
        let inAppBundle = path.hasSuffix(".app")

        guard !rest.isEmpty else {
            return inAppBundle ? .gui : .version
        }

        // Inside a bundle, a list made only of `-Key value` user-default
        // pairs (Xcode's `-NSDocumentRevisionsDebugMode YES`, a tester's
        // `-AppleLanguages (de)`) is a GUI launch: NSUserDefaults reads
        // them, the headless parser must not reject them.
        if inAppBundle && isUserDefaultPairs(rest) {
            return .gui
        }

        if rest == ["--version"] {
            return .version
        }
        if rest == ["--update-probe"] {
            return .updateProbe
        }

        // `rest` is non-empty here, so `parse` never returns nil.
        let options = try HeadlessOptions.parse(rest)!
        return .headless(options)
    }

    /// True when `arguments` is a non-empty, even-length list whose every
    /// even-index token is a single-dash key (`-Key`, not `--flag`, not a
    /// bare "-"): the NSUserDefaults argument-domain form.
    private static func isUserDefaultPairs(_ arguments: [String]) -> Bool {
        guard !arguments.isEmpty, arguments.count % 2 == 0 else {
            return false
        }
        return stride(from: 0, to: arguments.count, by: 2).allSatisfy { index in
            let key = arguments[index]
            return key.count > 1 && key.hasPrefix("-") && !key.hasPrefix("--")
        }
    }
}
