// HeadlessOptions — command-line parsing for the headless entry point
// (Metal design, non-UI).
//
// `LMWarden --headless --python <abs> [--app-support <abs>]` boots
// `AppController` with the live environment and stops cleanly on
// SIGINT/SIGTERM. With no arguments the binary just prints its version, so
// `parse` returns `nil` for the empty argument list.
import Foundation

/// A failure to parse the headless command line.
enum HeadlessOptionsError: Error, Equatable {
    /// A flag was given but no following value.
    case missingValue(String)
    /// A value that is not an absolute path (does not start with "/").
    case relativePath(String)
    /// A token that is not one of the known flags.
    case unknownArgument(String)
    /// `--headless` was given but `--python <abs>` was never supplied.
    case missingPython
}

/// The parsed headless command line.
struct HeadlessOptions: Equatable {
    let pythonPath: String
    let applicationSupport: String?

    /// Parse `arguments` (excluding argv[0]).
    ///
    /// - `[]` → `nil` (version mode).
    /// - Otherwise the first argument must be `--headless`; then `--python
    ///   <v>` (required) and `--app-support <v>` (optional) in any order.
    /// - A flag with no following value → `.missingValue(flag)`.
    /// - A value not starting with `/` → `.relativePath(value)`.
    /// - Any other token → `.unknownArgument(token)`.
    /// - No `--python` → `.missingPython`.
    /// - A repeated flag: the last one wins.
    static func parse(_ arguments: [String]) throws -> HeadlessOptions? {
        guard !arguments.isEmpty else {
            return nil
        }

        guard arguments[0] == "--headless" else {
            throw HeadlessOptionsError.unknownArgument(arguments[0])
        }

        var pythonPath: String?
        var applicationSupport: String?

        var index = 1
        while index < arguments.count {
            let token = arguments[index]
            switch token {
            case "--python", "--app-support":
                let valueIndex = index + 1
                guard valueIndex < arguments.count else {
                    throw HeadlessOptionsError.missingValue(token)
                }
                let value = arguments[valueIndex]
                guard value.hasPrefix("/") else {
                    throw HeadlessOptionsError.relativePath(value)
                }
                if token == "--python" {
                    pythonPath = value
                } else {
                    applicationSupport = value
                }
                index = valueIndex + 1
            default:
                throw HeadlessOptionsError.unknownArgument(token)
            }
        }

        guard let pythonPath else {
            throw HeadlessOptionsError.missingPython
        }

        return HeadlessOptions(pythonPath: pythonPath, applicationSupport: applicationSupport)
    }
}
