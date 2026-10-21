// MenuBarModel — the menu bar menu's exact rows, labels, check marks and
// enabled states for running / starting / stopped / failed (Metal design §2).
//
// Pure model: Foundation only, no AppKit, no SwiftUI. The menu bar view reads
// this struct and renders it verbatim; every string, enabled state and check
// mark is decided here.
import Foundation

// MARK: - Value types

/// A model currently loaded on the server.
struct LoadedModel: Equatable {
    let name: String
    let backend: String
    let bytes: Int64
}

/// A group of in-flight requests, keyed by caller and model.
struct InFlightGroup: Equatable {
    let who: String
    let model: String
    let requests: Int
}

/// A snapshot of server status the menu bar model needs.
struct ShellStatus: Equatable {
    let loadedModels: [LoadedModel]
    let usedBytes: Int64
    let usableBytes: Int64
    let inFlight: [InFlightGroup]

    var inFlightCount: Int {
        inFlight.reduce(0) { $0 + $1.requests }
    }
}

// MARK: - Menu model types

enum MenuTone: Equatable {
    case green, amber, gray, red
}

enum MenuAction: Equatable {
    case openDashboard
    case copyAPIAddress
    case toggleShareOnLAN
    case startServer
    case stopServer
    case toggleOpenAtLogin
    case checkForUpdates
    case about
    case quit
    case showLog
}

struct MenuItemModel: Equatable {
    let title: String
    let detail: String?
    let keyEquivalent: String?
    let checked: Bool
    let enabled: Bool
    let action: MenuAction
}

struct MenuModelRow: Equatable {
    let title: String
    let size: String
}

struct MenuModelsBlock: Equatable {
    let rows: [MenuModelRow]
    let memoryLine: String
    let memoryFraction: Double
}

struct MenuFailureBlock: Equatable {
    let message: String
    let showLog: MenuItemModel
}

// MARK: - Input and output

struct MenuBarInput: Equatable {
    let state: SupervisorState
    let port: Int
    let shareOnLAN: Bool
    let lanAddress: String?
    let openAtLogin: Bool
    let canCheckForUpdates: Bool
    let status: ShellStatus?
}

struct MenuBarModel: Equatable {
    let title: String
    let statusWord: String
    let tone: MenuTone
    let statusLine: String
    let address: String
    let copyText: String
    let models: MenuModelsBlock?
    let failure: MenuFailureBlock?
    let sections: [[MenuItemModel]]

    // MARK: - make

    static func make(_ input: MenuBarInput) -> MenuBarModel {
        let (statusWord, tone, statusLine) = header(for: input)
        let address = makeAddress(input)
        let copyText = "http://" + address + "/v1"
        let models: MenuModelsBlock?
        if input.state == .running, let status = input.status {
            models = makeModelsBlock(status)
        } else {
            models = nil
        }
        let failure: MenuFailureBlock?
        if case .failed(let failures) = input.state {
            failure = MenuFailureBlock(
                message: "The server stopped \(ShellFormat.count(failures, "time", "times")) in a row and LM Warden gave up restarting it.",
                showLog: MenuItemModel(
                    title: "Show log…",
                    detail: nil,
                    keyEquivalent: nil,
                    checked: false,
                    enabled: true,
                    action: .showLog
                )
            )
        } else {
            failure = nil
        }
        let sections = makeSections(input, address: address)
        return MenuBarModel(
            title: "LM Warden",
            statusWord: statusWord,
            tone: tone,
            statusLine: statusLine,
            address: address,
            copyText: copyText,
            models: models,
            failure: failure,
            sections: sections
        )
    }

    // MARK: - Header

    private static func header(for input: MenuBarInput) -> (String, MenuTone, String) {
        switch input.state {
        case .running:
            let line: String
            if input.shareOnLAN {
                if let lan = input.lanAddress {
                    line = "Serving on your network · \(lan):\(input.port)"
                } else {
                    line = "Serving on your network · port \(input.port)"
                }
            } else {
                line = "Serving on this Mac only · 127.0.0.1:\(input.port)"
            }
            return ("Running", .green, line)
        case .starting:
            return ("Starting", .amber, "Starting the server and checking its health…")
        case .backingOff:
            return ("Starting", .amber, "Starting the server and checking its health…")
        case .stopped:
            return ("Stopped", .gray, "The server is not running. Models are unloaded.")
        case .failed:
            return ("Failed", .red, "The server could not stay up.")
        }
    }

    // MARK: - Address

    private static func makeAddress(_ input: MenuBarInput) -> String {
        if input.shareOnLAN, let lan = input.lanAddress {
            return "\(lan):\(input.port)"
        }
        return "127.0.0.1:\(input.port)"
    }

    // MARK: - Models block

    private static func makeModelsBlock(_ status: ShellStatus) -> MenuModelsBlock {
        let rows: [MenuModelRow]
        if status.loadedModels.isEmpty {
            rows = [MenuModelRow(title: "No models loaded", size: "")]
        } else {
            rows = status.loadedModels.map {
                MenuModelRow(title: "\($0.name) · \($0.backend)", size: ShellFormat.gigabytes($0.bytes))
            }
        }
        let memoryLine = "\(ShellFormat.gigabytesNumber(status.usedBytes)) of \(ShellFormat.gigabytes(status.usableBytes)) usable for models · \(ShellFormat.count(status.inFlightCount, "request", "requests")) in flight"
        let memoryFraction: Double
        if status.usableBytes > 0 {
            memoryFraction = min(max(Double(status.usedBytes) / Double(status.usableBytes), 0), 1)
        } else {
            memoryFraction = 0
        }
        return MenuModelsBlock(rows: rows, memoryLine: memoryLine, memoryFraction: memoryFraction)
    }

    // MARK: - Sections

    private static func makeSections(_ input: MenuBarInput, address: String) -> [[MenuItemModel]] {
        let isRunning = input.state == .running
        let serverActionTitle: String
        let serverAction: MenuAction
        switch input.state {
        case .running, .starting, .backingOff:
            serverActionTitle = "Stop Server"
            serverAction = .stopServer
        case .stopped, .failed:
            serverActionTitle = "Start Server"
            serverAction = .startServer
        }

        let section0: [MenuItemModel] = [
            MenuItemModel(
                title: "Open Dashboard",
                detail: nil,
                keyEquivalent: "d",
                checked: false,
                enabled: isRunning,
                action: .openDashboard
            ),
            MenuItemModel(
                title: "Copy API Address",
                detail: address,
                keyEquivalent: nil,
                checked: false,
                enabled: isRunning,
                action: .copyAPIAddress
            ),
            MenuItemModel(
                title: "Share on Local Network",
                detail: nil,
                keyEquivalent: nil,
                checked: input.shareOnLAN,
                enabled: true,
                action: .toggleShareOnLAN
            ),
        ]

        let section1: [MenuItemModel] = [
            MenuItemModel(
                title: serverActionTitle,
                detail: nil,
                keyEquivalent: nil,
                checked: false,
                enabled: true,
                action: serverAction
            ),
            MenuItemModel(
                title: "Open at Login",
                detail: nil,
                keyEquivalent: nil,
                checked: input.openAtLogin,
                enabled: true,
                action: .toggleOpenAtLogin
            ),
            MenuItemModel(
                title: "Check for Updates…",
                detail: nil,
                keyEquivalent: nil,
                checked: false,
                enabled: input.canCheckForUpdates,
                action: .checkForUpdates
            ),
            MenuItemModel(
                title: "About LM Warden",
                detail: nil,
                keyEquivalent: nil,
                checked: false,
                enabled: true,
                action: .about
            ),
        ]

        let section2: [MenuItemModel] = [
            MenuItemModel(
                title: "Quit LM Warden",
                detail: nil,
                keyEquivalent: "q",
                checked: false,
                enabled: true,
                action: .quit
            ),
        ]

        return [section0, section1, section2]
    }
}
