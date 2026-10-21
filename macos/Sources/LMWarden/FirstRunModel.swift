// FirstRunModel — the four-step first-run machine (Metal design §2, mockup 2
// "First run setup"): Welcome (Mac check), Network, Models folder with the
// 10 GB guard, Finish.
//
// Pure model: Foundation only, no AppKit, no SwiftUI. The first-run view reads
// these strings and renders them verbatim; every label, enabled state and
// check mark is decided here. Facts and free space are injected closures;
// reading them from the system belongs to the view/wiring tasks.
import Foundation

/// The four first-run steps.
enum FirstRunStep: Int {
    case welcome = 1
    case network
    case models
    case finish
}

/// The user's choices collected across the four steps.
struct FirstRunResult: Equatable {
    let shareOnLAN: Bool
    let modelsDir: String
    let excludeModelsFromBackup: Bool
    let openAtLogin: Bool
}

struct FirstRunModel {
    let facts: MacFacts
    let homeDir: String
    let port: Int
    let lanAddress: String?
    let keychainHasSecret: Bool
    let freeSpace: (String) -> Int64?
    let volumeName: (String) -> String?

    private(set) var step: FirstRunStep = .welcome
    private(set) var shareOnLAN: Bool = false
    private(set) var modelsDir: String
    private(set) var excludeModelsFromBackup: Bool = true
    private(set) var openAtLogin: Bool = true

    init(
        facts: MacFacts,
        defaultModelsDir: String,
        homeDir: String,
        port: Int,
        lanAddress: String?,
        keychainHasSecret: Bool,
        freeSpace: @escaping (String) -> Int64?,
        volumeName: @escaping (String) -> String?
    ) {
        self.facts = facts
        self.homeDir = homeDir
        self.port = port
        self.lanAddress = lanAddress
        self.keychainHasSecret = keychainHasSecret
        self.freeSpace = freeSpace
        self.volumeName = volumeName
        self.modelsDir = defaultModelsDir
    }

    // MARK: - Navigation

    /// Advance one step; on .finish, return the collected result. `next()`
    /// does nothing when `canContinue` is false.
    mutating func next() -> FirstRunResult? {
        guard canContinue else { return nil }
        if step == .finish {
            return FirstRunResult(
                shareOnLAN: shareOnLAN,
                modelsDir: modelsDir,
                excludeModelsFromBackup: excludeModelsFromBackup,
                openAtLogin: openAtLogin
            )
        }
        step = FirstRunStep(rawValue: step.rawValue + 1) ?? .finish
        return nil
    }

    /// Return one step; does nothing on .welcome.
    mutating func back() {
        guard step != .welcome else { return }
        step = FirstRunStep(rawValue: step.rawValue - 1) ?? .welcome
    }

    mutating func chooseNetwork(lan: Bool) {
        shareOnLAN = lan
    }

    /// Replace the models folder; ignored unless the path is absolute and
    /// the machine is on .models (the only step whose 10 GB guard checks the
    /// folder, so Finish can never carry a folder that skipped the guard).
    mutating func chooseModelsDir(_ path: String) {
        guard step == .models, path.hasPrefix("/") else { return }
        modelsDir = path
    }

    mutating func toggleExcludeModelsFromBackup() {
        excludeModelsFromBackup.toggle()
    }

    mutating func toggleOpenAtLogin() {
        openAtLogin.toggle()
    }

    // MARK: - Derived: navigation state

    /// false only on .models when the free space is known and below 10 GB.
    var canContinue: Bool {
        if step == .models, let free = freeSpace(modelsDir), free < 10_000_000_000 {
            return false
        }
        return true
    }

    var backEnabled: Bool {
        step != .welcome
    }

    // MARK: - Derived: window and sidebar

    var windowTitle: String {
        "Set Up LM Warden"
    }

    /// Done steps "✓ <Name>", others "<n>. <Name>".
    var stepLabels: [String] {
        let names = ["Welcome", "Network", "Models", "Finish"]
        let steps: [FirstRunStep] = [.welcome, .network, .models, .finish]
        return zip(steps, names).map { step, name in
            if step.rawValue < self.step.rawValue {
                return "✓ \(name)"
            }
            return "\(step.rawValue). \(name)"
        }
    }

    var nextLabel: String {
        step == .finish ? "Open Dashboard" : "Continue"
    }

    // MARK: - Derived: per-step copy

    var headline: String {
        switch step {
        case .welcome:
            return "Welcome to LM Warden"
        case .network:
            return "Who can use this server?"
        case .models:
            return "Where should models be stored?"
        case .finish:
            return "You are ready"
        }
    }

    /// The body paragraph; set on .welcome and .models, empty elsewhere.
    var body: String {
        switch step {
        case .welcome:
            return "LM Warden runs AI models on this Mac and serves them to the apps and people you choose. This setup takes about a minute. Everything else is in the dashboard."
        case .models:
            return "Models are large, often 5 to 50 GB each. Pick a disk with room to spare."
        case .network, .finish:
            return ""
        }
    }

    /// The Welcome step's Mac check rows (never blocks).
    var macCheckRows: [MacCheckRow] {
        MacCheck.rows(facts)
    }

    // MARK: - Network step

    var localCardTitle: String {
        "Only this Mac"
    }

    var localCardBody: String {
        "Apps on this Mac reach it at 127.0.0.1:\(port). Recommended."
    }

    var lanCardTitle: String {
        "People on my network"
    }

    var lanCardBody: String {
        if let lan = lanAddress {
            return "Others reach it at \(lan):\(port). macOS asks you to allow local network access."
        }
        return "Others reach it at this Mac's network address. macOS asks you to allow local network access."
    }

    /// Present only when sharing on the LAN.
    var lanWarning: String? {
        guard shareOnLAN else { return nil }
        return "Traffic on your network is not encrypted. API tokens and sign-ins can be read by others on the same network. Use Tailscale or WireGuard for anything sensitive."
    }

    // MARK: - Models step

    /// modelsDir with a leading homeDir replaced by "~" (only on a
    /// path-component boundary).
    var displayedModelsDir: String {
        abbreviateHome(modelsDir)
    }

    var freeLine: String {
        guard let free = freeSpace(modelsDir) else {
            return "Free space on this disk is unknown. LM Warden will not start a download that would leave less than 10 GB free."
        }
        let volume = volumeName(modelsDir) ?? "this disk"
        return "\(ShellFormat.wholeGigabytes(free)) free on \(volume). LM Warden will not start a download that would leave less than 10 GB free."
    }

    var backupCheckboxLabel: String {
        "Leave models out of Time Machine backups"
    }

    // MARK: - Finish step

    var finishLines: [String] {
        var lines: [String] = []
        if shareOnLAN {
            lines.append("✓ The server is shared with people on your network.")
        } else {
            lines.append("✓ The server starts on this Mac only.")
        }
        lines.append("✓ Models are stored in \(displayedModelsDir).")
        if keychainHasSecret {
            lines.append("✓ Your sign-in secret is saved in your Keychain.")
        } else {
            lines.append("✓ A sign-in secret was created and saved in your Keychain.")
        }
        return lines
    }

    var openAtLoginCheckboxLabel: String {
        "Open LM Warden when I log in"
    }

    var footer: String {
        "Next, the dashboard asks you to create the admin account and download a first model."
    }

    // MARK: - Helpers

    private func abbreviateHome(_ path: String) -> String {
        guard path.hasPrefix(homeDir), path.count >= homeDir.count else { return path }
        // The next character must be "/" (a component boundary); a bare prefix
        // like "/Users/ann2" over home "/Users/ann" is left unchanged.
        if path.count == homeDir.count {
            return "~"
        }
        let afterPrefix = path.index(path.startIndex, offsetBy: homeDir.count)
        guard path[afterPrefix] == "/" else { return path }
        return "~" + path.dropFirst(homeDir.count)
    }
}
