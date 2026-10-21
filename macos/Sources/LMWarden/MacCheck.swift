// MacCheck — the Welcome step's hardware check rows (Metal design §2, mockup
// 2 "First run setup").
//
// Pure model: Foundation only. Facts and free space are injected; reading
// them from the system belongs to the view/wiring tasks. The rows are
// informational — they never block first run.
import Foundation

/// The hardware facts the Welcome step displays.
struct MacFacts: Equatable {
    let chipName: String?
    let isAppleSilicon: Bool
    let memoryBytes: UInt64
    let freeDiskBytes: Int64?
}

/// One row of the Mac check: a label, a formatted value and an ok flag.
struct MacCheckRow: Equatable {
    let label: String
    let value: String
    let ok: Bool
}

enum MacCheck {
    /// 16 GiB, the minimum memory for first run.
    static let minimumMemoryBytes: UInt64 = 16 * 1_073_741_824

    /// The three rows, in order: Apple Silicon, Memory, Free disk space.
    static func rows(_ f: MacFacts) -> [MacCheckRow] {
        var rows: [MacCheckRow] = []

        // Apple Silicon
        let chip = f.chipName ?? "Intel"
        if f.isAppleSilicon {
            let value = (f.chipName ?? "Apple Silicon") + " — OK"
            rows.append(MacCheckRow(label: "Apple Silicon", value: value, ok: true))
        } else {
            rows.append(MacCheckRow(label: "Apple Silicon", value: chip + " — Apple Silicon required", ok: false))
        }

        // Memory
        let gib = f.memoryBytes / 1_073_741_824
        let memoryValue: String
        let memoryOK: Bool
        if f.memoryBytes >= minimumMemoryBytes {
            memoryValue = "\(gib) GB — OK (16 GB is the minimum)"
            memoryOK = true
        } else {
            memoryValue = "\(gib) GB — below the 16 GB minimum"
            memoryOK = false
        }
        rows.append(MacCheckRow(label: "Memory", value: memoryValue, ok: memoryOK))

        // Free disk space
        let diskValue: String
        let diskOK: Bool
        if let free = f.freeDiskBytes {
            let wholeGigabytes = ShellFormat.wholeGigabytes(free)
            if free >= 10_000_000_000 {
                diskValue = "\(wholeGigabytes) — OK"
                diskOK = true
            } else {
                diskValue = "\(wholeGigabytes) — less than 10 GB free"
                diskOK = false
            }
        } else {
            diskValue = "Unknown"
            diskOK = true
        }
        rows.append(MacCheckRow(label: "Free disk space", value: diskValue, ok: diskOK))

        return rows
    }
}
