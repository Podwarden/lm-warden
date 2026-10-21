// ShellFormat — formatting helpers for the menu bar model (Metal design §2).
//
// Pure Foundation helpers: no AppKit, no SwiftUI. The menu bar view reads
// pre-formatted strings from `MenuBarModel`; these functions produce them.
import Foundation

enum ShellFormat {
    /// Bytes as a decimal-gigabyte number with exactly one decimal place,
    /// e.g. 6_100_000_000 -> "6.1". Negative values clamp to 0.
    static func gigabytesNumber(_ bytes: Int64) -> String {
        let value = Double(max(bytes, 0)) / 1_000_000_000
        return String(format: "%.1f", value)
    }

    /// `gigabytesNumber` followed by " GB".
    static func gigabytes(_ bytes: Int64) -> String {
        gigabytesNumber(bytes) + " GB"
    }

    /// Whole (floored) decimal gigabytes, e.g. 412_999_999_999 -> "412 GB".
    /// Negative values produce "0 GB".
    static func wholeGigabytes(_ bytes: Int64) -> String {
        let value = max(bytes, 0) / 1_000_000_000
        return "\(value) GB"
    }

    /// Pluralised count: "1 request", "0 requests", "3 requests".
    static func count(_ n: Int, _ singular: String, _ plural: String) -> String {
        if n == 1 {
            return "1 \(singular)"
        }
        return "\(n) \(plural)"
    }
}
