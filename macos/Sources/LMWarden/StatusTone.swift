// StatusTone — the colour and spoken description of the server status dot
// (menu bar icon and menu header), per MenuTone. Colours are the approved
// mockup's (Main.dc.html, screen 1): running #1f8f3a, starting #b26b00,
// stopped #8e8e93, failed #c4302b.
//
// Pure value: Foundation only. The view turns the sRGB components into an
// NSColor; the description is what VoiceOver says for the dot, so colour is
// never the only signal.
import Foundation

struct StatusTone: Equatable {
    /// sRGB components, 0...1.
    let red: Double
    let green: Double
    let blue: Double
    /// Lowercase "#rrggbb".
    let hex: String
    /// The state in words, for the dot's accessibility description.
    let accessibilityDescription: String

    init(_ tone: MenuTone) {
        let rgb: UInt32
        switch tone {
        case .green:
            rgb = 0x1f8f3a
            accessibilityDescription = "Server running"
        case .amber:
            rgb = 0xb26b00
            accessibilityDescription = "Server starting"
        case .gray:
            rgb = 0x8e8e93
            accessibilityDescription = "Server stopped"
        case .red:
            rgb = 0xc4302b
            accessibilityDescription = "Server failed"
        }
        red = Double((rgb >> 16) & 0xff) / 255
        green = Double((rgb >> 8) & 0xff) / 255
        blue = Double(rgb & 0xff) / 255
        hex = String(format: "#%06x", rgb)
    }
}
