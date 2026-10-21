// EditMenuTests — the app is an accessory (no visible menu bar), but AppKit
// still routes ⌘C/⌘V/⌘X/⌘A/⌘Z through NSApp.mainMenu. Without an Edit menu
// those keys do nothing in the dashboard window or the first-run fields.
import AppKit
import XCTest
@testable import LMWarden

@MainActor
final class EditMenuTests: XCTestCase {
    private func item(_ menu: NSMenu, _ key: String, _ mods: NSEvent.ModifierFlags = .command) -> NSMenuItem? {
        menu.items.first { $0.keyEquivalent == key && $0.keyEquivalentModifierMask == mods }
    }

    func testMainMenuHasEditMenuWithStandardShortcuts() throws {
        let main = EditMenu.makeMainMenu()
        let edit = try XCTUnwrap(main.items.first { $0.submenu?.title == "Edit" }?.submenu)

        let expected: [(String, NSEvent.ModifierFlags, Selector)] = [
            ("z", .command, Selector(("undo:"))),
            ("z", [.command, .shift], Selector(("redo:"))),
            ("x", .command, #selector(NSText.cut(_:))),
            ("c", .command, #selector(NSText.copy(_:))),
            ("v", .command, #selector(NSText.paste(_:))),
            ("a", .command, #selector(NSText.selectAll(_:))),
        ]
        for (key, mods, action) in expected {
            let it = try XCTUnwrap(item(edit, key, mods), "missing \(mods) \(key)")
            XCTAssertEqual(it.action, action, "\(key)")
            // nil target: the action goes down the responder chain to the
            // focused field (a WKWebView input or an NSTextField).
            XCTAssertNil(it.target, "\(key)")
        }
    }

    func testEditMenuIsNotTheFirstItem() throws {
        // AppKit treats the first top-level item as the application menu.
        let main = EditMenu.makeMainMenu()
        XCTAssertGreaterThanOrEqual(main.items.count, 2)
        XCTAssertNotEqual(main.items.first?.submenu?.title, "Edit")
    }
}
