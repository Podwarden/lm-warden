// EditMenu — the main menu an accessory app needs for text editing.
//
// LM Warden runs as an accessory (menu bar only), so no menu bar is ever
// shown, but AppKit still matches key equivalents against NSApp.mainMenu.
// Without an Edit menu ⌘C, ⌘V, ⌘X, ⌘A and ⌘Z do nothing in the dashboard
// window and the first-run fields. Every item has a nil target, so the action
// travels the responder chain to whatever field has focus.
import AppKit

enum EditMenu {
    @MainActor
    static func makeMainMenu() -> NSMenu {
        let main = NSMenu()

        // The first top-level item is the application menu; keep it, even empty.
        let appItem = NSMenuItem()
        appItem.submenu = NSMenu(title: "LM Warden")
        main.addItem(appItem)

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        let editItem = NSMenuItem()
        editItem.submenu = edit
        main.addItem(editItem)
        return main
    }
}
