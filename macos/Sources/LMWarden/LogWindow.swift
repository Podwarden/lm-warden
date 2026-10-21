// LogWindow — the read-only log window (mockup #1, failed-state "Show log…").
//
// Thin AppKit view over the supervisor's `LogRing` lines: one 720x480 window
// with a non-editable, selectable text view in the monospaced system font. No
// decisions of its own -- the caller passes the lines to show. Not
// instantiated in tests (a test host has no window server); p7-wire-views
// creates it.
import AppKit

@MainActor
final class LogWindowController {
    private var window: NSWindow?
    private var textView: NSTextView?

    func show(lines: [String]) {
        if window == nil {
            let w = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 720, height: 480),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            w.title = "LM Warden Log"
            w.isReleasedWhenClosed = false
            let scroll = NSTextView.scrollableTextView()
            let tv = scroll.documentView as! NSTextView
            tv.isEditable = false
            tv.isSelectable = true
            tv.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
            w.contentView = scroll
            w.center()
            window = w
            textView = tv
        }
        textView?.string = lines.isEmpty ? "No log output yet." : lines.joined(separator: "\n")
        textView?.scrollToEndOfDocument(nil)
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }
}
