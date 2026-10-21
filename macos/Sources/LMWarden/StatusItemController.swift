// StatusItemController — the menu bar item and its menu (mockup #1, Main.dc.html).
//
// Thin AppKit view over MenuBarModel: every string, enabled state, check mark
// and key equivalent comes from the model. `render` rebuilds the whole menu on
// every call, on the main thread. Not instantiated in tests (a test host has
// no window server); p7-wire-views creates it.
//
// The menu-bar button keeps the server.rack symbol as a TEMPLATE image (so
// AppKit tints it for light/dark menu bars and the highlighted state) padded
// with transparent room on its right; the coloured status dot is a separate
// non-template subview placed in that room. Colours and the dot's spoken
// description come from StatusTone.
import AppKit

@MainActor
final class StatusItemController: NSObject {
    private let onAction: (MenuAction) -> Void
    private let statusItem: NSStatusItem
    private let menu: NSMenu
    private let buttonDot = StatusDotView()

    /// Right-aligned tab stop for trailing text (sizes, the API address).
    private static let trailingTabLocation: CGFloat = 300

    init(onAction: @escaping (MenuAction) -> Void) {
        self.onAction = onAction
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        self.menu = NSMenu()
        super.init()
        menu.autoenablesItems = false
        statusItem.menu = menu
        if let button = statusItem.button {
            let icon = Self.paddedIcon()
            button.image = icon
            button.imagePosition = .imageOnly
            button.setAccessibilityLabel("LM Warden")
            buttonDot.translatesAutoresizingMaskIntoConstraints = false
            button.addSubview(buttonDot)
            // The image is centred in the button; the dot sits centred in the
            // transparent room at the image's right edge.
            let dotCenterOffset = icon.size.width / 2 - Self.menuBarDotSize / 2
            NSLayoutConstraint.activate([
                buttonDot.widthAnchor.constraint(equalToConstant: Self.menuBarDotSize),
                buttonDot.heightAnchor.constraint(equalToConstant: Self.menuBarDotSize),
                buttonDot.centerXAnchor.constraint(equalTo: button.centerXAnchor, constant: dotCenterOffset),
                buttonDot.centerYAnchor.constraint(equalTo: button.centerYAnchor),
            ])
        }
    }

    // MARK: - Render

    func render(_ model: MenuBarModel) {
        let tone = StatusTone(model.tone)
        buttonDot.tone = tone
        statusItem.button?.setAccessibilityLabel("\(model.title) — \(model.statusWord)")
        statusItem.button?.toolTip = "\(model.title) — \(model.statusWord)"

        menu.removeAllItems()

        let header = disabledItem("\(model.title) — \(model.statusWord)")
        header.image = Self.dotImage(tone: tone)
        menu.addItem(header)
        menu.addItem(disabledItem(model.statusLine))

        if let models = model.models {
            menu.addItem(.separator())
            menu.addItem(disabledItem("MODELS LOADED"))
            for row in models.rows {
                let item = disabledItem(row.title)
                if !row.size.isEmpty {
                    item.attributedTitle = Self.trailingTitle(row.title, trailing: row.size)
                }
                menu.addItem(item)
            }
            let bar = NSMenuItem()
            bar.isEnabled = false
            bar.view = MemoryBarView(fraction: models.memoryFraction, caption: models.memoryLine)
            menu.addItem(bar)
        }

        if let failure = model.failure {
            menu.addItem(.separator())
            menu.addItem(disabledItem(failure.message))
            menu.addItem(actionItem(failure.showLog))
        }

        for section in model.sections {
            menu.addItem(.separator())
            for itemModel in section {
                menu.addItem(actionItem(itemModel))
            }
        }
    }

    // MARK: - Items

    private func disabledItem(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        return item
    }

    private func actionItem(_ model: MenuItemModel) -> NSMenuItem {
        let item = NSMenuItem(
            title: model.title,
            action: #selector(menuItemSelected(_:)),
            keyEquivalent: model.keyEquivalent ?? ""
        )
        if model.keyEquivalent != nil {
            item.keyEquivalentModifierMask = [.command]
        }
        // NSMenuItem.target is weak: no retain cycle with the controller.
        item.target = self
        item.representedObject = ActionBox(model.action)
        item.isEnabled = model.enabled
        item.state = model.checked ? .on : .off
        if let detail = model.detail {
            item.attributedTitle = Self.trailingTitle(model.title, trailing: detail)
        }
        return item
    }

    @objc private func menuItemSelected(_ sender: NSMenuItem) {
        guard let box = sender.representedObject as? ActionBox else { return }
        onAction(box.action)
    }

    // MARK: - Drawing

    private static func trailingTitle(_ title: String, trailing: String) -> NSAttributedString {
        let font = NSFont.menuFont(ofSize: 0)
        let paragraph = NSMutableParagraphStyle()
        paragraph.tabStops = [NSTextTab(textAlignment: .right, location: trailingTabLocation)]
        let result = NSMutableAttributedString(
            string: title,
            attributes: [.font: font, .paragraphStyle: paragraph]
        )
        result.append(NSAttributedString(
            string: "\t" + trailing,
            attributes: [
                .font: font,
                .paragraphStyle: paragraph,
                .foregroundColor: NSColor.secondaryLabelColor,
            ]
        ))
        return result
    }

    /// Menu-bar dot diameter and its gap from the icon (mockup: 7 pt, 6 pt).
    private static let menuBarDotSize: CGFloat = 7
    private static let menuBarDotGap: CGFloat = 6

    /// The server.rack symbol as a template image, widened with transparent
    /// room on the right for the status dot subview. Stays a template, so the
    /// menu bar tints it for light, dark and highlighted states.
    private static func paddedIcon() -> NSImage {
        guard let symbol = NSImage(systemSymbolName: "server.rack", accessibilityDescription: "LM Warden") else {
            return NSImage(size: NSSize(width: 16 + menuBarDotGap + menuBarDotSize, height: 16))
        }
        let size = NSSize(
            width: symbol.size.width + menuBarDotGap + menuBarDotSize,
            height: symbol.size.height
        )
        let image = NSImage(size: size, flipped: false) { _ in
            symbol.draw(in: NSRect(origin: .zero, size: symbol.size))
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "LM Warden"
        return image
    }

    /// A small filled circle in the tone's colour, for the menu header. Not a
    /// template image (the colour carries meaning), so its accessibility
    /// description carries the state in words.
    private static func dotImage(tone: StatusTone) -> NSImage {
        let color = tone.color
        let size = NSSize(width: 8, height: 8)
        let image = NSImage(size: size, flipped: false) { rect in
            color.setFill()
            NSBezierPath(ovalIn: rect).fill()
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = tone.accessibilityDescription
        return image
    }
}

private extension StatusTone {
    var color: NSColor {
        NSColor(srgbRed: red, green: green, blue: blue, alpha: 1)
    }
}

/// The coloured status dot beside the menu-bar icon. A plain view rather than
/// part of the button image, so the icon can stay a template while the dot
/// keeps its colour; it is its own accessibility element naming the state.
private final class StatusDotView: NSView {
    var tone: StatusTone? {
        didSet {
            guard tone != oldValue else { return }
            setAccessibilityLabel(tone?.accessibilityDescription)
            needsDisplay = true
        }
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setAccessibilityElement(true)
        setAccessibilityRole(.image)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    // Clicks go to the status bar button underneath.
    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    override func draw(_ dirtyRect: NSRect) {
        guard let tone else { return }
        tone.color.setFill()
        NSBezierPath(ovalIn: bounds).fill()
    }
}

/// The memory bar under the loaded models: a thin rounded track filled to
/// `fraction` in the accent colour, with the memory caption beneath it.
/// Width-resizable, so NSMenu sizes it to the menu's width.
private final class MemoryBarView: NSView {
    private static let leadingInset: CGFloat = 20
    private static let trailingInset: CGFloat = 14
    private static let barHeight: CGFloat = 6
    private static let captionFont = NSFont.menuFont(ofSize: 11)

    private let fraction: CGFloat
    private let caption = NSTextField(labelWithString: "")

    init(fraction: Double, caption text: String) {
        self.fraction = CGFloat(min(max(fraction, 0), 1))
        let captionHeight = ceil(Self.captionFont.ascender - Self.captionFont.descender + Self.captionFont.leading)
        let height = 4 + Self.barHeight + 4 + captionHeight + 4
        super.init(frame: NSRect(x: 0, y: 0, width: 320, height: height))
        autoresizingMask = [.width]

        caption.stringValue = text
        caption.font = Self.captionFont
        caption.textColor = .secondaryLabelColor
        caption.lineBreakMode = .byTruncatingTail
        caption.frame = NSRect(
            x: Self.leadingInset,
            y: 4,
            width: bounds.width - Self.leadingInset - Self.trailingInset,
            height: captionHeight
        )
        caption.autoresizingMask = [.width]
        caption.setAccessibilityElement(false)
        addSubview(caption)

        setAccessibilityElement(true)
        setAccessibilityRole(.levelIndicator)
        setAccessibilityLabel(text)
        setAccessibilityMinValue(0)
        setAccessibilityMaxValue(100)
        setAccessibilityValue(Int((self.fraction * 100).rounded()))
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    override var isFlipped: Bool { false }

    override func draw(_ dirtyRect: NSRect) {
        let track = NSRect(
            x: Self.leadingInset,
            y: bounds.height - 4 - Self.barHeight,
            width: bounds.width - Self.leadingInset - Self.trailingInset,
            height: Self.barHeight
        )
        let radius = Self.barHeight / 2
        let trackPath = NSBezierPath(roundedRect: track, xRadius: radius, yRadius: radius)
        NSColor.quaternaryLabelColor.setFill()
        trackPath.fill()
        guard fraction > 0 else { return }
        NSGraphicsContext.saveGraphicsState()
        trackPath.addClip()
        NSColor.controlAccentColor.setFill()
        NSRect(x: track.minX, y: track.minY, width: track.width * fraction, height: track.height).fill()
        NSGraphicsContext.restoreGraphicsState()
    }
}

/// Carries a MenuAction on an NSMenuItem (representedObject must be an object).
private final class ActionBox: NSObject {
    let action: MenuAction
    init(_ action: MenuAction) {
        self.action = action
    }
}
