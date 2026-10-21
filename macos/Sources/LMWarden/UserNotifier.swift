// UserNotifier — posts a `ShellNotification` as a user notification.
//
// Thin view over the UserNotifications framework: no decisions of its own, it
// only translates the model's kind/title/body into a `UNNotificationRequest`.
// A `UNUserNotificationCenter` cannot exist outside an app bundle (it raises
// `bundleProxyForCurrentProcess is nil` under `swift run` or the test host),
// so every method no-ops unless the process runs from a `.app` bundle with a
// bundle identifier. The identifier alone is not enough: under `swift test`
// it is `com.apple.dt.xctest.tool`. `p7-wire-views` owns the call sites.
import Foundation
import UserNotifications

final class UserNotifier {
    private var authorizationRequested = false

    private var isBundled: Bool {
        Bundle.main.bundleURL.pathExtension == "app" && Bundle.main.bundleIdentifier != nil
    }

    init() {}

    /// Ask for notification permission once (alert + sound). A no-op when the
    /// permission was already requested, or outside an app bundle.
    func requestAuthorization() {
        guard isBundled, !authorizationRequested else {
            return
        }
        authorizationRequested = true
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, error in
            if let error {
                NSLog("LM Warden: notification authorization failed: %@", error.localizedDescription)
            }
        }
    }

    /// Deliver `n` as a user notification. A request carries the identifier
    /// `lmw.<kind>`, so a newer notice of the same kind replaces the older one.
    func post(_ n: ShellNotification) {
        guard isBundled else {
            return
        }
        requestAuthorization()
        let content = UNMutableNotificationContent()
        content.title = n.title
        content.body = n.body
        let request = UNNotificationRequest(identifier: "lmw." + n.kind.rawValue, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request) { error in
            if let error {
                NSLog("LM Warden: notification delivery failed: %@", error.localizedDescription)
            }
        }
    }
}
