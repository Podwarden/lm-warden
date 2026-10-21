import XCTest
@testable import LMWarden

/// The bundle guard is the one decision `UserNotifier` makes. The test host is
/// not a `.app` (its main bundle is the xctest tool, which HAS an identifier),
/// so every call must return before touching `UNUserNotificationCenter`; a
/// guard that lets the call through aborts this process with
/// `bundleProxyForCurrentProcess is nil`.
final class UserNotifierTests: XCTestCase {
    func testOutsideAppBundleEveryCallIsANoOp() {
        XCTAssertNotEqual(Bundle.main.bundleURL.pathExtension, "app")
        let notifier = UserNotifier()
        notifier.requestAuthorization()
        notifier.requestAuthorization()
        notifier.post(ShellNotification(kind: .restarted, title: "t", body: "b"))
    }
}
