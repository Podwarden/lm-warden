import XCTest
@testable import LMWarden

/// The dashboard window's navigation policy (BACKLOG-macos P7, mockup screen 5):
/// same-origin navigation stays in the window, other http/https opens in the
/// default browser, every other scheme is cancelled. Pure Foundation, so these
/// tests never touch a window server.
final class DashboardNavigationPolicyTests: XCTestCase {
    private let dashboard = URL(string: "http://127.0.0.1:8080/ui/")!

    private func decide(_ string: String?) -> DashboardNavigation {
        DashboardNavigationPolicy.decide(url: string.flatMap(URL.init(string:)), dashboard: dashboard)
    }

    func testSameOriginIsAllowed() {
        XCTAssertEqual(decide("http://127.0.0.1:8080/ui/"), .allow)
        XCTAssertEqual(decide("http://127.0.0.1:8080/ui/models?x=1#top"), .allow)
        XCTAssertEqual(decide("http://127.0.0.1:8080/api/health"), .allow)
        XCTAssertEqual(decide("http://127.0.0.1:8080"), .allow)
    }

    func testSameHostOtherPortOpensExternally() {
        XCTAssertEqual(decide("http://127.0.0.1:8081/ui/"), .openExternally)
        XCTAssertEqual(decide("http://127.0.0.1/ui/"), .openExternally)
    }

    func testSameHostAndPortOtherSchemeOpensExternally() {
        XCTAssertEqual(decide("https://127.0.0.1:8080/ui/"), .openExternally)
    }

    func testOtherHostSamePortOpensExternally() {
        XCTAssertEqual(decide("http://localhost:8080/ui/"), .openExternally)
        XCTAssertEqual(decide("http://127.0.0.2:8080/ui/"), .openExternally)
    }

    func testHttpsAndHttpElsewhereOpenExternally() {
        XCTAssertEqual(decide("https://github.com/Podwarden/lm-warden"), .openExternally)
        XCTAssertEqual(decide("http://example.com/"), .openExternally)
    }

    func testNonWebSchemesAreCancelled() {
        XCTAssertEqual(decide("file:///etc/passwd"), .cancel)
        XCTAssertEqual(decide("javascript:alert(1)"), .cancel)
        XCTAssertEqual(decide("data:text/html,<b>hi</b>"), .cancel)
        XCTAssertEqual(decide("mailto:someone@example.com"), .cancel)
        XCTAssertEqual(decide("about:srcdoc"), .cancel)
    }

    func testAboutBlankIsAllowed() {
        XCTAssertEqual(decide("about:blank"), .allow)
    }

    func testNilURLIsCancelled() {
        XCTAssertEqual(decide(nil), .cancel)
    }

    func testDashboardPathIsIgnoredOriginOnly() {
        let deep = URL(string: "http://127.0.0.1:9123/ui/settings/tokens")!
        XCTAssertEqual(
            DashboardNavigationPolicy.decide(url: URL(string: "http://127.0.0.1:9123/other/page"), dashboard: deep),
            .allow
        )
        XCTAssertEqual(
            DashboardNavigationPolicy.decide(url: URL(string: "http://127.0.0.1:9124/ui/settings/tokens"), dashboard: deep),
            .openExternally
        )
    }

    func testDefaultPortsCountAsTheSameOrigin() {
        let plain = URL(string: "http://127.0.0.1/ui/")!
        XCTAssertEqual(DashboardNavigationPolicy.decide(url: URL(string: "http://127.0.0.1:80/x"), dashboard: plain), .allow)
        XCTAssertEqual(DashboardNavigationPolicy.decide(url: URL(string: "http://127.0.0.1:81/x"), dashboard: plain), .openExternally)
    }

    func testSchemeAndHostCompareCaseInsensitively() {
        XCTAssertEqual(decide("HTTP://127.0.0.1:8080/ui/"), .allow)
        XCTAssertEqual(
            DashboardNavigationPolicy.decide(
                url: URL(string: "http://LocalHost:8080/ui/"),
                dashboard: URL(string: "http://localhost:8080/ui/")!
            ),
            .allow
        )
    }
}
