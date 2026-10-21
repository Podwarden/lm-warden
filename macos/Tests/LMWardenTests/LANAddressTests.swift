import XCTest
@testable import LMWarden

/// Helper to build a `NetworkInterface` for the pick-rules tests.
private func iface(
    _ name: String,
    _ ipv4: String,
    isUp: Bool = true,
    isLoopback: Bool = false
) -> NetworkInterface {
    NetworkInterface(name: name, ipv4: ipv4, isUp: isUp, isLoopback: isLoopback)
}

final class LANAddressTests: XCTestCase {
    // MARK: - Preference order

    func testPrefersEn0OverEn1() {
        // en1 is listed first, but en0 must win.
        let result = LANAddress.pick([
            iface("en1", "10.0.0.2"),
            iface("en0", "10.0.0.1"),
        ])
        XCTAssertEqual(result, "10.0.0.1")
    }

    func testPrefersEn1OverOtherEn() {
        // en5 is listed first, but en1 must win (en1 tier 1 beats en5 tier 2).
        let result = LANAddress.pick([
            iface("en5", "10.0.0.5"),
            iface("en1", "10.0.0.1"),
        ])
        XCTAssertEqual(result, "10.0.0.1")
    }

    func testOtherEnBeforeNonEn() {
        // bridge0 is listed first, but en5 (tier 2) beats bridge0 (tier 3).
        let result = LANAddress.pick([
            iface("bridge0", "10.0.0.9"),
            iface("en5", "10.0.0.5"),
        ])
        XCTAssertEqual(result, "10.0.0.5")
    }

    func testNonEnUsedWhenNoEn() {
        // Only bridge0 is up and qualifying — it must be returned.
        let result = LANAddress.pick([
            iface("bridge0", "10.0.0.9"),
        ])
        XCTAssertEqual(result, "10.0.0.9")
    }

    // MARK: - Input order within tier

    func testInputOrderWithinTier() {
        // Both bridge0 and bridge1 are tier 3; bridge0 (listed first) must win.
        let result = LANAddress.pick([
            iface("bridge0", "10.0.0.9"),
            iface("bridge1", "10.0.0.8"),
        ])
        XCTAssertEqual(result, "10.0.0.9")
    }

    // MARK: - Filter rules

    func testLinkLocalSkipped() {
        // 169.254.10.2 is link-local and must be skipped;
        // 169.255.0.1 is just past the boundary and must be kept.
        let result = LANAddress.pick([
            iface("en0", "169.254.10.2"),
            iface("en1", "169.255.0.1"),
        ])
        XCTAssertEqual(result, "169.255.0.1")
    }

    func testLoopbackSkipped() {
        // A loopback-flagged interface must be ignored even if it is en0.
        let result = LANAddress.pick([
            iface("en0", "10.0.0.1", isLoopback: true),
            iface("en1", "10.0.0.2"),
        ])
        XCTAssertEqual(result, "10.0.0.2")
    }

    func testDownSkipped() {
        // A down interface must be ignored even if it is en0.
        let result = LANAddress.pick([
            iface("en0", "10.0.0.1", isUp: false),
            iface("en1", "10.0.0.2"),
        ])
        XCTAssertEqual(result, "10.0.0.2")
    }

    // MARK: - Nil cases

    func testEmptyIsNil() {
        XCTAssertNil(LANAddress.pick([]))
    }

    func testAllDisqualifiedIsNil() {
        // Every interface is disqualified: down, loopback, or link-local.
        let result = LANAddress.pick([
            iface("en0", "10.0.0.1", isUp: false),
            iface("en1", "10.0.0.2", isLoopback: true),
            iface("en5", "169.254.1.1"),
        ])
        XCTAssertNil(result)
    }

    // MARK: - systemInterfaces smoke

    func testSystemInterfacesDoesNotCrash() {
        // Call the real system call; every returned address must have 3 dots.
        let interfaces = LANAddress.systemInterfaces()
        for iface in interfaces {
            XCTAssertEqual(iface.ipv4.filter { $0 == "." }.count, 3,
                           "expected 3 dots in IPv4 address, got \(iface.ipv4)")
        }
    }
}
