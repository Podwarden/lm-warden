import Foundation
import XCTest
@testable import LMWarden

final class PortSelectorTests: XCTestCase {
    /// The preferred port wins when it is free: it is probed first and chosen,
    /// and nothing in the fallback range is probed.
    func testPreferredWinsWhenFree() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return port == 8080
        }
        XCTAssertEqual(selector.choose(preferred: 8080, fallback: 8081...8099), 8080)
        XCTAssertEqual(calls, [8080])
    }

    /// When the preferred port is taken, the fallback range is probed in
    /// ascending order and the first free port is chosen.
    func testFallsBackInAscendingOrder() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return port == 8083
        }
        XCTAssertEqual(selector.choose(preferred: 8080, fallback: 8081...8099), 8083)
        XCTAssertEqual(calls, [8080, 8081, 8082, 8083])
    }

    /// A preferred port that falls inside the fallback range is probed exactly
    /// once — never again from the fallback scan.
    func testPreferredNotRetriedInFallback() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return port == 8082
        }
        XCTAssertEqual(selector.choose(preferred: 8085, fallback: 8081...8089), 8082)
        XCTAssertEqual(calls, [8085, 8081, 8082])

        // The scan must run PAST the preferred port's slot to show it is skipped.
        var fullScan: [Int] = []
        let busy = PortSelector { port in
            fullScan.append(port)
            return false
        }
        XCTAssertNil(busy.choose(preferred: 8083, fallback: 8081...8085))
        XCTAssertEqual(fullScan, [8083, 8081, 8082, 8084, 8085])
    }

    /// When every candidate is unavailable the selector returns nil and every
    /// in-range candidate was probed exactly once.
    func testNilWhenNoneFree() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return false
        }
        XCTAssertNil(selector.choose(preferred: 8080, fallback: 8081...8083))
        XCTAssertEqual(calls, [8080, 8081, 8082, 8083])
    }

    /// Candidates outside 1...65535 are skipped before any availability check:
    /// preferred 0 and 65536 cost no probes, 65535 and 1 (the exact bounds)
    /// are still probed.
    func testOutOfRangeCandidatesSkippedWithoutProbe() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return false
        }
        XCTAssertNil(selector.choose(preferred: 0, fallback: 65534...65536))
        XCTAssertEqual(calls, [65534, 65535])

        calls = []
        XCTAssertNil(selector.choose(preferred: 1, fallback: -1...0))
        XCTAssertEqual(calls, [1])
    }

    /// Probing stops at the first available candidate: later candidates —
    /// including a second one that is also free — are never reached.
    func testStopsAtFirstAvailable() {
        var calls: [Int] = []
        let selector = PortSelector { port in
            calls.append(port)
            return port == 8082
        }
        XCTAssertEqual(selector.choose(preferred: 8081, fallback: 8082...8085), 8082)
        XCTAssertEqual(calls, [8081, 8082])
    }

    /// The loopback probe rejects ports outside 1...65535 without probing:
    /// both just-past and far-past the bounds read as unavailable.
    func testLoopbackRejectsInvalidPorts() {
        let invalid = [0, -1, -100, 65536, 65537, 70000, Int.min, Int.max]
        for port in invalid {
            let selector = PortSelector()
            XCTAssertFalse(selector.isAvailable(port), "port \(port) must be unavailable")
        }
    }

    /// A port another socket already holds (bound and listening on
    /// 127.0.0.1) reads as unavailable. No claim is made about any port
    /// being free.
    func testLoopbackSeesAHeldPort() {
        let holder = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(holder, 0, "socket() should succeed")
        defer { close(holder) }
        let heldPort = Self.bindLoopbackEphemeral(holder)
        XCTAssertEqual(listen(holder, 1), 0, "listen() should succeed")
        XCTAssertGreaterThanOrEqual(heldPort, 1, "kernel should assign a real port")
        XCTAssertLessThanOrEqual(heldPort, 65535, "kernel should assign a real port")
        XCTAssertFalse(PortSelector.loopbackIsAvailable(heldPort), "a held port must read as unavailable")
    }

    /// A port whose last connection is still closing (the server side closed
    /// first, so 127.0.0.1:<port> sits in FIN_WAIT/TIME_WAIT) reads as
    /// unavailable: the probe must not set SO_REUSEADDR, which would let that
    /// bind succeed.
    func testLoopbackSeesATimeWaitPort() {
        let listener = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(listener, 0, "socket() should succeed")
        let port = Self.bindLoopbackEphemeral(listener)
        XCTAssertEqual(listen(listener, 1), 0, "listen() should succeed")
        let client = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(client, 0, "socket() should succeed")
        var target = Self.loopbackAddress(port: UInt16(clamping: port))
        let connected = withUnsafeMutablePointer(to: &target) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { address in
                connect(client, address, Self.addressLength)
            }
        }
        XCTAssertEqual(connected, 0, "connect() should succeed")
        let accepted = accept(listener, nil, nil)
        XCTAssertGreaterThanOrEqual(accepted, 0, "accept() should succeed")
        // Server side closes first, then the client and the listener: no
        // socket holds the port any more, only the closing connection does.
        close(accepted)
        close(client)
        close(listener)
        XCTAssertFalse(PortSelector.loopbackIsAvailable(port), "a port in TIME_WAIT must read as unavailable")
    }

    private static let addressLength = socklen_t(MemoryLayout<sockaddr_in>.size)

    private static func loopbackAddress(port: UInt16) -> sockaddr_in {
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        return address
    }

    /// Binds `fd` to 127.0.0.1 port 0 and returns the kernel-assigned port.
    private static func bindLoopbackEphemeral(_ fd: Int32) -> Int {
        var address = loopbackAddress(port: 0)
        let bound = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                Darwin.bind(fd, raw, addressLength)
            }
        }
        XCTAssertEqual(bound, 0, "bind() to a kernel-assigned port should succeed")
        var length = addressLength
        let named = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { raw in
                getsockname(fd, raw, &length)
            }
        }
        XCTAssertEqual(named, 0, "getsockname() should succeed")
        return Int(UInt16(bigEndian: address.sin_port))
    }
}
