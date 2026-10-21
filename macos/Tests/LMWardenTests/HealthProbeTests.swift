import XCTest
@testable import LMWarden

final class HealthProbeTests: XCTestCase {
    // MARK: - url(port:)

    /// The lowest legal port, 1, yields the loopback health URL.
    func testUrlLowerBound() {
        XCTAssertEqual(
            HealthProbe.url(port: 1)?.absoluteString,
            "http://127.0.0.1:1/healthz"
        )
    }

    /// The highest legal port, 65535, still resolves.
    func testUrlUpperBound() {
        XCTAssertEqual(
            HealthProbe.url(port: 65535)?.absoluteString,
            "http://127.0.0.1:65535/healthz"
        )
    }

    /// Port 0 is not a valid port: `url(port:)` returns nil.
    func testUrlPortZeroInvalid() {
        XCTAssertNil(HealthProbe.url(port: 0))
    }

    /// A negative port is not a valid port: `url(port:)` returns nil.
    func testUrlPortNegativeInvalid() {
        XCTAssertNil(HealthProbe.url(port: -1))
    }

    /// 65536 is just past the legal range: `url(port:)` returns nil.
    func testUrlPortPastUpperInvalid() {
        XCTAssertNil(HealthProbe.url(port: 65536))
    }

    /// A mid-range port yields the expected absolute string.
    func testUrlMidRange() {
        XCTAssertEqual(
            HealthProbe.url(port: 8000)?.absoluteString,
            "http://127.0.0.1:8000/healthz"
        )
    }

    // MARK: - interpret(statusCode:body:)

    /// 200 with a JSON object whose "ok" is Bool true is healthy.
    func testInterpretHealthy() {
        let body = Data(#"{"ok": true, "gpu": {}}"# .utf8)
        XCTAssertEqual(HealthProbe.interpret(statusCode: 200, body: body), .healthy)
    }

    /// 200 with an empty body is not a JSON object with "ok" true.
    func testInterpretHealthyRequiresBody() {
        let result = HealthProbe.interpret(statusCode: 200, body: Data())
        if case .unhealthy(let reason) = result {
            XCTAssertEqual(reason, "bad body")
        } else {
            XCTFail("expected unhealthy, got \(result)")
        }
    }

    /// 199 (one under 200) is unhealthy with a status reason.
    func testInterpretStatus199Unhealthy() {
        let body = Data(#"{"ok": true}"# .utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 199, body: body),
            .unhealthy("status 199")
        )
    }

    /// 201 (one over 200) is unhealthy with a status reason.
    func testInterpretStatus201Unhealthy() {
        let body = Data(#"{"ok": true}"# .utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 201, body: body),
            .unhealthy("status 201")
        )
    }

    /// 500 is unhealthy with a status reason.
    func testInterpretStatus500Unhealthy() {
        let body = Data("oops".utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 500, body: body),
            .unhealthy("status 500")
        )
    }

    /// 200 with non-JSON bytes is unhealthy with a bad-body reason.
    func testInterpret200NonJSON() {
        let body = Data("not json".utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 200, body: body),
            .unhealthy("bad body")
        )
    }

    /// 200 with a JSON array (not an object) is unhealthy with a bad-body reason.
    func testInterpret200JSONArray() {
        let body = Data("[]".utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 200, body: body),
            .unhealthy("bad body")
        )
    }

    /// 200 with a JSON object that has no "ok" key is unhealthy with a bad-body reason.
    func testInterpret200MissingOk() {
        let body = Data(#"{"gpu": {}}"# .utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 200, body: body),
            .unhealthy("bad body")
        )
    }

    /// 200 with "ok": false is unhealthy with a bad-body reason.
    func testInterpret200OkFalse() {
        let body = Data(#"{"ok": false}"# .utf8)
        XCTAssertEqual(
            HealthProbe.interpret(statusCode: 200, body: body),
            .unhealthy("bad body")
        )
    }

    // MARK: - check(completion:)

    /// A healthy response (200 + {"ok": true}) produces .healthy, and the
    /// request targets the /healthz URL for the configured port.
    func testCheckHealthy() {
        var seen: [URLRequest] = []
        let probe = HealthProbe(
            port: 8000,
            timeout: 3,
            fetch: { request, callback in
                seen.append(request)
                callback(Data(#"{"ok": true, "gpu": {}}"# .utf8), self.http200, nil)
            }
        )
        let exp = expectation(description: "check")
        probe.check { result in
            XCTAssertEqual(result, .healthy)
            exp.fulfill()
        }
        waitForExpectations(timeout: 5)
        XCTAssertEqual(seen.count, 1)
        XCTAssertEqual(seen.first?.url?.absoluteString, "http://127.0.0.1:8000/healthz")
        XCTAssertEqual(seen.first?.httpMethod, "GET")
        XCTAssertEqual(seen.first?.timeoutInterval, 3)
        XCTAssertEqual(seen.first?.cachePolicy, .reloadIgnoringLocalCacheData)
    }

    /// A 500 response produces .unhealthy("status 500").
    func testCheckServerError() {
        let probe = HealthProbe(
            port: 8000,
            fetch: { _, callback in
                callback(Data(), self.http500, nil)
            }
        )
        let exp = expectation(description: "check")
        probe.check { result in
            XCTAssertEqual(result, .unhealthy("status 500"))
            exp.fulfill()
        }
        waitForExpectations(timeout: 5)
    }

    /// A transport error (no data, no response) produces .unhealthy with the
    /// error's localizedDescription.
    func testCheckTransportError() {
        let probe = HealthProbe(
            port: 8000,
            fetch: { _, callback in
                callback(nil, nil, URLError(.cannotConnectToHost))
            }
        )
        let exp = expectation(description: "check")
        probe.check { result in
            guard case .unhealthy(let reason) = result else {
                XCTFail("expected unhealthy, got \(result)")
                exp.fulfill()
                return
            }
            let expected = URLError(.cannotConnectToHost).localizedDescription
            XCTAssertEqual(reason, expected)
            exp.fulfill()
        }
        waitForExpectations(timeout: 5)
    }

    /// An invalid port makes no request at all and reports "invalid port".
    func testCheckInvalidPortMakesNoRequest() {
        var fetchWasCalled = false
        let probe = HealthProbe(
            port: 0,
            fetch: { _, callback in
                fetchWasCalled = true
                callback(nil, nil, URLError(.cannotConnectToHost))
            }
        )
        let exp = expectation(description: "check")
        probe.check { result in
            XCTAssertEqual(result, .unhealthy("invalid port"))
            exp.fulfill()
        }
        waitForExpectations(timeout: 5)
        XCTAssertFalse(fetchWasCalled)
    }

    // MARK: - helpers

    private var http200: HTTPURLResponse {
        HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:8000/healthz")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
    }

    private var http500: HTTPURLResponse {
        HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:8000/healthz")!,
            statusCode: 500,
            httpVersion: nil,
            headerFields: nil
        )!
    }
}
