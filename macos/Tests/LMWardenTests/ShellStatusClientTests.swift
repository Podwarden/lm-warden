import XCTest
@testable import LMWarden

final class ShellStatusClientTests: XCTestCase {
    // The exact 200 body the client must decode (extra keys id/state/
    // pressure_level/count are present and must be ignored).
    static let fixture = #"""
        {"models":[{"id":"e1","name":"Llama 3 8B","backend":"mlx","state":"running","bytes":6100000000},{"id":"e2","name":"Qwen 2.5 14B","backend":"vllm","state":"starting","bytes":null}],"memory":{"used_bytes":6100000000,"usable_bytes":21300000000,"pressure_level":1},"in_flight":{"count":4,"groups":[{"who":"web","model":"Llama 3 8B","requests":3},{"who":"unknown","model":"Qwen 2.5 14B","requests":1}]}}
        """#

    let token = "tok-0123456789abcdef0123456789abcdef"

    // MARK: - url(port:)

    /// The lowest legal port, 1, yields the loopback status URL.
    func testUrlLowerBound() {
        XCTAssertEqual(
            ShellStatusClient.url(port: 1)?.absoluteString,
            "http://127.0.0.1:1/api/shell/status"
        )
    }

    /// The highest legal port, 65535, still resolves.
    func testUrlUpperBound() {
        XCTAssertEqual(
            ShellStatusClient.url(port: 65535)?.absoluteString,
            "http://127.0.0.1:65535/api/shell/status"
        )
    }

    /// Port 0 is not a valid port: `url(port:)` returns nil.
    func testUrlPortZeroInvalid() {
        XCTAssertNil(ShellStatusClient.url(port: 0))
    }

    /// 65536 is just past the legal range: `url(port:)` returns nil.
    func testUrlPortPastUpperInvalid() {
        XCTAssertNil(ShellStatusClient.url(port: 65536))
    }

    // MARK: - request(port:token:timeout:)

    /// A legal port yields a GET with the exact bearer header, URL, timeout
    /// and cache policy.
    func testRequestCarriesBearerHeader() {
        let request = ShellStatusClient.request(port: 8000, token: token, timeout: 2)
        XCTAssertNotNil(request)
        guard let request else { return }
        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(
            request.value(forHTTPHeaderField: "Authorization"),
            "Bearer " + token
        )
        XCTAssertEqual(
            request.url?.absoluteString,
            "http://127.0.0.1:8000/api/shell/status"
        )
        XCTAssertEqual(request.timeoutInterval, 2)
        XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData)
    }

    /// An invalid port (0) yields nil.
    func testRequestInvalidPortIsNil() {
        XCTAssertNil(ShellStatusClient.request(port: 0, token: token, timeout: 2))
    }

    // MARK: - decode(statusCode:body:)

    /// The fixture decodes to the exact ShellStatus, in wire order.
    func testDecodeFixture() {
        let body = Data(Self.fixture.utf8)
        let status = ShellStatusClient.decode(statusCode: 200, body: body)
        XCTAssertEqual(
            status,
            ShellStatus(
                loadedModels: [
                    LoadedModel(name: "Llama 3 8B", backend: "mlx", bytes: 6_100_000_000),
                    LoadedModel(name: "Qwen 2.5 14B", backend: "vllm", bytes: 0),
                ],
                usedBytes: 6_100_000_000,
                usableBytes: 21_300_000_000,
                inFlight: [
                    InFlightGroup(who: "web", model: "Llama 3 8B", requests: 3),
                    InFlightGroup(who: "unknown", model: "Qwen 2.5 14B", requests: 1),
                ]
            )
        )
        XCTAssertEqual(status?.inFlightCount, 4)
    }

    /// A JSON-null model `bytes` maps to 0.
    func testDecodeNullModelBytesBecomesZero() {
        let json = #"""
        {"models":[{"name":"M","backend":"mlx","bytes":null}],"memory":{"used_bytes":0,"usable_bytes":null},"in_flight":{"count":0,"groups":[]}}
        """#
        let status = ShellStatusClient.decode(
            statusCode: 200, body: Data(json.utf8))
        XCTAssertEqual(status?.loadedModels, [LoadedModel(name: "M", backend: "mlx", bytes: 0)])
    }

    /// A JSON-null memory `usable_bytes` maps to 0.
    func testDecodeNullUsableBytesBecomesZero() {
        let json = #"""
        {"models":[],"memory":{"used_bytes":0,"usable_bytes":null},"in_flight":{"count":0,"groups":[]}}
        """#
        let status = ShellStatusClient.decode(
            statusCode: 200, body: Data(json.utf8))
        XCTAssertEqual(status?.usableBytes, 0)
    }

    /// A JSON-null `pressure_level` is ignored (still decodes).
    func testDecodeNullPressureLevelIsIgnored() {
        let json = #"""
        {"models":[],"memory":{"used_bytes":0,"usable_bytes":null,"pressure_level":null},"in_flight":{"count":0,"groups":[]}}
        """#
        let status = ShellStatusClient.decode(
            statusCode: 200, body: Data(json.utf8))
        XCTAssertEqual(status?.usedBytes, 0)
        XCTAssertEqual(status?.usableBytes, 0)
    }

    /// Empty models and groups decode to empty arrays with 0/0 memory.
    func testDecodeEmptyModelsAndGroups() {
        let json = #"""
        {"models":[],"memory":{"used_bytes":0,"usable_bytes":null,"pressure_level":null},"in_flight":{"count":0,"groups":[]}}
        """#
        let status = ShellStatusClient.decode(
            statusCode: 200, body: Data(json.utf8))
        XCTAssertEqual(status?.loadedModels, [])
        XCTAssertEqual(status?.inFlight, [])
        XCTAssertEqual(status?.usedBytes, 0)
        XCTAssertEqual(status?.usableBytes, 0)
    }

    /// Every non-200 status yields nil, even with the good fixture body.
    func testDecodeNon200IsNil() {
        let body = Data(Self.fixture.utf8)
        for code in [199, 201, 401, 404, 500] {
            XCTAssertNil(
                ShellStatusClient.decode(statusCode: code, body: body),
                "status \(code) must be nil"
            )
        }
    }

    /// Malformed JSON yields nil.
    func testDecodeMalformedJsonIsNil() {
        XCTAssertNil(
            ShellStatusClient.decode(statusCode: 200, body: Data("not json".utf8))
        )
    }

    /// A missing `memory` key yields nil.
    func testDecodeMissingMemoryIsNil() {
        let json = #"""
        {"models":[],"in_flight":{"count":0,"groups":[]}}
        """#
        XCTAssertNil(
            ShellStatusClient.decode(statusCode: 200, body: Data(json.utf8))
        )
    }

    /// A wrong type (`used_bytes` as a string) yields nil.
    func testDecodeWrongTypeIsNil() {
        let json = #"""
        {"models":[],"memory":{"used_bytes":"1","usable_bytes":null},"in_flight":{"count":0,"groups":[]}}
        """#
        XCTAssertNil(
            ShellStatusClient.decode(statusCode: 200, body: Data(json.utf8))
        )
    }

    // MARK: - fetchStatus(completion:)

    /// A valid port and token issue exactly one GET carrying the exact bearer.
    func testFetchStatusSendsOneRequestWithBearer() {
        var recorded: [URLRequest] = []
        let client = ShellStatusClient(
            port: 8000, token: token,
            fetch: { request, _ in
                recorded.append(request)
            }
        )
        client.fetchStatus { _ in }
        XCTAssertEqual(recorded.count, 1)
        XCTAssertEqual(
            recorded.first?.value(forHTTPHeaderField: "Authorization"),
            "Bearer " + token
        )
        XCTAssertEqual(recorded.first?.httpMethod, "GET")
        XCTAssertEqual(
            recorded.first?.url?.absoluteString,
            "http://127.0.0.1:8000/api/shell/status"
        )
        // The default timeout (2 s) reaches the request fetchStatus sends.
        XCTAssertEqual(recorded.first?.timeoutInterval, 2)
    }

    /// A 200 fixture body decodes to the exact ShellStatus.
    func testFetchStatusDecodesFixture() {
        let response = HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:8000/api/shell/status")!,
            statusCode: 200, httpVersion: nil, headerFields: nil
        )
        let client = ShellStatusClient(
            port: 8000, token: token,
            fetch: { _, callback in
                callback(Data(Self.fixture.utf8), response, nil)
            }
        )
        var result: ShellStatus?
        client.fetchStatus { result = $0 }
        XCTAssertEqual(result?.loadedModels.count, 2)
        XCTAssertEqual(result?.usedBytes, 6_100_000_000)
        XCTAssertEqual(result?.usableBytes, 21_300_000_000)
        XCTAssertEqual(result?.inFlightCount, 4)
    }

    /// A transport error yields nil.
    func testFetchStatusTransportErrorIsNil() {
        let client = ShellStatusClient(
            port: 8000, token: token,
            fetch: { _, callback in
                callback(nil, nil, URLError(.notConnectedToInternet))
            }
        )
        var result: ShellStatus? = ShellStatus(
            loadedModels: [], usedBytes: 0, usableBytes: 0, inFlight: []
        )
        client.fetchStatus { result = $0 }
        XCTAssertNil(result)
    }

    /// A transport error yields nil even when a 200 response and a good body
    /// arrive alongside it (the error wins).
    func testFetchStatusTransportErrorWithResponseIsNil() {
        let response = HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:8000/api/shell/status")!,
            statusCode: 200, httpVersion: nil, headerFields: nil
        )
        let client = ShellStatusClient(
            port: 8000, token: token,
            fetch: { _, callback in
                callback(Data(Self.fixture.utf8), response, URLError(.timedOut))
            }
        )
        var result: ShellStatus? = ShellStatus(
            loadedModels: [], usedBytes: 0, usableBytes: 0, inFlight: []
        )
        client.fetchStatus { result = $0 }
        XCTAssertNil(result)
    }

    /// Printing, debug-printing or dumping a client never shows the token.
    func testDescriptionRedactsToken() {
        let client = ShellStatusClient(port: 8000, token: token, fetch: { _, _ in })
        var dumped = ""
        dump(client, to: &dumped)
        for text in [String(describing: client), String(reflecting: client), "\(client)", dumped] {
            XCTAssertFalse(text.contains(token), "token leaked in a rendering of the client")
            XCTAssertTrue(text.contains("<redacted>"))
        }
    }

    /// A nil response (with no error) yields nil.
    func testFetchStatusNoResponseIsNil() {
        let client = ShellStatusClient(
            port: 8000, token: token,
            fetch: { _, callback in
                callback(Data(Self.fixture.utf8), nil, nil)
            }
        )
        var result: ShellStatus? = ShellStatus(
            loadedModels: [], usedBytes: 0, usableBytes: 0, inFlight: []
        )
        client.fetchStatus { result = $0 }
        XCTAssertNil(result)
    }

    /// An invalid port (0) short-circuits without any fetch call.
    func testFetchStatusInvalidPortSkipsRequest() {
        var calls = 0
        let client = ShellStatusClient(
            port: 0, token: token,
            fetch: { _, callback in
                calls += 1
                callback(Data(Self.fixture.utf8), nil, nil)
            }
        )
        var result: ShellStatus?
        client.fetchStatus { result = $0 }
        XCTAssertEqual(calls, 0)
        XCTAssertNil(result)
    }

    /// An empty token short-circuits without any fetch call.
    func testFetchStatusEmptyTokenSkipsRequest() {
        var calls = 0
        let client = ShellStatusClient(
            port: 8000, token: "",
            fetch: { _, callback in
                calls += 1
                callback(Data(Self.fixture.utf8), nil, nil)
            }
        )
        var result: ShellStatus?
        client.fetchStatus { result = $0 }
        XCTAssertEqual(calls, 0)
        XCTAssertNil(result)
    }
}
