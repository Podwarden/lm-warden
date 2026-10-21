// ShellStatusClient — fetches the local control plane's /api/shell/status feed.
//
// The control plane (app/main.py) answers GET /api/shell/status with a 200
// whose body is the JSON produced by build_shell_status (snake_case keys).
// The client authenticates with "Authorization: Bearer <token>" (one space).
// Any failure — bad port, empty token, transport error, non-200 status, or a
// body that does not decode to the exact feed shape — is nil.
//
// `fetchStatus(completion:)` is transport-injectable: the `fetch` closure
// performs the GET and calls back with (Data?, HTTPURLResponse?, Error?).
// When no `fetch` is given, the client runs `session.dataTask`. The token is
// carried only in the request header and never in any log or error message.
import Foundation

// MARK: - Private wire shapes

/// The top-level feed body, with the exact keys the Python models emit.
/// Extra keys the server may send (id, state, pressure_level, count) are
/// simply absent here and are ignored by JSONDecoder.
private struct StatusFeed: Decodable {
    let models: [ModelItem]
    let memory: MemoryItem
    let inFlight: InFlightItem

    enum CodingKeys: String, CodingKey {
        case models
        case memory
        case inFlight = "in_flight"
    }
}

/// One loaded engine: name and backend are required, bytes may be null.
private struct ModelItem: Decodable {
    let name: String
    let backend: String
    let bytes: Int64?

    enum CodingKeys: String, CodingKey {
        case name
        case backend
        case bytes
    }
}

/// Used and usable memory; usable_bytes may be null, pressure_level is ignored.
private struct MemoryItem: Decodable {
    let usedBytes: Int64
    let usableBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case usedBytes = "used_bytes"
        case usableBytes = "usable_bytes"
    }
}

/// The in-flight section; count is ignored, groups is the per-(caller,model) list.
private struct InFlightItem: Decodable {
    let groups: [GroupItem]

    enum CodingKeys: String, CodingKey {
        case groups
    }
}

/// One in-flight group, keyed by caller and served model.
private struct GroupItem: Decodable {
    let who: String
    let model: String
    let requests: Int

    enum CodingKeys: String, CodingKey {
        case who
        case model
        case requests
    }
}

struct ShellStatusClient {
    let port: Int
    let token: String
    let timeout: TimeInterval
    let session: URLSession
    /// Performs the GET for `request` and calls `callback` with the
    /// (data, HTTP response, error) triple. The default closure runs
    /// `session.dataTask`.
    let fetch: (URLRequest, @escaping (Data?, HTTPURLResponse?, Error?) -> Void) -> Void

    init(
        port: Int,
        token: String,
        timeout: TimeInterval = 2,
        session: URLSession = .shared,
        fetch: ((URLRequest, @escaping (Data?, HTTPURLResponse?, Error?) -> Void) -> Void)? = nil
    ) {
        self.port = port
        self.token = token
        self.timeout = timeout
        self.session = session
        self.fetch = fetch ?? { request, callback in
            session.dataTask(with: request) { data, response, error in
                callback(data, response as? HTTPURLResponse, error)
            }.resume()
        }
    }

    /// http://127.0.0.1:<port>/api/shell/status for a legal port (1...65535),
    /// else nil.
    static func url(port: Int) -> URL? {
        guard (1...65535).contains(port) else {
            return nil
        }
        return URL(string: "http://127.0.0.1:\(port)/api/shell/status")
    }

    /// A GET request for the status feed with the bearer header, the given
    /// timeout and a no-cache policy. nil for an invalid port.
    static func request(port: Int, token: String, timeout: TimeInterval) -> URLRequest? {
        guard let url = ShellStatusClient.url(port: port) else {
            return nil
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalCacheData
        return request
    }

    /// Decodes a 200 body into a ShellStatus. nil unless the status is 200
    /// AND the body decodes to the exact feed shape; a null model bytes or
    /// usable_bytes maps to 0, and a non-200 status, malformed JSON, a missing
    /// required key or a wrong type all yield nil.
    static func decode(statusCode: Int, body: Data) -> ShellStatus? {
        guard statusCode == 200 else {
            return nil
        }
        guard let feed = try? JSONDecoder().decode(StatusFeed.self, from: body) else {
            return nil
        }
        let loadedModels: [LoadedModel] = feed.models.map {
            LoadedModel(name: $0.name, backend: $0.backend, bytes: $0.bytes ?? 0)
        }
        let inFlight: [InFlightGroup] = feed.inFlight.groups.map {
            InFlightGroup(who: $0.who, model: $0.model, requests: $0.requests)
        }
        return ShellStatus(
            loadedModels: loadedModels,
            usedBytes: feed.memory.usedBytes,
            usableBytes: feed.memory.usableBytes ?? 0,
            inFlight: inFlight
        )
    }

    /// GETs the status feed and decodes the outcome. An empty token or an
    /// invalid port (no request) short-circuits to completion(nil) WITHOUT
    /// calling fetch; otherwise exactly one fetch call. A transport error or a
    /// nil response yields nil; otherwise the outcome is
    /// decode(statusCode:body:). The token never reaches any log or error.
    func fetchStatus(completion: @escaping (ShellStatus?) -> Void) {
        guard !token.isEmpty, let request = ShellStatusClient.request(
            port: port, token: token, timeout: timeout
        ) else {
            completion(nil)
            return
        }
        fetch(request) { data, response, error in
            if error != nil {
                completion(nil)
                return
            }
            guard let response else {
                completion(nil)
                return
            }
            completion(ShellStatusClient.decode(
                statusCode: response.statusCode, body: data ?? Data()))
        }
    }
}

// The token is a bearer secret: a client printed, interpolated or dumped
// (String(describing:), String(reflecting:), dump, an XCTest failure message)
// shows it redacted, never the value.
extension ShellStatusClient: CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    var description: String {
        "ShellStatusClient(port: \(port), token: <redacted>, timeout: \(timeout))"
    }

    var debugDescription: String { description }

    var customMirror: Mirror {
        Mirror(self, children: ["port": port, "token": "<redacted>", "timeout": timeout])
    }
}
