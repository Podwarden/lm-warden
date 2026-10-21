// HealthProbe — polls the local vLLM control plane's /healthz liveness route.
//
// The control plane (app/main.py) answers GET /healthz with a 200 whose body
// is a JSON object carrying "ok": true while it is up. A probe is healthy only
// when the status is 200 AND the body decodes to a JSON object with "ok"
// Bool true; anything else (other status, non-JSON, array, missing "ok",
// "ok": false) is unhealthy with a reason string the supervisor can surface.
//
// `check(completion:)` is transport-injectable: the `fetch` closure performs
// the GET and calls back with (Data?, HTTPURLResponse?, Error?). When no
// `fetch` is given, the probe runs `session.dataTask`. Tests inject a fake
// `fetch` so no real network and no URLProtocol stub is involved.
import Foundation

enum HealthResult: Equatable {
    case healthy
    case unhealthy(String)
}

struct HealthProbe {
    let port: Int
    let timeout: TimeInterval
    let session: URLSession
    /// Performs the GET for `request` and calls `callback` with the
    /// (data, HTTP response, error) triple. The default closure runs
    /// `session.dataTask`.
    let fetch: (URLRequest, @escaping (Data?, HTTPURLResponse?, Error?) -> Void) -> Void

    init(
        port: Int,
        timeout: TimeInterval = 2,
        session: URLSession = .shared,
        fetch: ((URLRequest, @escaping (Data?, HTTPURLResponse?, Error?) -> Void) -> Void)? = nil
    ) {
        self.port = port
        self.timeout = timeout
        self.session = session
        self.fetch = fetch ?? { request, callback in
            session.dataTask(with: request) { data, response, error in
                callback(data, response as? HTTPURLResponse, error)
            }.resume()
        }
    }

    /// http://127.0.0.1:<port>/healthz for a legal port (1...65535), else nil.
    static func url(port: Int) -> URL? {
        guard (1...65535).contains(port) else {
            return nil
        }
        return URL(string: "http://127.0.0.1:\(port)/healthz")
    }

    /// Healthy only for a 200 whose body is a JSON object with "ok" Bool true.
    /// Any other status is unhealthy("status <n>"); a 200 with a bad body is
    /// unhealthy("bad body").
    static func interpret(statusCode: Int, body: Data) -> HealthResult {
        guard statusCode == 200 else {
            return .unhealthy("status \(statusCode)")
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: body),
            let dict = object as? [String: Any],
            dict["ok"] as? Bool == true
        else {
            return .unhealthy("bad body")
        }
        return .healthy
    }

    /// GETs the /healthz URL and decodes the outcome to a HealthResult.
    /// An invalid port (no URL) short-circuits to .unhealthy("invalid port")
    /// without issuing any request. A transport error yields
    /// .unhealthy(error.localizedDescription); otherwise the outcome is
    /// interpret(statusCode:body:).
    func check(completion: @escaping (HealthResult) -> Void) {
        guard let requestURL = HealthProbe.url(port: port) else {
            completion(.unhealthy("invalid port"))
            return
        }
        var request = URLRequest(url: requestURL)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        request.cachePolicy = .reloadIgnoringLocalCacheData
        fetch(request) { data, response, error in
            if let error {
                completion(.unhealthy(error.localizedDescription))
                return
            }
            guard let response else {
                completion(.unhealthy("no response"))
                return
            }
            completion(HealthProbe.interpret(statusCode: response.statusCode, body: data ?? Data()))
        }
    }
}
