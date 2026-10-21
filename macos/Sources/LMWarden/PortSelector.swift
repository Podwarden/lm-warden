// PortSelector — uvicorn port choice for the ServerSupervisor shell
// (Metal design §2: WKWebView -> 127.0.0.1:8080/ui/, shell passes --port).
//
// The selector tries the preferred port first, then a fallback range in
// ascending order, and reports the first port an injectable availability
// check accepts. The production check is a real loopback bind probe:
// `bind` on 127.0.0.1 without SO_REUSEADDR, so a port held by another
// listener (or in TIME_WAIT) reads as unavailable.
import Foundation

/// Chooses a free local port: preferred first, then a fallback range.
///
/// The ServerSupervisor calls this before launching uvicorn with
/// `--port`; the WKWebView points at whatever port it returns.
struct PortSelector {
    static let portRange = 1...65535

    let isAvailable: (Int) -> Bool

    init(isAvailable: @escaping (Int) -> Bool = PortSelector.loopbackIsAvailable) {
        self.isAvailable = isAvailable
    }

    /// First candidate for which `isAvailable` is true: `preferred` first,
    /// then `fallback` in ascending order with the preferred port removed
    /// when it falls inside the range. Candidates outside 1...65535 are
    /// skipped without probing; nil when none is free.
    func choose(
        preferred: Int = 8080,
        fallback: ClosedRange<Int> = 8081...8099
    ) -> Int? {
        var candidates: [Int] = []
        if Self.portRange.contains(preferred) {
            candidates.append(preferred)
        }
        for port in fallback {
            if port == preferred { continue }
            guard Self.portRange.contains(port) else { continue }
            candidates.append(port)
        }
        for port in candidates where isAvailable(port) {
            return port
        }
        return nil
    }

    /// Real loopback probe: `bind` a stream socket to 127.0.0.1:<port> and
    /// report whether it succeeded. Ports outside 1...65535 are false
    /// without a syscall. SO_REUSEADDR is deliberately NOT set — a port in
    /// TIME_WAIT or held by another listener must read as unavailable.
    static func loopbackIsAvailable(_ port: Int) -> Bool {
        guard portRange.contains(port) else { return false }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr = in_addr(s_addr: INADDR_LOOPBACK.bigEndian)
        return withUnsafePointer(to: &addr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddr in
                bind(fd, sockaddr, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }
}
