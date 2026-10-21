// LANAddress — picks the Mac's LAN IPv4 address for the menu and first-run copy.
//
// `pick` is a pure function over an injectable interface list so tests can
// exercise the preference rule without a live network stack. `systemInterfaces`
// is the production seam: it calls getifaddrs (AF_INET only) and converts each
// result to a `NetworkInterface` via getnameinfo.
import Foundation

struct NetworkInterface: Equatable {
    let name: String
    let ipv4: String
    let isUp: Bool
    let isLoopback: Bool
}

enum LANAddress {
    /// Pick the best LAN IPv4 address from `interfaces`.
    ///
    /// Filters out down, loopback, and 169.254.x.x link-local addresses.
    /// Preference order: en0 > en1 > any other en* > any other interface,
    /// each tier preserving input order. Returns nil when nothing qualifies.
    static func pick(_ interfaces: [NetworkInterface]) -> String? {
        let qualified = interfaces.enumerated().compactMap { offset, iface -> (offset: Int, rank: Int, ipv4: String)? in
            guard iface.isUp, !iface.isLoopback, !iface.ipv4.hasPrefix("169.254.") else {
                return nil
            }
            let rank: Int
            switch iface.name {
            case "en0": rank = 0
            case "en1": rank = 1
            default: rank = iface.name.hasPrefix("en") ? 2 : 3
            }
            return (offset, rank, iface.ipv4)
        }
        let sorted = qualified.sorted { lhs, rhs in
            lhs.rank == rhs.rank ? lhs.offset < rhs.offset : lhs.rank < rhs.rank
        }
        return sorted.first?.ipv4
    }

    /// Production seam: enumerate the real system's IPv4 interfaces via
    /// getifaddrs (AF_INET only). Returns an empty list if the syscall fails.
    static func systemInterfaces() -> [NetworkInterface] {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0 else { return [] }
        defer { freeifaddrs(head) }
        var result: [NetworkInterface] = []
        var cursor = head
        while let ifa = cursor {
            cursor = ifa.pointee.ifa_next
            guard let addr = ifa.pointee.ifa_addr,
                  addr.pointee.sa_family == sa_family_t(AF_INET) else {
                continue
            }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(addr, socklen_t(addr.pointee.sa_len),
                              &host, socklen_t(host.count),
                              nil, 0, NI_NUMERICHOST) == 0 else {
                continue
            }
            let flags = Int32(ifa.pointee.ifa_flags)
            result.append(NetworkInterface(
                name: String(cString: ifa.pointee.ifa_name),
                ipv4: host.withUnsafeBufferPointer { String(cString: $0.baseAddress!) },
                isUp: flags & IFF_UP != 0,
                isLoopback: flags & IFF_LOOPBACK != 0
            ))
        }
        return result
    }
}
