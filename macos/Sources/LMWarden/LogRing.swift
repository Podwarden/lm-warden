// LogRing — bounded line buffer for the uvicorn child's stdout/stderr.
//
// The ServerSupervisor (Metal design §2) pipes the child's output into this
// buffer so an operator can inspect the most recent lines. It is intentionally
// free of any Foundation timers, actors, or async: the supervisor owns the
// reading and the flushing; this type owns only the line-splitting and the
// drop-oldest bound.
//
// A multi-byte UTF-8 character can be split across two `append` chunks, so the
// bytes after the last newline are held as `Data` (not decoded) until a later
// chunk completes the line.
import Foundation

/// Bounded line buffer for the uvicorn child's stdout/stderr.
///
/// `append` splits on `0x0A`, drops one trailing `0x0D` per line, and decodes
/// each complete line as UTF-8. Bytes after the last newline are held as
/// pending `Data` until a later chunk (or `flush`) completes the line. When the
/// lines would exceed `capacity` the oldest are removed and `droppedCount`
/// grows by the number removed.
struct LogRing: Equatable {
    private let capacity: Int
    private(set) var lines: [String]
    private(set) var droppedCount: Int
    private var pending: Data

    init(capacity: Int) {
        self.capacity = max(capacity, 1)
        self.lines = []
        self.droppedCount = 0
        self.pending = Data()
    }

    /// Fold a chunk of bytes into the ring. Bytes after the last newline are
    /// held as pending and NOT decoded until a later chunk completes the line,
    /// so a multi-byte UTF-8 character split across chunks decodes intact.
    mutating func append(_ chunk: Data) {
        pending.append(chunk)
        while let index = pending.firstIndex(of: 0x0A) {
            let lineData = pending.subdata(in: pending.startIndex..<index)
            pending.removeSubrange(pending.startIndex...index)
            addLine(lineData)
        }
    }

    /// Emit the pending bytes as a line if non-empty, then clear them. A flush
    /// with nothing pending adds nothing.
    mutating func flush() {
        if !pending.isEmpty {
            addLine(pending)
            pending = Data()
        }
    }

    /// Decode one line (dropping a single trailing `0x0D` if present) and push
    /// it into the ring, dropping the oldest to stay within `capacity`.
    private mutating func addLine(_ lineData: Data) {
        var body = lineData
        if lineData.last == 0x0D {
            body.removeLast()
        }
        lines.append(String(decoding: body, as: UTF8.self))
        if lines.count > capacity {
            let overflow = lines.count - capacity
            lines.removeFirst(overflow)
            droppedCount += overflow
        }
    }
}
