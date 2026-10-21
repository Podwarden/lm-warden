import XCTest
@testable import LMWarden

final class LogRingTests: XCTestCase {
    /// Bytes after a newline become complete lines; bytes after the last
    /// newline are held as pending and not emitted.
    func testSplitsCompleteLines() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("a\nb".utf8))
        XCTAssertEqual(ring.lines, ["a"])
        XCTAssertEqual(ring.droppedCount, 0)
        ring.append(Data("c\nd".utf8))
        XCTAssertEqual(ring.lines, ["a", "bc"])
        XCTAssertEqual(ring.droppedCount, 0)
    }

    /// Bytes after the last newline are held as pending Data until a later
    /// chunk completes the line; the held bytes are NOT decoded prematurely.
    func testHoldsPartialLine() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("hello ".utf8))
        XCTAssertEqual(ring.lines, [])
        XCTAssertEqual(ring.droppedCount, 0)
        ring.append(Data("world\n".utf8))
        XCTAssertEqual(ring.lines, ["hello world"])
    }

    /// A multi-byte UTF-8 character split across two chunks decodes intact
    /// because the pending bytes are held as Data, not decoded per chunk.
    func testMultibyteSplitAcrossChunks() {
        var ring = LogRing(capacity: 10)
        // "é" is 0xC3 0xA9 in UTF-8; split between the two bytes.
        ring.append(Data([0x61, 0xC3]))
        XCTAssertEqual(ring.lines, [])
        ring.append(Data([0xA9, 0x0A]))
        XCTAssertEqual(ring.lines, ["aé"])
    }

    /// Each complete line drops one trailing 0x0D if present (CRLF), and a
    /// line that ends in 0x0A without a preceding 0x0D is left as-is.
    func testStripsCarriageReturn() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("a\r\nb\n".utf8))
        XCTAssertEqual(ring.lines, ["a", "b"])
        // Only ONE trailing 0x0D is dropped, and an interior 0x0D is kept.
        ring.append(Data("c\r\r\nd\re\n".utf8))
        XCTAssertEqual(ring.lines, ["a", "b", "c\r", "d\re"])
    }

    /// flush() emits the pending bytes as a line, then clears them.
    func testFlushEmitsPending() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("tail".utf8))
        XCTAssertEqual(ring.lines, [])
        ring.flush()
        XCTAssertEqual(ring.lines, ["tail"])
        XCTAssertEqual(ring.droppedCount, 0)
        // The pending bytes were cleared: a second flush adds nothing and the
        // next line does not carry the flushed tail.
        ring.flush()
        XCTAssertEqual(ring.lines, ["tail"])
        ring.append(Data("x\n".utf8))
        XCTAssertEqual(ring.lines, ["tail", "x"])
    }

    /// flush with nothing pending adds nothing.
    func testFlushWithNothingPending() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("a\n".utf8))
        XCTAssertEqual(ring.lines, ["a"])
        ring.flush()
        XCTAssertEqual(ring.lines, ["a"])
        XCTAssertEqual(ring.droppedCount, 0)
        ring.flush()
        XCTAssertEqual(ring.lines, ["a"])
    }

    /// Exactly `capacity` lines are kept with nothing dropped (the exact
    /// bound, not past it).
    func testCapacityBoundExact() {
        var ring = LogRing(capacity: 3)
        ring.append(Data("1\n2\n3\n".utf8))
        XCTAssertEqual(ring.lines, ["1", "2", "3"])
        XCTAssertEqual(ring.droppedCount, 0)
    }

    /// `capacity + 1` lines drops the oldest: the first line is gone and
    /// `droppedCount` is 1 (just past the bound).
    func testCapacityPlusOneDropsOldest() {
        var ring = LogRing(capacity: 3)
        ring.append(Data("1\n2\n3\n4\n".utf8))
        XCTAssertEqual(ring.lines, ["2", "3", "4"])
        XCTAssertEqual(ring.droppedCount, 1)
        // droppedCount accumulates across later overflows.
        ring.append(Data("5\n".utf8))
        XCTAssertEqual(ring.lines, ["3", "4", "5"])
        XCTAssertEqual(ring.droppedCount, 2)
    }

    /// A capacity below 1 is clamped to 1, so at least one line is always
    /// kept and overflow past one drops the oldest.
    func testCapacityClampedToOne() {
        var ring = LogRing(capacity: 0)
        ring.append(Data("a\n".utf8))
        XCTAssertEqual(ring.lines, ["a"])
        XCTAssertEqual(ring.droppedCount, 0)
        ring.append(Data("b\n".utf8))
        XCTAssertEqual(ring.lines, ["b"])
        XCTAssertEqual(ring.droppedCount, 1)
        // A negative capacity is clamped to 1 as well.
        var negative = LogRing(capacity: -5)
        negative.append(Data("x\ny\n".utf8))
        XCTAssertEqual(negative.lines, ["y"])
        XCTAssertEqual(negative.droppedCount, 1)
    }

    /// Empty lines are kept; "a\n\nb\n" yields ["a", "", "b"].
    func testKeepsEmptyLines() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("a\n\nb\n".utf8))
        XCTAssertEqual(ring.lines, ["a", "", "b"])
        XCTAssertEqual(ring.droppedCount, 0)
    }

    /// An empty chunk is a no-op: it changes neither lines nor droppedCount.
    func testEmptyChunkNoOp() {
        var ring = LogRing(capacity: 10)
        ring.append(Data("a\n".utf8))
        XCTAssertEqual(ring.lines, ["a"])
        ring.append(Data())
        XCTAssertEqual(ring.lines, ["a"])
        XCTAssertEqual(ring.droppedCount, 0)
    }
}
