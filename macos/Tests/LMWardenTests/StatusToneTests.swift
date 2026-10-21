// StatusToneTests — the menu bar status dot's colour and spoken description
// for every server tone, as in the approved mockup screen 1 (Main.dc.html):
// running #1f8f3a, starting #b26b00, stopped #8e8e93, failed #c4302b.
import XCTest
@testable import LMWarden

final class StatusToneTests: XCTestCase {
    private func assertRGB(
        _ tone: StatusTone, _ r: Int, _ g: Int, _ b: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(tone.red, Double(r) / 255, accuracy: 1e-9, "red", file: file, line: line)
        XCTAssertEqual(tone.green, Double(g) / 255, accuracy: 1e-9, "green", file: file, line: line)
        XCTAssertEqual(tone.blue, Double(b) / 255, accuracy: 1e-9, "blue", file: file, line: line)
    }

    func testGreenIsRunning() {
        let tone = StatusTone(.green)
        assertRGB(tone, 0x1f, 0x8f, 0x3a)
        XCTAssertEqual(tone.hex, "#1f8f3a")
        XCTAssertEqual(tone.accessibilityDescription, "Server running")
    }

    func testAmberIsStarting() {
        let tone = StatusTone(.amber)
        assertRGB(tone, 0xb2, 0x6b, 0x00)
        XCTAssertEqual(tone.hex, "#b26b00")
        XCTAssertEqual(tone.accessibilityDescription, "Server starting")
    }

    func testGrayIsStopped() {
        let tone = StatusTone(.gray)
        assertRGB(tone, 0x8e, 0x8e, 0x93)
        XCTAssertEqual(tone.hex, "#8e8e93")
        XCTAssertEqual(tone.accessibilityDescription, "Server stopped")
    }

    func testRedIsFailed() {
        let tone = StatusTone(.red)
        assertRGB(tone, 0xc4, 0x30, 0x2b)
        XCTAssertEqual(tone.hex, "#c4302b")
        XCTAssertEqual(tone.accessibilityDescription, "Server failed")
    }

    func testEveryToneIsDistinct() {
        let tones: [MenuTone] = [.green, .amber, .gray, .red]
        let hexes = Set(tones.map { StatusTone($0).hex })
        let words = Set(tones.map { StatusTone($0).accessibilityDescription })
        XCTAssertEqual(hexes.count, tones.count)
        XCTAssertEqual(words.count, tones.count)
    }

    func testModelToneDrivesTheDot() {
        // The tone the menu bar model picks for each server state lands on the
        // matching dot (running/starting/backing off/stopped/failed).
        func tone(_ state: SupervisorState) -> StatusTone {
            StatusTone(MenuBarModel.make(MenuBarInput(
                state: state, port: 8080, shareOnLAN: false, lanAddress: nil,
                openAtLogin: false, canCheckForUpdates: true, status: nil
            )).tone)
        }
        XCTAssertEqual(tone(.running).accessibilityDescription, "Server running")
        XCTAssertEqual(tone(.starting).accessibilityDescription, "Server starting")
        XCTAssertEqual(tone(.backingOff(failures: 1, delay: 2)).accessibilityDescription, "Server starting")
        XCTAssertEqual(tone(.stopped).accessibilityDescription, "Server stopped")
        XCTAssertEqual(tone(.failed(failures: 5)).accessibilityDescription, "Server failed")
    }
}
