import Foundation
import XCTest
@testable import LMWarden

final class SystemSignalsTests: XCTestCase {
    func testThermalNominal() {
        XCTAssertEqual(SystemSignals.thermal(.nominal), .nominal)
    }

    func testThermalFair() {
        XCTAssertEqual(SystemSignals.thermal(.fair), .fair)
    }

    func testThermalSerious() {
        XCTAssertEqual(SystemSignals.thermal(.serious), .serious)
    }

    func testThermalCritical() {
        XCTAssertEqual(SystemSignals.thermal(.critical), .critical)
    }

    func testBatteryPowerIsOnBattery() {
        XCTAssertTrue(SystemSignals.onBattery(powerSourceState: "Battery Power"))
    }

    func testACPowerIsNotOnBattery() {
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: "AC Power"))
    }

    func testNilIsNotOnBattery() {
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: nil))
    }

    func testUPSPowerIsNotOnBattery() {
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: "UPS Power"))
    }

    func testNearMissesAreNotOnBattery() {
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: ""))
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: "battery power"))
        XCTAssertFalse(SystemSignals.onBattery(powerSourceState: "Battery Power "))
    }
}
