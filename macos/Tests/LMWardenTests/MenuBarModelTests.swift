import XCTest
@testable import LMWarden

final class MenuBarModelTests: XCTestCase {

    // MARK: - ShellFormat

    func testGigabytesZero() {
        XCTAssertEqual(ShellFormat.gigabytes(0), "0.0 GB")
    }

    func testGigabytesSixPointOne() {
        XCTAssertEqual(ShellFormat.gigabytes(6_100_000_000), "6.1 GB")
    }

    func testGigabytesJustUnderOnePointFive() {
        // 1 049 999 999 / 1e9 = 1.049999999 -> %.1f rounds to 1.0
        XCTAssertEqual(ShellFormat.gigabytes(1_049_999_999), "1.0 GB")
    }

    func testGigabytesJustOverOnePointFive() {
        // 1 050 000 001 / 1e9 = 1.050000001 -> %.1f rounds to 1.1
        XCTAssertEqual(ShellFormat.gigabytes(1_050_000_001), "1.1 GB")
    }

    func testGigabytesNegativeClampedToZero() {
        XCTAssertEqual(ShellFormat.gigabytes(-5), "0.0 GB")
    }

    func testGigabytesNumberZero() {
        XCTAssertEqual(ShellFormat.gigabytesNumber(0), "0.0")
    }

    func testGigabytesNumberJustUnderOnePointFive() {
        XCTAssertEqual(ShellFormat.gigabytesNumber(1_049_999_999), "1.0")
    }

    func testGigabytesNumberJustOverOnePointFive() {
        XCTAssertEqual(ShellFormat.gigabytesNumber(1_050_000_001), "1.1")
    }

    func testWholeGigabytesFourHundredTwelve() {
        // 412 999 999 999 / 1e9 = 412.999999999 -> floor = 412
        XCTAssertEqual(ShellFormat.wholeGigabytes(412_999_999_999), "412 GB")
    }

    func testWholeGigabytesNegative() {
        XCTAssertEqual(ShellFormat.wholeGigabytes(-5), "0 GB")
    }

    func testWholeGigabytesZero() {
        XCTAssertEqual(ShellFormat.wholeGigabytes(0), "0 GB")
    }

    func testCountZero() {
        XCTAssertEqual(ShellFormat.count(0, "request", "requests"), "0 requests")
    }

    func testCountOne() {
        XCTAssertEqual(ShellFormat.count(1, "request", "requests"), "1 request")
    }

    func testCountTwo() {
        XCTAssertEqual(ShellFormat.count(2, "request", "requests"), "2 requests")
    }

    // MARK: - State headers and server-action titles

    func testRunningHeader() {
        let model = MenuBarModel.make(input(for: .running))
        XCTAssertEqual(model.statusWord, "Running")
        XCTAssertEqual(model.tone, .green)
        // Open Dashboard and Copy API Address must be enabled when running
        let section0 = model.sections[0]
        let openDashboard = section0.first { $0.action == .openDashboard }
        XCTAssertNotNil(openDashboard)
        XCTAssertEqual(openDashboard?.enabled, true)
        let copyAPI = section0.first { $0.action == .copyAPIAddress }
        XCTAssertNotNil(copyAPI)
        XCTAssertEqual(copyAPI?.enabled, true)
        // Server action is "Stop Server"
        let stopItem = model.sections[1].first { $0.action == .stopServer }
        XCTAssertNotNil(stopItem)
        XCTAssertEqual(stopItem?.title, "Stop Server")
        let startItem = model.sections[1].first { $0.action == .startServer }
        XCTAssertNil(startItem)
    }

    func testStartingHeader() {
        let model = MenuBarModel.make(input(for: .starting))
        XCTAssertEqual(model.statusWord, "Starting")
        XCTAssertEqual(model.tone, .amber)
        XCTAssertEqual(model.statusLine, "Starting the server and checking its health…")
        // Open Dashboard disabled when not running
        let openDashboard = model.sections[0].first { $0.action == .openDashboard }
        XCTAssertEqual(openDashboard?.enabled, false)
        let copyAPI = model.sections[0].first { $0.action == .copyAPIAddress }
        XCTAssertEqual(copyAPI?.enabled, false)
        // Server action is "Stop Server" during starting
        let stopItem = model.sections[1].first { $0.action == .stopServer }
        XCTAssertNotNil(stopItem)
        XCTAssertEqual(stopItem?.title, "Stop Server")
    }

    func testBackingOffHeader() {
        let model = MenuBarModel.make(input(for: .backingOff(failures: 2, delay: 4)))
        XCTAssertEqual(model.statusWord, "Starting")
        XCTAssertEqual(model.tone, .amber)
        XCTAssertEqual(model.statusLine, "Starting the server and checking its health…")
        // Server action is "Stop Server" during backingOff
        let stopItem = model.sections[1].first { $0.action == .stopServer }
        XCTAssertNotNil(stopItem)
        XCTAssertEqual(stopItem?.title, "Stop Server")
    }

    func testStoppedHeader() {
        let model = MenuBarModel.make(input(for: .stopped))
        XCTAssertEqual(model.statusWord, "Stopped")
        XCTAssertEqual(model.tone, .gray)
        XCTAssertEqual(model.statusLine, "The server is not running. Models are unloaded.")
        // Open Dashboard disabled
        let openDashboard = model.sections[0].first { $0.action == .openDashboard }
        XCTAssertEqual(openDashboard?.enabled, false)
        let copyAPI = model.sections[0].first { $0.action == .copyAPIAddress }
        XCTAssertEqual(copyAPI?.enabled, false)
        // Server action is "Start Server"
        let startItem = model.sections[1].first { $0.action == .startServer }
        XCTAssertNotNil(startItem)
        XCTAssertEqual(startItem?.title, "Start Server")
        let stopItem = model.sections[1].first { $0.action == .stopServer }
        XCTAssertNil(stopItem)
    }

    func testFailedHeader() {
        let model = MenuBarModel.make(input(for: .failed(failures: 5)))
        XCTAssertEqual(model.statusWord, "Failed")
        XCTAssertEqual(model.tone, .red)
        XCTAssertEqual(model.statusLine, "The server could not stay up.")
        // Open Dashboard disabled
        let openDashboard = model.sections[0].first { $0.action == .openDashboard }
        XCTAssertEqual(openDashboard?.enabled, false)
        // Server action is "Start Server"
        let startItem = model.sections[1].first { $0.action == .startServer }
        XCTAssertNotNil(startItem)
        XCTAssertEqual(startItem?.title, "Start Server")
    }

    // MARK: - Status line (running only)

    func testRunningLineNoShare() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: false, lanAddress: nil, port: 8080))
        XCTAssertEqual(model.statusLine, "Serving on this Mac only · 127.0.0.1:8080")
    }

    func testRunningLineShareWithAddress() {
        let model = MenuBarModel.make(
            input(for: .running, shareOnLAN: true, lanAddress: "192.168.1.50", port: 8080)
        )
        XCTAssertEqual(model.statusLine, "Serving on your network · 192.168.1.50:8080")
    }

    func testRunningLineShareWithoutAddress() {
        let model = MenuBarModel.make(
            input(for: .running, shareOnLAN: true, lanAddress: nil, port: 8080)
        )
        XCTAssertEqual(model.statusLine, "Serving on your network · port 8080")
    }

    // MARK: - Address and copy text

    func testAddressNoShare() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: false, lanAddress: "10.0.0.1", port: 9000))
        XCTAssertEqual(model.address, "127.0.0.1:9000")
        XCTAssertEqual(model.copyText, "http://127.0.0.1:9000/v1")
    }

    func testAddressShareWithAddress() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: true, lanAddress: "10.0.0.1", port: 9000))
        XCTAssertEqual(model.address, "10.0.0.1:9000")
        XCTAssertEqual(model.copyText, "http://10.0.0.1:9000/v1")
    }

    func testAddressShareWithoutAddress() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: true, lanAddress: nil, port: 9000))
        XCTAssertEqual(model.address, "127.0.0.1:9000")
        XCTAssertEqual(model.copyText, "http://127.0.0.1:9000/v1")
    }

    // MARK: - Models block

    func testModelsBlockAbsentForNilStatus() {
        let model = MenuBarModel.make(input(for: .running, status: nil))
        XCTAssertNil(model.models)
    }

    func testModelsBlockAbsentForStopped() {
        let status = ShellStatus(loadedModels: [LoadedModel(name: "m", backend: "b", bytes: 0)], usedBytes: 0, usableBytes: 10, inFlight: [])
        let model = MenuBarModel.make(input(for: .stopped, status: status))
        XCTAssertNil(model.models)
    }

    func testModelsBlockAbsentForStarting() {
        let status = ShellStatus(loadedModels: [LoadedModel(name: "m", backend: "b", bytes: 0)], usedBytes: 0, usableBytes: 10, inFlight: [])
        let model = MenuBarModel.make(input(for: .starting, status: status))
        XCTAssertNil(model.models)
    }

    func testModelsBlockAbsentForFailed() {
        let status = ShellStatus(loadedModels: [LoadedModel(name: "m", backend: "b", bytes: 0)], usedBytes: 0, usableBytes: 10, inFlight: [])
        let model = MenuBarModel.make(input(for: .failed(failures: 3), status: status))
        XCTAssertNil(model.models)
    }

    func testModelsBlockAbsentForBackingOff() {
        let status = ShellStatus(loadedModels: [LoadedModel(name: "m", backend: "b", bytes: 0)], usedBytes: 0, usableBytes: 10, inFlight: [])
        let model = MenuBarModel.make(input(for: .backingOff(failures: 1, delay: 1), status: status))
        XCTAssertNil(model.models)
    }

    func testModelsBlockWithModels() {
        let status = ShellStatus(
            loadedModels: [
                LoadedModel(name: "Llama 3 8B", backend: "MLX", bytes: 6_100_000_000),
                LoadedModel(name: "Qwen 2.5 14B", backend: "vLLM", bytes: 12_300_000_000),
            ],
            usedBytes: 8_500_000_000,
            usableBytes: 21_300_000_000,
            inFlight: [InFlightGroup(who: "web", model: "Llama 3 8B", requests: 3)]
        )
        let model = MenuBarModel.make(input(for: .running, status: status))
        let block = model.models
        XCTAssertNotNil(block)
        XCTAssertEqual(block?.rows.count, 2)
        XCTAssertEqual(block?.rows[0].title, "Llama 3 8B · MLX")
        XCTAssertEqual(block?.rows[0].size, "6.1 GB")
        XCTAssertEqual(block?.rows[1].title, "Qwen 2.5 14B · vLLM")
        XCTAssertEqual(block?.rows[1].size, "12.3 GB")
        XCTAssertEqual(block?.memoryLine, "8.5 of 21.3 GB usable for models · 3 requests in flight")
    }

    func testModelsBlockEmptyModels() {
        let status = ShellStatus(loadedModels: [], usedBytes: 0, usableBytes: 21_300_000_000, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        let block = model.models
        XCTAssertNotNil(block)
        XCTAssertEqual(block?.rows.count, 1)
        XCTAssertEqual(block?.rows[0].title, "No models loaded")
        XCTAssertEqual(block?.rows[0].size, "")
    }

    func testMemoryFractionUsedEqualsUsable() {
        let status = ShellStatus(loadedModels: [], usedBytes: 10_000_000_000, usableBytes: 10_000_000_000, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryFraction, 1.0)
    }

    func testMemoryFractionUsedGreaterThanUsable() {
        let status = ShellStatus(loadedModels: [], usedBytes: 20_000_000_000, usableBytes: 10_000_000_000, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryFraction, 1.0)
    }

    func testMemoryFractionUsableZero() {
        let status = ShellStatus(loadedModels: [], usedBytes: 5_000_000_000, usableBytes: 0, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryFraction, 0.0)
    }

    func testInFlightCountSumsRequests() {
        let status = ShellStatus(
            loadedModels: [],
            usedBytes: 0,
            usableBytes: 10_000_000_000,
            inFlight: [
                InFlightGroup(who: "a", model: "m1", requests: 2),
                InFlightGroup(who: "b", model: "m2", requests: 3),
            ]
        )
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(status.inFlightCount, 5)
        XCTAssertEqual(model.models?.memoryLine, "0.0 of 10.0 GB usable for models · 5 requests in flight")
    }

    func testInFlightCountZeroSingular() {
        let status = ShellStatus(loadedModels: [], usedBytes: 0, usableBytes: 10_000_000_000, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryLine, "0.0 of 10.0 GB usable for models · 0 requests in flight")
    }

    func testInFlightCountOneSingular() {
        let status = ShellStatus(
            loadedModels: [], usedBytes: 0, usableBytes: 10_000_000_000,
            inFlight: [InFlightGroup(who: "a", model: "m", requests: 1)]
        )
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryLine, "0.0 of 10.0 GB usable for models · 1 request in flight")
    }

    // MARK: - Failure block

    func testFailureBlockFiveFailures() {
        let model = MenuBarModel.make(input(for: .failed(failures: 5)))
        XCTAssertNotNil(model.failure)
        XCTAssertEqual(model.failure?.message, "The server stopped 5 times in a row and LM Warden gave up restarting it.")
        XCTAssertEqual(model.failure?.showLog.title, "Show log…")
        XCTAssertEqual(model.failure?.showLog.action, .showLog)
        XCTAssertEqual(model.failure?.showLog.enabled, true)
    }

    func testFailureBlockOneFailure() {
        let model = MenuBarModel.make(input(for: .failed(failures: 1)))
        XCTAssertEqual(model.failure?.message, "The server stopped 1 time in a row and LM Warden gave up restarting it.")
    }

    func testFailureBlockAbsentForOtherStates() {
        XCTAssertNil(MenuBarModel.make(input(for: .running)).failure)
        XCTAssertNil(MenuBarModel.make(input(for: .stopped)).failure)
        XCTAssertNil(MenuBarModel.make(input(for: .starting)).failure)
    }

    // MARK: - Check for Updates

    func testCheckForUpdatesDisabled() {
        let model = MenuBarModel.make(input(for: .running, canCheckForUpdates: false))
        let item = model.sections[1].first { $0.action == .checkForUpdates }
        XCTAssertEqual(item?.enabled, false)
    }

    func testCheckForUpdatesEnabled() {
        let model = MenuBarModel.make(input(for: .running, canCheckForUpdates: true))
        let item = model.sections[1].first { $0.action == .checkForUpdates }
        XCTAssertEqual(item?.enabled, true)
    }

    // MARK: - Section structure

    func testSectionCountAndOrder() {
        let model = MenuBarModel.make(input(for: .running))
        XCTAssertEqual(model.sections.count, 3)

        // Section 0: Open Dashboard, Copy API Address, Share on LAN
        XCTAssertEqual(model.sections[0].count, 3)
        XCTAssertEqual(model.sections[0][0].action, .openDashboard)
        XCTAssertEqual(model.sections[0][0].keyEquivalent, "d")
        XCTAssertEqual(model.sections[0][1].action, .copyAPIAddress)
        XCTAssertEqual(model.sections[0][1].detail, model.address)
        XCTAssertEqual(model.sections[0][2].action, .toggleShareOnLAN)
        XCTAssertEqual(model.sections[0][2].checked, false)

        // Section 1: Stop Server, Open at Login, Check for Updates, About
        XCTAssertEqual(model.sections[1].count, 4)
        XCTAssertEqual(model.sections[1][0].action, .stopServer)
        XCTAssertEqual(model.sections[1][1].action, .toggleOpenAtLogin)
        XCTAssertEqual(model.sections[1][2].action, .checkForUpdates)
        XCTAssertEqual(model.sections[1][3].action, .about)
        XCTAssertEqual(model.sections[1][3].title, "About LM Warden")

        // Section 2: Quit
        XCTAssertEqual(model.sections[2].count, 1)
        XCTAssertEqual(model.sections[2][0].action, .quit)
        XCTAssertEqual(model.sections[2][0].keyEquivalent, "q")
        XCTAssertEqual(model.sections[2][0].title, "Quit LM Warden")
    }

    func testShareOnLANChecked() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: true, lanAddress: "10.0.0.1"))
        let item = model.sections[0].first { $0.action == .toggleShareOnLAN }
        XCTAssertEqual(item?.checked, true)
    }

    func testOpenAtLoginChecked() {
        let model = MenuBarModel.make(input(for: .running, openAtLogin: true))
        let item = model.sections[1].first { $0.action == .toggleOpenAtLogin }
        XCTAssertEqual(item?.checked, true)
    }

    // MARK: - Exact rows (reviewer pins: every title, key, check mark and enabled flag)

    private func item(
        _ title: String, detail: String? = nil, key: String? = nil,
        checked: Bool = false, enabled: Bool = true, _ action: MenuAction
    ) -> MenuItemModel {
        MenuItemModel(title: title, detail: detail, keyEquivalent: key, checked: checked, enabled: enabled, action: action)
    }

    func testExactSectionsRunning() {
        let model = MenuBarModel.make(input(for: .running, shareOnLAN: false, port: 8080, openAtLogin: true, canCheckForUpdates: false))
        XCTAssertEqual(model.sections, [
            [
                item("Open Dashboard", key: "d", .openDashboard),
                item("Copy API Address", detail: "127.0.0.1:8080", .copyAPIAddress),
                item("Share on Local Network", .toggleShareOnLAN),
            ],
            [
                item("Stop Server", .stopServer),
                item("Open at Login", checked: true, .toggleOpenAtLogin),
                item("Check for Updates\u{2026}", enabled: false, .checkForUpdates),
                item("About LM Warden", .about),
            ],
            [item("Quit LM Warden", key: "q", .quit)],
        ])
    }

    func testExactSectionsStopped() {
        let model = MenuBarModel.make(input(for: .stopped, shareOnLAN: true, lanAddress: "192.168.1.20", port: 8080, openAtLogin: false, canCheckForUpdates: true))
        XCTAssertEqual(model.sections, [
            [
                item("Open Dashboard", key: "d", enabled: false, .openDashboard),
                item("Copy API Address", detail: "192.168.1.20:8080", enabled: false, .copyAPIAddress),
                item("Share on Local Network", checked: true, .toggleShareOnLAN),
            ],
            [
                item("Start Server", .startServer),
                item("Open at Login", .toggleOpenAtLogin),
                item("Check for Updates\u{2026}", .checkForUpdates),
                item("About LM Warden", .about),
            ],
            [item("Quit LM Warden", key: "q", .quit)],
        ])
    }

    func testCopyAPIAndDashboardDisabledWhenBackingOffOrFailed() {
        for state: SupervisorState in [.backingOff(failures: 1, delay: 1), .failed(failures: 2)] {
            let section0 = MenuBarModel.make(input(for: state)).sections[0]
            XCTAssertEqual(section0[0].enabled, false, "\(state)")
            XCTAssertEqual(section0[1].enabled, false, "\(state)")
            XCTAssertEqual(section0[2].enabled, true, "\(state)")
        }
    }

    func testMemoryFractionMockupValue() {
        // mockup: 8.5 of 21.3 GB -> the bar is ~40 % full, not empty and not full
        let status = ShellStatus(loadedModels: [], usedBytes: 8_500_000_000, usableBytes: 21_300_000_000, inFlight: [])
        let fraction = MenuBarModel.make(input(for: .running, status: status)).models?.memoryFraction ?? -1
        XCTAssertEqual(fraction, 8.5 / 21.3, accuracy: 1e-9)
    }

    func testMemoryFractionNegativeUsedClampsToZero() {
        let status = ShellStatus(loadedModels: [], usedBytes: -1_000_000_000, usableBytes: 10_000_000_000, inFlight: [])
        let model = MenuBarModel.make(input(for: .running, status: status))
        XCTAssertEqual(model.models?.memoryFraction, 0.0)
        XCTAssertEqual(model.models?.memoryLine, "0.0 of 10.0 GB usable for models \u{00B7} 0 requests in flight")
    }

    func testTitle() {
        let model = MenuBarModel.make(input(for: .running))
        XCTAssertEqual(model.title, "LM Warden")
    }

    // MARK: - Helpers

    private func input(
        for state: SupervisorState,
        shareOnLAN: Bool = false,
        lanAddress: String? = nil,
        port: Int = 8080,
        openAtLogin: Bool = false,
        canCheckForUpdates: Bool = true,
        status: ShellStatus? = nil
    ) -> MenuBarInput {
        MenuBarInput(
            state: state,
            port: port,
            shareOnLAN: shareOnLAN,
            lanAddress: lanAddress,
            openAtLogin: openAtLogin,
            canCheckForUpdates: canCheckForUpdates,
            status: status
        )
    }
}
