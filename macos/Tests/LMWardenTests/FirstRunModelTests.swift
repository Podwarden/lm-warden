import XCTest
@testable import LMWarden

final class FirstRunModelTests: XCTestCase {

    // MARK: - Fixtures

    private let giB: UInt64 = 1_073_741_824
    private let tenGB: Int64 = 10_000_000_000

    /// A healthy Apple-Silicon Mac with room to spare.
    private func facts(
        chip: String? = "Apple M4",
        memoryBytes: UInt64 = 24 * 1_073_741_824,
        freeDiskBytes: Int64? = 100_000_000_000
    ) -> MacFacts {
        MacFacts(chipName: chip, isAppleSilicon: chip != nil, memoryBytes: memoryBytes, freeDiskBytes: freeDiskBytes)
    }

    /// A default first-run model: local-only, default models dir, plenty of
    /// free space, no LAN address, no keychain secret yet.
    private func model(
        factsOverride: MacFacts? = nil,
        freeSpace: Int64? = 100_000_000_000,
        volumeName: String? = "Macintosh HD",
        homeDir: String = "/Users/ann",
        port: Int = 8080,
        lanAddress: String? = nil,
        keychainHasSecret: Bool = false
    ) -> FirstRunModel {
        FirstRunModel(
            facts: factsOverride ?? facts(),
            defaultModelsDir: homeDir + "/Library/Application Support/LM Warden/models",
            homeDir: homeDir,
            port: port,
            lanAddress: lanAddress,
            keychainHasSecret: keychainHasSecret,
            freeSpace: { _ in freeSpace },
            volumeName: { _ in volumeName }
        )
    }

    /// Walk a fresh model to the Models step (the only step whose folder
    /// chooser is live).
    private func toModels(_ m: inout FirstRunModel) {
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.step, .models)
    }

    // MARK: - MacCheck rows

    func testMacCheckRowsOrderAndHeaders() {
        let rows = MacCheck.rows(facts())
        XCTAssertEqual(rows.count, 3)
        XCTAssertEqual(rows.map { $0.label }, ["Apple Silicon", "Memory", "Free disk space"])
    }

    func testMacCheckAppleSiliconWithChip() {
        let row = MacCheck.rows(facts())[0]
        XCTAssertEqual(row.value, "Apple M4 — OK")
        XCTAssertEqual(row.ok, true)
    }

    func testMacCheckAppleSiliconNilChip() {
        // isAppleSilicon true but chip name unknown -> "Apple Silicon — OK"
        let f = MacFacts(chipName: nil, isAppleSilicon: true, memoryBytes: 16 * giB, freeDiskBytes: 20_000_000_000)
        XCTAssertEqual(MacCheck.rows(f)[0].value, "Apple Silicon — OK")
        XCTAssertEqual(MacCheck.rows(f)[0].ok, true)
    }

    func testMacCheckIntelNotOK() {
        let f = MacFacts(chipName: "Intel Core i7", isAppleSilicon: false, memoryBytes: 32 * giB, freeDiskBytes: 20_000_000_000)
        let row = MacCheck.rows(f)[0]
        XCTAssertEqual(row.value, "Intel Core i7 — Apple Silicon required")
        XCTAssertEqual(row.ok, false)
    }

    func testMacCheckIntelNilChip() {
        let f = MacFacts(chipName: nil, isAppleSilicon: false, memoryBytes: 32 * giB, freeDiskBytes: 20_000_000_000)
        let row = MacCheck.rows(f)[0]
        XCTAssertEqual(row.value, "Intel — Apple Silicon required")
        XCTAssertEqual(row.ok, false)
    }

    func testMacCheckMemoryExactlySixteenGiBIsOK() {
        let f = MacFacts(chipName: "Apple M4", isAppleSilicon: true, memoryBytes: 16 * giB, freeDiskBytes: 20_000_000_000)
        let row = MacCheck.rows(f)[1]
        XCTAssertEqual(row.value, "16 GB — OK (16 GB is the minimum)")
        XCTAssertEqual(row.ok, true)
    }

    func testMacCheckMemoryOneByteBelowSixteenGiBNotOK() {
        let f = MacFacts(chipName: "Apple M4", isAppleSilicon: true, memoryBytes: 16 * giB - 1, freeDiskBytes: 20_000_000_000)
        let row = MacCheck.rows(f)[1]
        // 16 GiB - 1 byte floors to 15 GiB
        XCTAssertEqual(row.value, "15 GB — below the 16 GB minimum")
        XCTAssertEqual(row.ok, false)
    }

    func testMacCheckMemoryAboveMinimum() {
        let f = MacFacts(chipName: "Apple M4", isAppleSilicon: true, memoryBytes: 32 * giB, freeDiskBytes: 20_000_000_000)
        XCTAssertEqual(MacCheck.rows(f)[1].value, "32 GB — OK (16 GB is the minimum)")
        XCTAssertEqual(MacCheck.rows(f)[1].ok, true)
    }

    func testMacCheckFreeDiskOK() {
        let row = MacCheck.rows(facts(freeDiskBytes: 50_000_000_000))[2]
        XCTAssertEqual(row.value, "50 GB — OK")
        XCTAssertEqual(row.ok, true)
    }

    func testMacCheckFreeDiskExactlyTenGBIsOK() {
        let row = MacCheck.rows(facts(freeDiskBytes: 10_000_000_000))[2]
        XCTAssertEqual(row.value, "10 GB — OK")
        XCTAssertEqual(row.ok, true)
    }

    func testMacCheckFreeDiskJustUnderTenGB() {
        let row = MacCheck.rows(facts(freeDiskBytes: 10_000_000_000 - 1))[2]
        XCTAssertEqual(row.value, "9 GB — less than 10 GB free")
        XCTAssertEqual(row.ok, false)
    }

    func testMacCheckFreeDiskNilUnknown() {
        let row = MacCheck.rows(facts(freeDiskBytes: nil))[2]
        XCTAssertEqual(row.value, "Unknown")
        XCTAssertEqual(row.ok, true)
    }

    func testMacCheckMinimumMemoryConstant() {
        XCTAssertEqual(MacCheck.minimumMemoryBytes, 16 * 1_073_741_824)
    }

    // MARK: - Initial state and navigation

    func testInitialState() {
        var m = model()
        XCTAssertEqual(m.step, .welcome)
        XCTAssertEqual(m.shareOnLAN, false)
        XCTAssertEqual(m.modelsDir, "/Users/ann/Library/Application Support/LM Warden/models")
        XCTAssertEqual(m.excludeModelsFromBackup, true)
        XCTAssertEqual(m.openAtLogin, true)
    }

    func testWindowTitle() {
        XCTAssertEqual(model().windowTitle, "Set Up LM Warden")
    }

    func testNextStepsThroughToFinish() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.step, .network)
        _ = m.next()
        XCTAssertEqual(m.step, .models)
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
    }

    func testBackOnWelcomeIsNoop() {
        var m = model()
        m.back()
        XCTAssertEqual(m.step, .welcome)
    }

    func testBackReturnsToPreviousStep() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.step, .network)
        m.back()
        XCTAssertEqual(m.step, .welcome)
    }

    // MARK: - Sidebar labels and next label per step

    func testSidebarLabelsWelcome() {
        XCTAssertEqual(model().stepLabels, ["1. Welcome", "2. Network", "3. Models", "4. Finish"])
    }

    func testSidebarLabelsNetwork() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.stepLabels, ["✓ Welcome", "2. Network", "3. Models", "4. Finish"])
    }

    func testSidebarLabelsModels() {
        var m = model()
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.stepLabels, ["✓ Welcome", "✓ Network", "3. Models", "4. Finish"])
    }

    func testSidebarLabelsFinish() {
        var m = model()
        _ = m.next()
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.stepLabels, ["✓ Welcome", "✓ Network", "✓ Models", "4. Finish"])
    }

    func testNextLabelContinueThenOpenDashboard() {
        var m = model()
        XCTAssertEqual(m.nextLabel, "Continue")
        _ = m.next()
        XCTAssertEqual(m.nextLabel, "Continue")
        _ = m.next()
        XCTAssertEqual(m.nextLabel, "Continue")
        _ = m.next()
        XCTAssertEqual(m.nextLabel, "Open Dashboard")
    }

    func testBackEnabledOnlyFalseOnWelcome() {
        var m = model()
        XCTAssertEqual(m.backEnabled, false)
        _ = m.next()
        XCTAssertEqual(m.backEnabled, true)
        m.back()
        XCTAssertEqual(m.backEnabled, false)
    }

    // MARK: - Next returns result only on finish

    func testNextNilOnNonFinishSteps() {
        var m = model()
        XCTAssertNil(m.next())
        XCTAssertEqual(m.step, .network)
        XCTAssertNil(m.next())
        XCTAssertEqual(m.step, .models)
    }

    // MARK: - canContinue and the 10 GB guard

    func testCanContinueTrueOnWelcome() {
        XCTAssertEqual(model().canContinue, true)
    }

    func testCanContinueTrueOnNetworkAndFinish() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.canContinue, true)
        _ = m.next()
        XCTAssertEqual(m.step, .models)
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
        XCTAssertEqual(m.canContinue, true)
    }

    func testGuardExactlyTenGBAllowsContinue() {
        var m = model(freeSpace: 10_000_000_000)
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.step, .models)
        XCTAssertEqual(m.canContinue, true)
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
    }

    func testGuardJustUnderTenGBBlocksContinue() {
        var m = model(freeSpace: 10_000_000_000 - 1)
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.step, .models)
        XCTAssertEqual(m.canContinue, false)
        // next() does nothing when canContinue is false
        _ = m.next()
        XCTAssertEqual(m.step, .models)
    }

    func testGuardUnknownFreeSpaceAllowsContinue() {
        var m = model(freeSpace: nil)
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.step, .models)
        XCTAssertEqual(m.canContinue, true)
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
    }

    func testGuardOnlyAppliesOnModelsStep() {
        // Plenty low, but on .network the guard is not in play.
        var m = model(freeSpace: 1)
        _ = m.next()
        XCTAssertEqual(m.step, .network)
        XCTAssertEqual(m.canContinue, true)
    }

    // MARK: - chooseModelsDir

    func testChooseModelsDirAbsolutePathAccepted() {
        var m = model()
        toModels(&m)
        m.chooseModelsDir("/Volumes/Extra/models")
        XCTAssertEqual(m.modelsDir, "/Volumes/Extra/models")
    }

    func testChooseModelsDirRelativeIgnored() {
        var m = model()
        toModels(&m)
        m.chooseModelsDir("Volumes/Extra/models")
        XCTAssertEqual(m.modelsDir, "/Users/ann/Library/Application Support/LM Warden/models")
    }

    func testChooseModelsDirEmptyIgnored() {
        var m = model()
        toModels(&m)
        m.chooseModelsDir("")
        XCTAssertEqual(m.modelsDir, "/Users/ann/Library/Application Support/LM Warden/models")
    }

    // MARK: - Toggles and chooseNetwork

    func testChooseNetworkLanTrue() {
        var m = model()
        m.chooseNetwork(lan: true)
        XCTAssertEqual(m.shareOnLAN, true)
    }

    func testChooseNetworkLanFalse() {
        var m = model()
        m.chooseNetwork(lan: true)
        XCTAssertEqual(m.shareOnLAN, true)
        m.chooseNetwork(lan: false)
        XCTAssertEqual(m.shareOnLAN, false)
    }

    func testToggleExcludeModelsFromBackup() {
        var m = model()
        XCTAssertEqual(m.excludeModelsFromBackup, true)
        m.toggleExcludeModelsFromBackup()
        XCTAssertEqual(m.excludeModelsFromBackup, false)
        m.toggleExcludeModelsFromBackup()
        XCTAssertEqual(m.excludeModelsFromBackup, true)
    }

    func testToggleOpenAtLogin() {
        var m = model()
        XCTAssertEqual(m.openAtLogin, true)
        m.toggleOpenAtLogin()
        XCTAssertEqual(m.openAtLogin, false)
        m.toggleOpenAtLogin()
        XCTAssertEqual(m.openAtLogin, true)
    }

    // MARK: - Welcome step copy

    func testWelcomeHeadlineAndBody() {
        var m = model()
        XCTAssertEqual(m.headline, "Welcome to LM Warden")
        XCTAssertEqual(m.body, "LM Warden runs AI models on this Mac and serves them to the apps and people you choose. This setup takes about a minute. Everything else is in the dashboard.")
    }

    func testWelcomeMacCheckRowsPresent() {
        var m = model()
        XCTAssertEqual(m.macCheckRows, MacCheck.rows(facts()))
    }

    // MARK: - Network step copy

    func testNetworkHeadline() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.headline, "Who can use this server?")
    }

    func testNetworkLocalCardDefaultPort() {
        var m = model()
        _ = m.next()
        XCTAssertEqual(m.localCardTitle, "Only this Mac")
        XCTAssertEqual(m.localCardBody, "Apps on this Mac reach it at 127.0.0.1:8080. Recommended.")
    }

    func testNetworkLocalCardCustomPort() {
        var m = model(port: 9090)
        _ = m.next()
        XCTAssertEqual(m.localCardBody, "Apps on this Mac reach it at 127.0.0.1:9090. Recommended.")
    }

    func testNetworkLanCardWithAddress() {
        var m = model(port: 8080, lanAddress: "192.168.1.50")
        _ = m.next()
        XCTAssertEqual(m.lanCardTitle, "People on my network")
        XCTAssertEqual(m.lanCardBody, "Others reach it at 192.168.1.50:8080. macOS asks you to allow local network access.")
    }

    func testNetworkLanCardNilAddress() {
        var m = model(port: 8080, lanAddress: nil)
        _ = m.next()
        XCTAssertEqual(m.lanCardTitle, "People on my network")
        XCTAssertEqual(m.lanCardBody, "Others reach it at this Mac's network address. macOS asks you to allow local network access.")
    }

    func testLanWarningOnlyWhenSharing() {
        var m = model()
        _ = m.next()
        XCTAssertNil(m.lanWarning)
        m.chooseNetwork(lan: true)
        XCTAssertEqual(
            m.lanWarning,
            "Traffic on your network is not encrypted. API tokens and sign-ins can be read by others on the same network. Use Tailscale or WireGuard for anything sensitive."
        )
    }

    func testLanWarningGoneWhenLocalAgain() {
        var m = model()
        _ = m.next()
        m.chooseNetwork(lan: true)
        XCTAssertNotNil(m.lanWarning)
        m.chooseNetwork(lan: false)
        XCTAssertNil(m.lanWarning)
    }

    // MARK: - Models step copy

    func testModelsHeadlineAndBody() {
        var m = model()
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.headline, "Where should models be stored?")
        XCTAssertEqual(m.body, "Models are large, often 5 to 50 GB each. Pick a disk with room to spare.")
    }

    func testModelsFreeLineKnownVolume() {
        var m = model(freeSpace: 50_000_000_000, volumeName: "Macintosh HD")
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(
            m.freeLine,
            "50 GB free on Macintosh HD. LM Warden will not start a download that would leave less than 10 GB free."
        )
    }

    func testModelsFreeLineUnknownVolume() {
        var m = model(freeSpace: 50_000_000_000, volumeName: nil)
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(
            m.freeLine,
            "50 GB free on this disk. LM Warden will not start a download that would leave less than 10 GB free."
        )
    }

    func testModelsFreeLineUnknownFreeSpace() {
        var m = model(freeSpace: nil)
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(
            m.freeLine,
            "Free space on this disk is unknown. LM Warden will not start a download that would leave less than 10 GB free."
        )
    }

    func testModelsCheckboxLabel() {
        var m = model()
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.backupCheckboxLabel, "Leave models out of Time Machine backups")
    }

    func testDisplayedFolderHomeAbbreviated() {
        var m = model()
        XCTAssertEqual(m.displayedModelsDir, "~/Library/Application Support/LM Warden/models")
    }

    func testDisplayedFolderHomePrefixNotAbbreviated() {
        // "/Users/ann2" does not start at a component boundary after "/Users/ann"
        var m = model(homeDir: "/Users/ann")
        toModels(&m)
        m.chooseModelsDir("/Users/ann2/Library/Application Support/LM Warden/models")
        XCTAssertEqual(m.displayedModelsDir, "/Users/ann2/Library/Application Support/LM Warden/models")
    }

    func testDisplayedFolderOutsideHomeUnchanged() {
        var m = model()
        toModels(&m)
        m.chooseModelsDir("/Volumes/Extra/models")
        XCTAssertEqual(m.displayedModelsDir, "/Volumes/Extra/models")
    }

    // MARK: - Finish step copy

    private func finished(_ m: inout FirstRunModel) {
        _ = m.next()
        _ = m.next()
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
    }

    func testFinishHeadline() {
        var m = model()
        finished(&m)
        XCTAssertEqual(m.headline, "You are ready")
    }

    func testFinishLocalLine() {
        var m = model()
        finished(&m)
        XCTAssertEqual(m.finishLines[0], "✓ The server starts on this Mac only.")
    }

    func testFinishLANLine() {
        var m = model()
        m.chooseNetwork(lan: true)
        finished(&m)
        XCTAssertEqual(m.finishLines[0], "✓ The server is shared with people on your network.")
    }

    func testFinishModelsLine() {
        var m = model()
        finished(&m)
        XCTAssertEqual(m.finishLines[1], "✓ Models are stored in ~/Library/Application Support/LM Warden/models.")
    }

    func testFinishSecretCreatedLine() {
        var m = model(keychainHasSecret: false)
        finished(&m)
        XCTAssertEqual(m.finishLines[2], "✓ A sign-in secret was created and saved in your Keychain.")
    }

    func testFinishSecretExistingLine() {
        var m = model(keychainHasSecret: true)
        finished(&m)
        XCTAssertEqual(m.finishLines[2], "✓ Your sign-in secret is saved in your Keychain.")
    }

    func testFinishCheckboxAndFooter() {
        var m = model()
        finished(&m)
        XCTAssertEqual(m.openAtLoginCheckboxLabel, "Open LM Warden when I log in")
        XCTAssertEqual(m.footer, "Next, the dashboard asks you to create the admin account and download a first model.")
    }

    func testFinishResultCarriesAllFourChoices() {
        var m = model(lanAddress: "192.168.1.50")
        m.chooseNetwork(lan: true)
        toModels(&m)
        m.chooseModelsDir("/Volumes/Extra/models")
        m.toggleExcludeModelsFromBackup()
        m.toggleOpenAtLogin()
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
        let result = m.next()
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.shareOnLAN, true)
        XCTAssertEqual(result?.modelsDir, "/Volumes/Extra/models")
        XCTAssertEqual(result?.excludeModelsFromBackup, false)
        XCTAssertEqual(result?.openAtLogin, false)
    }

    func testFinishResultDefaults() {
        var m = model()
        finished(&m)
        let result = m.next()
        XCTAssertEqual(result, FirstRunResult(
            shareOnLAN: false,
            modelsDir: "/Users/ann/Library/Application Support/LM Warden/models",
            excludeModelsFromBackup: true,
            openAtLogin: true
        ))
    }

    // MARK: - Reviewer legs (farm 43)

    func testMacCheckFreeDiskArtboardExample() {
        // Approved artboard: "412 GB — OK"; whole decimal GB, floored.
        let row = MacCheck.rows(facts(freeDiskBytes: 412_999_999_999))[2]
        XCTAssertEqual(row.value, "412 GB — OK")
    }

    func testModelsFreeLineArtboardExample() {
        var m = model(freeSpace: 412_999_999_999, volumeName: "Macintosh HD")
        toModels(&m)
        XCTAssertEqual(
            m.freeLine,
            "412 GB free on Macintosh HD. LM Warden will not start a download that would leave less than 10 GB free."
        )
    }

    func testMacCheckSixteenGBMacPhysicalMemoryIsOK() {
        // A "16 GB" Mac reports physicalMemory 17_179_869_184 (16 GiB).
        let row = MacCheck.rows(facts(memoryBytes: 17_179_869_184))[1]
        XCTAssertEqual(row.value, "16 GB — OK (16 GB is the minimum)")
        XCTAssertTrue(row.ok)
    }

    func testMacCheckSixteenDecimalGBIsBelowMinimum() {
        // The minimum is 16 GiB, not 16 * 10^9 bytes.
        let row = MacCheck.rows(facts(memoryBytes: 16_000_000_000))[1]
        XCTAssertEqual(row.value, "14 GB — below the 16 GB minimum")
        XCTAssertFalse(row.ok)
    }

    func testChooseModelsDirIgnoredOffModelsStep() {
        var m = model()
        m.chooseModelsDir("/Volumes/Extra/models")
        XCTAssertEqual(m.modelsDir, "/Users/ann/Library/Application Support/LM Warden/models")
        _ = m.next()
        m.chooseModelsDir("/Volumes/Extra/models")
        XCTAssertEqual(m.modelsDir, "/Users/ann/Library/Application Support/LM Warden/models")
    }

    func testFinishCannotCarryAFolderThatSkippedTheGuard() {
        // Big default disk passes the guard; a tiny folder chosen on Finish
        // must not reach the result.
        let tiny = "/Volumes/Tiny/models"
        var m = FirstRunModel(
            facts: facts(),
            defaultModelsDir: "/Users/ann/models",
            homeDir: "/Users/ann",
            port: 8080,
            lanAddress: nil,
            keychainHasSecret: false,
            freeSpace: { $0 == tiny ? 1_000_000_000 : 100_000_000_000 },
            volumeName: { _ in nil }
        )
        toModels(&m)
        _ = m.next()
        XCTAssertEqual(m.step, .finish)
        m.chooseModelsDir(tiny)
        XCTAssertEqual(m.next()?.modelsDir, "/Users/ann/models")
    }

    func testTinyFolderOnModelsStepBlocksFinish() {
        let tiny = "/Volumes/Tiny/models"
        var m = FirstRunModel(
            facts: facts(),
            defaultModelsDir: "/Users/ann/models",
            homeDir: "/Users/ann",
            port: 8080,
            lanAddress: nil,
            keychainHasSecret: false,
            freeSpace: { $0 == tiny ? 1_000_000_000 : 100_000_000_000 },
            volumeName: { _ in nil }
        )
        toModels(&m)
        m.chooseModelsDir(tiny)
        XCTAssertFalse(m.canContinue)
        XCTAssertNil(m.next())
        XCTAssertEqual(m.step, .models)
        m.chooseModelsDir("/Users/ann/models")
        XCTAssertTrue(m.canContinue)
    }

    func testCannotSkipAStep() {
        var m = model()
        var seen: [FirstRunStep] = [m.step]
        for _ in 0..<3 { XCTAssertNil(m.next()); seen.append(m.step) }
        XCTAssertEqual(seen, [.welcome, .network, .models, .finish])
        XCTAssertNotNil(m.next())
        XCTAssertEqual(m.step, .finish)
    }
}
