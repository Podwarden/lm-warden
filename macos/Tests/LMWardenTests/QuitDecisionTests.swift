import XCTest
@testable import LMWarden

final class QuitDecisionTests: XCTestCase {
    // MARK: - decide: quitNow arms

    /// Server not running: quit at once, no prompt, regardless of the status
    /// feed (a stale snapshot must not block quitting).
    func testDecideQuitNowWhenNotRunning() {
        let status = QuitDecisionTests.status(requests: [("alpha", "gemma-4b", 3)])
        XCTAssertEqual(QuitDecision.decide(serverRunning: false, status: status), .quitNow)
    }

    /// Running but the feed is down (nil status): no snapshot to warn from,
    /// so quit at once.
    func testDecideQuitNowWhenStatusNil() {
        XCTAssertEqual(QuitDecision.decide(serverRunning: true, status: nil), .quitNow)
    }

    /// Running with the feed and zero in-flight requests: quit at once.
    func testDecideQuitNowWhenNoRequestsInFlight() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 0),
            ("beta", "llama-3-8b", 0),
        ])
        XCTAssertEqual(QuitDecision.decide(serverRunning: true, status: status), .quitNow)
    }

    // MARK: - decide: confirm arms

    /// One in-flight request: the prompt, with the singular message line.
    func testDecideConfirmOneRequest() {
        let status = QuitDecisionTests.status(requests: [("alpha", "gemma-4b", 1)])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "1 request is still running. Quitting stops them and unloads all models.",
            details: ["alpha — gemma-4b · 1 request"],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    /// Three in-flight requests across two groups: plural message line, one
    /// details line per group in input order, no "and N more" line.
    func testDecideConfirmThreeRequests() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 2),
            ("beta", "llama-3-8b", 1),
        ])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "3 requests are still running. Quitting stops them and unloads all models.",
            details: [
                "alpha — gemma-4b · 2 requests",
                "beta — llama-3-8b · 1 request",
            ],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    /// A group with zero requests is skipped from the details even though
    /// the total is still above the quitNow threshold.
    func testDecideSkipsGroupWithZeroRequests() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 0),
            ("beta", "llama-3-8b", 1),
        ])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "1 request is still running. Quitting stops them and unloads all models.",
            details: ["beta — llama-3-8b · 1 request"],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    /// Exactly four non-empty groups: all four lines shown, and no
    /// "and N more" line (the bound).
    func testDecideConfirmExactlyFourGroups() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 1),
            ("beta", "llama-3-8b", 1),
            ("gamma", "mistral-7b", 1),
            ("delta", "qwen-2-7b", 1),
        ])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "4 requests are still running. Quitting stops them and unloads all models.",
            details: [
                "alpha — gemma-4b · 1 request",
                "beta — llama-3-8b · 1 request",
                "gamma — mistral-7b · 1 request",
                "delta — qwen-2-7b · 1 request",
            ],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    /// Five non-empty groups: the first four lines plus "and 1 more"
    /// (one past the bound).
    func testDecideConfirmFiveGroupsAddsMoreLine() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 1),
            ("beta", "llama-3-8b", 1),
            ("gamma", "mistral-7b", 1),
            ("delta", "qwen-2-7b", 1),
            ("epsilon", "phi-3", 2),
        ])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "6 requests are still running. Quitting stops them and unloads all models.",
            details: [
                "alpha — gemma-4b · 1 request",
                "beta — llama-3-8b · 1 request",
                "gamma — mistral-7b · 1 request",
                "delta — qwen-2-7b · 1 request",
                "and 1 more",
            ],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    /// Seven non-empty groups, one of them with zero requests (skipped):
    /// six groups remain, so the details are the first four plus
    /// "and 2 more".
    func testDecideConfirmMoreLineCountsOnlyRemainingGroups() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 1),
            ("beta", "llama-3-8b", 0),
            ("gamma", "mistral-7b", 1),
            ("delta", "qwen-2-7b", 1),
            ("epsilon", "phi-3", 1),
            ("zeta", "deepseek-2", 1),
            ("eta", "tiny-1", 1),
        ])
        let decision = QuitDecision.decide(serverRunning: true, status: status)
        XCTAssertEqual(decision, .confirm(QuitPrompt(
            title: "Quit LM Warden?",
            message: "6 requests are still running. Quitting stops them and unloads all models.",
            details: [
                "alpha — gemma-4b · 1 request",
                "gamma — mistral-7b · 1 request",
                "delta — qwen-2-7b · 1 request",
                "epsilon — phi-3 · 1 request",
                "and 2 more",
            ],
            buttons: ["Wait, Then Quit", "Quit Now", "Cancel"],
            footnote: "Closing the dashboard window does not quit. LM Warden keeps serving from the menu bar."
        )))
    }

    // MARK: - choice(forButtonIndex:)

    func testChoiceForButtonIndex() {
        XCTAssertEqual(QuitDecision.choice(forButtonIndex: 0), .waitThenQuit)
        XCTAssertEqual(QuitDecision.choice(forButtonIndex: 1), .quitNow)
        XCTAssertEqual(QuitDecision.choice(forButtonIndex: 2), .cancel)
        XCTAssertEqual(QuitDecision.choice(forButtonIndex: 3), .cancel)
        XCTAssertEqual(QuitDecision.choice(forButtonIndex: -1), .cancel)
    }

    // MARK: - shouldQuitAfterPoll

    /// No feed at all: nothing left to wait for.
    func testShouldQuitAfterPollNilStatus() {
        XCTAssertTrue(QuitDecision.shouldQuitAfterPoll(status: nil))
    }

    /// Feed up but the last in-flight request finished.
    func testShouldQuitAfterPollZeroRequests() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 0),
            ("beta", "llama-3-8b", 0),
        ])
        XCTAssertTrue(QuitDecision.shouldQuitAfterPoll(status: status))
    }

    /// One request left in flight: keep polling.
    func testShouldNotQuitAfterPollWhileRequestsRemain() {
        let status = QuitDecisionTests.status(requests: [
            ("alpha", "gemma-4b", 0),
            ("beta", "llama-3-8b", 1),
        ])
        XCTAssertFalse(QuitDecision.shouldQuitAfterPoll(status: status))
    }

    // MARK: - Fixtures

    /// A ShellStatus whose in-flight groups are the given `(who, model, count)` rows.
    static func status(requests: [(String, String, Int)]) -> ShellStatus {
        let groups = requests.map { InFlightGroup(who: $0.0, model: $0.1, requests: $0.2) }
        return ShellStatus(loadedModels: [], usedBytes: 0, usableBytes: 0, inFlight: groups)
    }
}
