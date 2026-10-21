import XCTest
@testable import LMWarden

/// Pins `TerminationGate`: every `.terminateLater` answer is followed by
/// exactly one reply, on every quit path, and nothing else ever replies.
///
/// `Harness` plays AppKit and the delegate: it records each answer to
/// `applicationShouldTerminate`, each `reply(toApplicationShouldTerminate:)`
/// and each `NSApp.terminate`, and turns a `.terminate` effect into the
/// `applicationShouldTerminate` call AppKit would make next.
final class TerminationGateTests: XCTestCase {
    private struct Harness {
        var gate = TerminationGate()
        var answers: [TerminationGate.Answer] = []
        var replies: [Bool] = []
        var askedCoordinator = 0

        /// AppKit asks (Dock/logout Quit Apple Event, SIGTERM, first-run cancel,
        /// or the delegate's own `NSApp.terminate`).
        mutating func appKitAsks(hasCoordinator: Bool = true) {
            let answer = gate.shouldTerminate(hasCoordinator: hasCoordinator)
            answers.append(answer)
            if answer == .laterAskCoordinator {
                askedCoordinator += 1
            }
        }

        mutating func apply(_ effect: TerminationGate.Effect) {
            switch effect {
            case .nothing:
                break
            case .reply(let value):
                replies.append(value)
            case .terminate:
                appKitAsks()
            }
        }

        mutating func coordinatorTerminated() {
            apply(gate.coordinatorTerminated())
        }

        mutating func quitCancelled() {
            apply(gate.quitCancelled())
        }

        /// Timers the delegate armed (one generation per armed deadline).
        var armed: [Int] = []

        /// The delegate forwards a quit to the shell queue; it arms a
        /// deadline only when the gate asks for one. Returns the generation
        /// the shell queue will acknowledge.
        @discardableResult
        mutating func forward() -> Int {
            if let generation = gate.quitForwarded() {
                armed.append(generation)
            }
            return gate.forwardGeneration
        }

        /// AppKit asks, and a `.later` answer is forwarded to the shell queue.
        @discardableResult
        mutating func appKitAsksAndForwards() -> Int? {
            appKitAsks()
            return answers.last == .laterAskCoordinator ? forward() : nil
        }

        mutating func acknowledge(_ generation: Int) {
            gate.quitAcknowledged(generation)
        }

        /// The main-thread deadline for `generation` fires.
        mutating func deadline(_ generation: Int) {
            apply(gate.acknowledgementDeadlinePassed(generation))
        }

        var laterCount: Int {
            answers.filter { $0 == .laterAskCoordinator }.count
        }
    }

    // MARK: - Menu Quit (the coordinator starts the quit)

    func testMenuQuitTerminatesWithoutAnyReply() {
        var h = Harness()
        h.coordinatorTerminated()
        XCTAssertEqual(h.answers, [.now])
        XCTAssertEqual(h.replies, [])
        XCTAssertEqual(h.gate.phase, .approved)
    }

    func testMenuQuitPromptCancelledRepliesNothing() {
        var h = Harness()
        h.quitCancelled()
        XCTAssertEqual(h.answers, [])
        XCTAssertEqual(h.replies, [])
        XCTAssertEqual(h.gate.phase, .idle)
    }

    func testMenuWaitThenQuitReachingTheCapTerminatesOnce() {
        var h = Harness()
        // The prompt said Wait, Then Quit; the coordinator quits at the cap.
        h.coordinatorTerminated()
        h.coordinatorTerminated()
        XCTAssertEqual(h.answers, [.now])
        XCTAssertEqual(h.replies, [])
    }

    // MARK: - Quit from outside (Dock, logout, SIGTERM)

    func testLogoutWithNothingInFlightRepliesTrueOnce() {
        var h = Harness()
        h.appKitAsks()
        XCTAssertEqual(h.answers, [.laterAskCoordinator])
        XCTAssertEqual(h.gate.phase, .pending)
        h.coordinatorTerminated()
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.gate.phase, .approved)
    }

    func testLogoutWaitThenQuitReachingTheCapRepliesTrueOnce() {
        var h = Harness()
        h.appKitAsks()
        // The prompt is up, then waits; nothing replies meanwhile.
        XCTAssertEqual(h.replies, [])
        h.coordinatorTerminated()
        // A late duplicate from the coordinator is inert.
        h.coordinatorTerminated()
        h.quitCancelled()
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.laterCount, 1)
    }

    func testLogoutCancelledRepliesFalseOnceAndReArms() {
        var h = Harness()
        h.appKitAsks()
        h.quitCancelled()
        XCTAssertEqual(h.replies, [false])
        XCTAssertEqual(h.gate.phase, .idle)
        // A second Cancel is inert.
        h.quitCancelled()
        XCTAssertEqual(h.replies, [false])
        // The next quit asks the coordinator again and gets its own reply.
        h.appKitAsks()
        h.coordinatorTerminated()
        XCTAssertEqual(h.answers, [.laterAskCoordinator, .laterAskCoordinator])
        XCTAssertEqual(h.replies, [false, true])
    }

    func testSIGTERMWhileAMenuWaitIsRunningRepliesTrueOnce() {
        var h = Harness()
        // Menu Quit -> Wait, Then Quit is running (gate idle), then SIGTERM.
        h.appKitAsks()
        XCTAssertEqual(h.answers, [.laterAskCoordinator])
        // The coordinator ignores the second request; the wait hits the cap.
        h.coordinatorTerminated()
        XCTAssertEqual(h.replies, [true])
    }

    func testSecondQuitWhilePendingIsCancelledAndKeepsTheSingleReply() {
        var h = Harness()
        h.appKitAsks()
        h.appKitAsks()
        XCTAssertEqual(h.answers, [.laterAskCoordinator, .cancel])
        XCTAssertEqual(h.gate.phase, .pending)
        XCTAssertEqual(h.askedCoordinator, 1)
        h.coordinatorTerminated()
        XCTAssertEqual(h.replies, [true])
    }

    func testSecondQuitWhilePendingThenCancelRepliesFalseOnce() {
        var h = Harness()
        h.appKitAsks()
        h.appKitAsks()
        h.quitCancelled()
        XCTAssertEqual(h.replies, [false])
        XCTAssertEqual(h.gate.phase, .idle)
    }

    // MARK: - Before launch and after approval

    func testQuitBeforeTheCoordinatorExistsIsImmediate() {
        var h = Harness()
        h.appKitAsks(hasCoordinator: false)
        XCTAssertEqual(h.answers, [.now])
        XCTAssertEqual(h.gate.phase, .approved)
        XCTAssertEqual(h.replies, [])
    }

    func testApprovedAnswersNowAndIgnoresLateOutcomes() {
        var h = Harness()
        h.appKitAsks()
        h.coordinatorTerminated()
        h.appKitAsks()
        h.quitCancelled()
        h.coordinatorTerminated()
        XCTAssertEqual(h.answers, [.laterAskCoordinator, .now])
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.gate.phase, .approved)
    }

    /// Every path: the number of replies equals the number of
    /// `.laterAskCoordinator` answers once the coordinator has finished.
    func testExactlyOneReplyPerLaterOnEveryPath() {
        let paths: [(String, (inout Harness) -> Void)] = [
            ("quit", { $0.coordinatorTerminated() }),
            ("logout", { $0.appKitAsks(); $0.coordinatorTerminated() }),
            ("logout-cancel", { $0.appKitAsks(); $0.quitCancelled() }),
            ("sigterm-twice", { $0.appKitAsks(); $0.appKitAsks(); $0.coordinatorTerminated() }),
            ("cancel-then-logout", { $0.appKitAsks(); $0.quitCancelled(); $0.appKitAsks(); $0.coordinatorTerminated() }),
        ]
        for (name, run) in paths {
            var h = Harness()
            run(&h)
            XCTAssertEqual(h.replies.count, h.laterCount, name)
        }
    }

    // MARK: - Unacknowledged quit (shell queue blocked on a Keychain prompt)

    func testUnacknowledgedQuitLimitIsFiveSeconds() {
        XCTAssertEqual(TerminationGate.unacknowledgedQuitLimit, 5)
    }

    func testDeadlineWithoutAckWhilePendingRepliesTrueOnce() {
        var h = Harness()
        let generation = h.appKitAsksAndForwards()
        XCTAssertEqual(h.armed, [generation].compactMap { $0 })
        h.deadline(h.armed[0])
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.gate.phase, .approved)
        // Nothing the coordinator does afterwards replies again.
        XCTAssertEqual(h.gate.coordinatorTerminated(), .nothing)
        XCTAssertEqual(h.gate.quitCancelled(), .nothing)
        h.appKitAsks()
        XCTAssertEqual(h.answers, [.laterAskCoordinator, .now])
        XCTAssertEqual(h.replies, [true])
    }

    func testDeadlineFiresThenLateAckIsInert() {
        var h = Harness()
        let generation = h.appKitAsksAndForwards()!
        h.deadline(generation)
        // The Keychain prompt is answered; the queue picks the quit up late.
        h.acknowledge(generation)
        h.coordinatorTerminated()
        h.quitCancelled()
        h.deadline(generation)
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.laterCount, 1)
        XCTAssertEqual(h.gate.phase, .approved)
    }

    func testDeadlineWithoutAckFromMenuQuitTerminates() {
        var h = Harness()
        let generation = h.forward()
        XCTAssertEqual(h.armed, [generation])
        XCTAssertEqual(h.gate.acknowledgementDeadlinePassed(generation), .terminate)
        XCTAssertEqual(h.gate.phase, .approved)
        // NSApp.terminate then asks AppKit, which answers now.
        h.apply(.terminate)
        XCTAssertEqual(h.answers, [.now])
        XCTAssertEqual(h.replies, [])
        // The late coordinator outcome is inert.
        h.acknowledge(generation)
        h.coordinatorTerminated()
        XCTAssertEqual(h.answers, [.now])
    }

    func testAckThenDeadlineIsInertAndCoordinatorRepliesOnce() {
        var h = Harness()
        let generation = h.appKitAsksAndForwards()!
        h.acknowledge(generation)
        h.deadline(generation)
        XCTAssertEqual(h.replies, [])
        XCTAssertEqual(h.gate.phase, .pending)
        h.coordinatorTerminated()
        h.coordinatorTerminated()
        XCTAssertEqual(h.replies, [true])
    }

    func testAckThenCancelRepliesFalseAndALateDeadlineIsInert() {
        var h = Harness()
        let generation = h.appKitAsksAndForwards()!
        h.acknowledge(generation)
        h.quitCancelled()
        h.deadline(generation)
        XCTAssertEqual(h.replies, [false])
        XCTAssertEqual(h.gate.phase, .idle)
    }

    func testMenuQuitAckedThenDeadlineIsInert() {
        var h = Harness()
        let generation = h.forward()
        h.acknowledge(generation)
        XCTAssertEqual(h.gate.acknowledgementDeadlinePassed(generation), .nothing)
        XCTAssertEqual(h.gate.phase, .idle)
    }

    func testSecondForwardWhileUnacknowledgedDoesNotArmASecondDeadline() {
        var h = Harness()
        // Menu Quit is forwarded but the queue is blocked; then SIGTERM.
        let first = h.forward()
        let second = h.appKitAsksAndForwards()
        XCTAssertEqual(second, first)
        XCTAssertEqual(h.armed, [first])
        h.deadline(first)
        XCTAssertEqual(h.replies, [true])
        XCTAssertEqual(h.gate.phase, .approved)
        // Both late acknowledgements are inert.
        h.acknowledge(first)
        h.acknowledge(first)
        h.deadline(first)
        XCTAssertEqual(h.replies, [true])
    }

    func testDeadlineAfterCancelledThenReforwardedQuitActsOnlyForTheNewest() {
        var h = Harness()
        let first = h.appKitAsksAndForwards()!
        h.acknowledge(first)
        h.quitCancelled()
        let second = h.appKitAsksAndForwards()!
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(h.armed, [first, second])
        // The stale timer of the first forward is inert.
        h.deadline(first)
        XCTAssertEqual(h.replies, [false])
        XCTAssertEqual(h.gate.phase, .pending)
        h.deadline(second)
        XCTAssertEqual(h.replies, [false, true])
        XCTAssertEqual(h.gate.phase, .approved)
    }

    func testStaleAckDoesNotAcknowledgeTheNewestForward() {
        var h = Harness()
        let first = h.forward()
        h.acknowledge(first)
        let second = h.forward()
        // An acknowledgement for an older forward does not cover the newest.
        h.acknowledge(first)
        XCTAssertEqual(h.gate.acknowledgementDeadlinePassed(second), .terminate)
    }

    func testForwardAfterApprovalArmsNothing() {
        var h = Harness()
        h.coordinatorTerminated()
        XCTAssertEqual(h.gate.quitForwarded(), nil)
    }

    /// Exactly one reply per `.later` on the watchdog paths too.
    func testExactlyOneReplyPerLaterOnTheWatchdogPaths() {
        let paths: [(String, (inout Harness) -> Void)] = [
            ("deadline", { h in let g = h.appKitAsksAndForwards()!; h.deadline(g) }),
            ("deadline-then-late-ack", { h in
                let g = h.appKitAsksAndForwards()!
                h.deadline(g); h.acknowledge(g); h.coordinatorTerminated()
            }),
            ("ack-then-late-deadline", { h in
                let g = h.appKitAsksAndForwards()!
                h.acknowledge(g); h.coordinatorTerminated(); h.deadline(g)
            }),
            ("ack-cancel-late-deadline", { h in
                let g = h.appKitAsksAndForwards()!
                h.acknowledge(g); h.quitCancelled(); h.deadline(g)
            }),
            ("menu-then-sigterm-deadline", { h in
                let g = h.forward(); h.appKitAsksAndForwards(); h.deadline(g); h.coordinatorTerminated()
            }),
        ]
        for (name, run) in paths {
            var h = Harness()
            run(&h)
            XCTAssertEqual(h.replies.count, h.laterCount, name)
        }
    }

    // MARK: - Pre-confirmed quits (P8 sparkle-updater)

    func testPreconfirmedQuitIsAnsweredConfirmedOnce() {
        var gate = TerminationGate()
        _ = gate.preconfirmQuit()
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterQuitConfirmed)
        XCTAssertFalse(gate.preconfirmed)
        // The user cancels nothing here; the coordinator terminates.
        XCTAssertEqual(gate.coordinatorTerminated(), .reply(true))
    }

    func testPreconfirmationIsConsumedByTheNextRequest() {
        var gate = TerminationGate()
        _ = gate.preconfirmQuit()
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterQuitConfirmed)
        XCTAssertEqual(gate.quitCancelled(), .reply(false))
        // A later ordinary quit asks again.
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterAskCoordinator)
    }

    func testExpiredPreconfirmationAsksAgain() {
        var gate = TerminationGate()
        let generation = gate.preconfirmQuit()
        gate.preconfirmationExpired(generation)
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterAskCoordinator)
    }

    func testStaleExpiryKeepsTheNewerPreconfirmation() {
        var gate = TerminationGate()
        let first = gate.preconfirmQuit()
        _ = gate.preconfirmQuit()
        gate.preconfirmationExpired(first)
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterQuitConfirmed)
    }

    func testPreconfirmationWithoutCoordinatorQuitsNowAndIsDropped() {
        var gate = TerminationGate()
        _ = gate.preconfirmQuit()
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: false), .now)
        XCTAssertFalse(gate.preconfirmed)
    }

    func testPreconfirmationDuringAPendingQuitChangesNothing() {
        var gate = TerminationGate()
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterAskCoordinator)
        _ = gate.preconfirmQuit()
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .cancel)
        XCTAssertEqual(gate.coordinatorTerminated(), .reply(true))
    }

    func testPreconfirmedQuitLimitIsSixtySeconds() {
        XCTAssertEqual(TerminationGate.preconfirmedQuitLimit, 60)
    }

    /// The install did not start or failed: a SIGTERM/logout after that asks.
    func testWithdrawnPreconfirmationAsksAgain() {
        var gate = TerminationGate()
        let generation = gate.preconfirmQuit()
        gate.withdrawPreconfirmation()
        XCTAssertFalse(gate.preconfirmed)
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterAskCoordinator)
        gate.preconfirmationExpired(generation)
        XCTAssertEqual(gate.quitCancelled(), .reply(false))
        XCTAssertEqual(gate.shouldTerminate(hasCoordinator: true), .laterAskCoordinator)
    }
}
