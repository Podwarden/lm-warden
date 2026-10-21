// ShellStatusFeedTests — the main-thread poll loop behind the coordinator's
// status seam.
//
// Fake-driven: a manual scheduler (records each delay and its work, fires on
// demand), a recording fetch (keeps each request's port and completion, never
// touches a socket) and `deliver: { $0() }`. Nothing waits on a real timer.
import XCTest
@testable import LMWarden

// MARK: - Fakes

private final class ManualScheduler {
    private(set) var delays: [TimeInterval] = []
    private var pending: [() -> Void] = []

    var pendingCount: Int { pending.count }

    func schedule(_ delay: TimeInterval, _ work: @escaping () -> Void) {
        delays.append(delay)
        pending.append(work)
    }

    /// Fires every work item scheduled so far (not those they schedule).
    func fireAll() {
        let due = pending
        pending = []
        due.forEach { $0() }
    }
}

private final class RecordingFetch {
    struct Call {
        let port: Int
        let completion: (ShellStatus?) -> Void
    }
    private(set) var calls: [Call] = []

    func fetch(_ port: Int, _ token: String, _ done: @escaping (ShellStatus?) -> Void) {
        calls.append(Call(port: port, completion: done))
    }

    func complete(_ index: Int, with status: ShellStatus?) {
        calls[index].completion(status)
    }
}

private final class Endpoint {
    var value: (port: Int, token: String)? = (port: 8080, token: String(repeating: "a", count: 64))
}

private func status(inFlight requests: Int, used: Int64 = 100) -> ShellStatus {
    ShellStatus(
        loadedModels: [LoadedModel(name: "m", backend: "mlx", bytes: 10)],
        usedBytes: used,
        usableBytes: 1000,
        inFlight: requests == 0 ? [] : [InFlightGroup(who: "web", model: "m", requests: requests)]
    )
}

final class ShellStatusFeedTests: XCTestCase {
    private var scheduler: ManualScheduler!
    private var fetcher: RecordingFetch!
    private var endpoint: Endpoint!
    private var changes: [ShellStatus?] = []

    override func setUp() {
        super.setUp()
        scheduler = ManualScheduler()
        fetcher = RecordingFetch()
        endpoint = Endpoint()
        changes = []
    }

    private func makeFeed(interval: TimeInterval = ShellStatusFeed.defaultInterval) -> ShellStatusFeed {
        let endpoint = self.endpoint!
        let fetcher = self.fetcher!
        let scheduler = self.scheduler!
        let feed = ShellStatusFeed(
            endpoint: { endpoint.value },
            fetch: { port, token, done in fetcher.fetch(port, token, done) },
            schedule: { delay, work in scheduler.schedule(delay, work) },
            deliver: { $0() },
            interval: interval
        )
        feed.onChange = { [weak self] value in self?.changes.append(value) }
        return feed
    }

    func testDefaultIntervalIsTwoSeconds() {
        XCTAssertEqual(ShellStatusFeed.defaultInterval, 2)
    }

    func testFirstPollIsImmediate() {
        let feed = makeFeed()
        XCTAssertEqual(fetcher.calls.count, 0, "nothing before start")
        feed.start()
        XCTAssertEqual(fetcher.calls.count, 1)
        XCTAssertEqual(fetcher.calls[0].port, 8080)
    }

    func testCadenceIsTheInterval() {
        let feed = makeFeed(interval: 5)
        feed.start()
        XCTAssertEqual(scheduler.delays, [5])
        fetcher.complete(0, with: status(inFlight: 0))
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2)
        XCTAssertEqual(scheduler.delays, [5, 5])
        fetcher.complete(1, with: status(inFlight: 0))
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 3)
        XCTAssertEqual(scheduler.delays, [5, 5, 5])
    }

    func testStartTwiceRunsOneLoop() {
        let feed = makeFeed()
        feed.start()
        feed.start()
        XCTAssertEqual(fetcher.calls.count, 1)
        XCTAssertEqual(scheduler.pendingCount, 1)
        fetcher.complete(0, with: nil)
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2)
        XCTAssertEqual(scheduler.pendingCount, 1)
    }

    func testNilEndpointFetchesNothingAndClearsLatest() {
        let feed = makeFeed()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 1))
        XCTAssertEqual(feed.latest, status(inFlight: 1))
        endpoint.value = nil
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 1, "no fetch while the server is not booted")
        XCTAssertNil(feed.latest)
        XCTAssertEqual(changes, [status(inFlight: 1), nil])
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 1)
        XCTAssertEqual(scheduler.pendingCount, 1, "the loop keeps ticking while down")
        XCTAssertEqual(changes.count, 2, "no repeated onChange while down")
    }

    func testNilEndpointFromTheStartFiresNoChange() {
        endpoint.value = nil
        let feed = makeFeed()
        feed.start()
        XCTAssertEqual(fetcher.calls.count, 0)
        XCTAssertNil(feed.latest)
        XCTAssertTrue(changes.isEmpty)
    }

    func testAtMostOneFetchOutstanding() {
        let feed = makeFeed()
        feed.start()
        scheduler.fireAll()
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 1, "ticks during an in-flight fetch issue none")
        fetcher.complete(0, with: status(inFlight: 0))
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2, "the next tick after the reply fetches again")
    }

    func testResultAppliedAndOnChangeOnce() {
        let feed = makeFeed()
        feed.start()
        XCTAssertNil(feed.latest)
        fetcher.complete(0, with: status(inFlight: 2))
        XCTAssertEqual(feed.latest, status(inFlight: 2))
        XCTAssertEqual(changes, [status(inFlight: 2)])
    }

    func testIdenticalResultFiresNoSecondChange() {
        let feed = makeFeed()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 2))
        scheduler.fireAll()
        fetcher.complete(1, with: status(inFlight: 2))
        XCTAssertEqual(changes, [status(inFlight: 2)])
        scheduler.fireAll()
        fetcher.complete(2, with: status(inFlight: 3))
        XCTAssertEqual(changes, [status(inFlight: 2), status(inFlight: 3)])
    }

    func testNilResultClearsLatest() {
        let feed = makeFeed()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 4))
        scheduler.fireAll()
        fetcher.complete(1, with: nil)
        XCTAssertNil(feed.latest, "a stale in-flight count must not survive a failed poll")
        XCTAssertEqual(changes, [status(inFlight: 4), nil])
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 3, "the tick after a failed poll fetches again")
        guard fetcher.calls.count == 3 else { return }
        fetcher.complete(2, with: nil)
        XCTAssertEqual(changes.count, 2, "nil then nil is no change")
    }

    func testNilReplyKeepsPollingAndRecovers() {
        let feed = makeFeed()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 1))
        scheduler.fireAll()
        fetcher.complete(1, with: nil)
        XCTAssertNil(feed.latest)
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 3, "a failed poll does not stop the feed: the next tick fetches again")
        XCTAssertEqual(scheduler.pendingCount, 1)
        guard fetcher.calls.count == 3 else { return }
        fetcher.complete(2, with: nil)
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 4, "still fetching while the server keeps failing")
        guard fetcher.calls.count == 4 else { return }
        fetcher.complete(3, with: status(inFlight: 2))
        XCTAssertEqual(feed.latest, status(inFlight: 2), "the feed recovers when the server answers again")
        XCTAssertEqual(changes, [status(inFlight: 1), nil, status(inFlight: 2)],
                       "onChange once on the transition to nil and once on recovery, never per tick")
    }

    func testEndpointBackAfterDownResumesFetching() {
        endpoint.value = nil
        let feed = makeFeed()
        feed.start()
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 0)
        endpoint.value = (port: 8080, token: String(repeating: "c", count: 64))
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 1, "the feed fetches once the server has booted")
        guard fetcher.calls.count == 1 else { return }
        fetcher.complete(0, with: status(inFlight: 1))
        XCTAssertEqual(feed.latest, status(inFlight: 1))
        XCTAssertEqual(changes, [status(inFlight: 1)])
    }

    func testStopMakesPendingTickAndLateCompletionInertAndClearsLatest() {
        let feed = makeFeed()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 1))
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2)
        feed.stop()
        XCTAssertNil(feed.latest)
        XCTAssertEqual(changes, [status(inFlight: 1), nil])
        fetcher.complete(1, with: status(inFlight: 9))
        XCTAssertNil(feed.latest, "a completion issued before stop is inert")
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2, "a tick scheduled before stop is inert")
        XCTAssertEqual(scheduler.pendingCount, 0, "no timer left after stop")
        XCTAssertEqual(changes.count, 2)
    }

    func testStopIsIdempotent() {
        let feed = makeFeed()
        feed.stop()
        feed.start()
        fetcher.complete(0, with: status(inFlight: 1))
        feed.stop()
        feed.stop()
        XCTAssertNil(feed.latest)
        XCTAssertEqual(changes, [status(inFlight: 1), nil])
    }

    func testCompletionAfterRebootWithDifferentTokenIsDropped() {
        let feed = makeFeed()
        feed.start()
        endpoint.value = (port: 8080, token: String(repeating: "b", count: 64))
        fetcher.complete(0, with: status(inFlight: 7))
        XCTAssertNil(feed.latest, "a reply from a previous boot never surfaces")
        XCTAssertTrue(changes.isEmpty)
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.count, 2, "the dropped reply frees the single flight")
        fetcher.complete(1, with: status(inFlight: 0))
        XCTAssertEqual(feed.latest, status(inFlight: 0))
    }

    func testCompletionAfterRebootOnDifferentPortIsDropped() {
        let feed = makeFeed()
        feed.start()
        endpoint.value = (port: 8081, token: String(repeating: "a", count: 64))
        fetcher.complete(0, with: status(inFlight: 7))
        XCTAssertNil(feed.latest)
        scheduler.fireAll()
        XCTAssertEqual(fetcher.calls.last?.port, 8081)
    }

    func testRestartAfterStopPollsAgain() {
        let feed = makeFeed()
        feed.start()
        feed.stop()
        XCTAssertEqual(fetcher.calls.count, 1)
        feed.start()
        XCTAssertEqual(fetcher.calls.count, 2, "restart polls immediately even though the old fetch never replied")
        scheduler.fireAll()
        XCTAssertEqual(scheduler.pendingCount, 1, "only the new loop reschedules")
        fetcher.complete(0, with: status(inFlight: 5))
        XCTAssertNil(feed.latest, "the pre-stop reply stays inert after restart")
        fetcher.complete(1, with: status(inFlight: 1))
        XCTAssertEqual(feed.latest, status(inFlight: 1))
    }

    func testCompletionIsHoppedThroughDeliver() {
        var hops: [() -> Void] = []
        let endpoint = self.endpoint!
        let fetcher = self.fetcher!
        let feed = ShellStatusFeed(
            endpoint: { endpoint.value },
            fetch: { port, token, done in fetcher.fetch(port, token, done) },
            schedule: { _, _ in },
            deliver: { hops.append($0) }
        )
        feed.start()
        fetcher.complete(0, with: status(inFlight: 1))
        XCTAssertNil(feed.latest, "not applied until delivered")
        XCTAssertEqual(hops.count, 1)
        hops[0]()
        XCTAssertEqual(feed.latest, status(inFlight: 1))
    }
}
