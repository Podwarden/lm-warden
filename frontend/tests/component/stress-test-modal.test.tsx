// Component tests for the stress-test modal.
//
// What warrants pinning down here is not the markup, it is the promises the
// design makes about this UI:
//
//   * the acknowledgement is an explicit choice, and the POST body carries it
//   * no mode is presented as crash-free
//   * progress is POLLED, and an indeterminate bar omits aria-valuenow
//   * the result LEADS with a setting, marks recommended_config as not-live,
//     and never shows a limit as a single scalar
//   * a failure says WHICH failure, because the remedies differ
//   * closing resets, so the next open is a fresh confirm screen
//
// Fetch is stubbed globally so the auth-fetch wrapper runs end to end,
// mirroring delete-model-modal.test.tsx.
//
// ---------------------------------------------------------------------------
// NO WALL CLOCK ANYWHERE IN THIS FILE.
// ---------------------------------------------------------------------------
// This suite has failed twice in CI on a contended runner, both times on an
// assertion whose correctness depended on how much real time had passed, and
// both times it was retried green rather than investigated. Neither failure
// was a defect in the component.
//
// Every wait here is therefore expressed as "yield until the thing I already
// know has started has finished", with an ITERATION ceiling and no time
// budget. `waitFor` and `findBy*` are deliberately absent: their 1 s
// real-timer budget is the wrong instrument for that question, because a
// descheduled worker expires the budget without the work ever being late.
//
// The mirror hazard is just as real and is what bit the second time: an
// assertion is also unsound when the state it names is only true for a
// BOUNDED window of real time. Fixtures here therefore hold a state open
// until the test has finished looking at it, rather than letting the next
// poll take it away. See the transient-poll-failure test.
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  act,
  render,
  screen,
  fireEvent,
  cleanup,
  within,
} from "@testing-library/react";
import { StressTestModal } from "@/components/stress/stress-test-modal";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface Call {
  url: string;
  method: string;
  body: string | null;
}

function installFetchStub(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response>,
): Call[] {
  const calls: Call[] = [];
  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push({
      url,
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? init.body : null,
    });
    if (url === "/api/auth/refresh") return json({ access_token: "jwt2" });
    if (url === "/api/csrf") return json({ csrf: "test-csrf" });
    return handler(url, init);
  });
  vi.stubGlobal("fetch", mock);
  return calls;
}

function renderModal(props: Record<string, unknown> = {}) {
  return render(
    <StressTestModal
      open
      onClose={() => {}}
      modelId="m-1"
      servedModelName="qwen3-8b"
      maxModelLen={32768}
      // Fast enough that a handful of yields covers several polls without
      // fake timers.
      pollIntervalMs={5}
      {...props}
    />,
  );
}

/**
 * Yield the event loop once and flush React's response to whatever settled.
 *
 * The point is what it does NOT have: a deadline. `waitFor` gives up after a
 * budget of real time, which is the wrong instrument for "the work I already
 * know has started has finished" — a descheduled CI worker expires that budget
 * without the work ever having been late. A `setTimeout(0)` drains the pending
 * microtasks (fetch → Response → json → setState) and the `act` flush commits
 * the render; a stalled process merely arrives at it later.
 *
 * Note the shape: the yield is OUTSIDE `act`, and the flush is `act`'s
 * SYNCHRONOUS form. The obvious `await act(async () => { await sleep(0) })`
 * looks equivalent and is not — its async form refuses to resolve until it
 * observes an empty act queue, and this modal refills that queue on every
 * turn of the loop as soon as its poll timer is chronically overdue, which is
 * the normal condition on a contended runner with a 5 ms interval. Measured:
 * with every poll retry late, the async form never returns and the test dies
 * on the runner's own timeout after 60 s; this form passes with each poll
 * 2000 ms late, just slower. The synchronous flush drains what is queued now
 * and returns, so it is bounded by construction. Updates that land during the
 * unwrapped yield are picked up by the very next flush, and React emits no
 * act warning for them (verified: zero across the file).
 */
async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  act(() => {});
}

/**
 * How many times `settleUntil` will yield before giving up.
 *
 * This is a COUNT OF TURNS, not a time budget, and that distinction is the
 * whole point. A descheduled worker needs exactly as many turns as a fast one
 * — each turn simply takes longer in wall clock — so no amount of scheduling
 * delay can exhaust this, and a stall can only make the suite slower, never
 * red. The ceiling exists solely so a genuinely broken component fails loudly
 * here instead of hanging until the runner's own job timeout.
 *
 * Measured turns needed by the conditions in this file: 7 at worst (three
 * failing polls), then 5, 4, 2, and 1 or 0 for the rest — and IDENTICAL under
 * 24 busy loops on 10 cores, which is the evidence for the paragraph above.
 * The 200 is therefore ~28x the worst case, not a budget anyone is spending.
 */
const MAX_SETTLES = 200;

/**
 * Yield until `holds()` is true, or fail after MAX_SETTLES turns.
 *
 * Only ever point this at a MONOTONE condition — one that, once true, stays
 * true for the rest of the test. A condition that flips back (the poll-error
 * banner, which the next successful poll clears) can be missed between two
 * turns, and hardening the wait does not fix that: the fixture has to hold
 * the state open instead.
 */
async function settleUntil(holds: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < MAX_SETTLES; i += 1) {
    if (holds()) return;
    await settle();
  }
  if (holds()) return;
  throw new Error(
    `settleUntil: ${what} — still false after ${MAX_SETTLES} yields. ` +
      `This is a stuck component, not a slow one: the ceiling counts turns ` +
      `of the event loop, not milliseconds.`,
  );
}

/** Convenience: the commonest monotone condition in this file. */
function present(testId: string): () => boolean {
  return () => screen.queryByTestId(testId) !== null;
}

/** No runs at all — the usual state when the modal first opens. */
const NO_RUNS = { limits: {}, recommended_config: null, runs: [] };

describe("StressTestModal — confirm phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("keeps Start disabled until disruption is explicitly acknowledged", async () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();

    const ack = screen.getByTestId("stress-acknowledge") as HTMLInputElement;
    // Explicitly NOT pre-ticked: a default-on acknowledgement is not one.
    expect(ack.checked).toBe(false);
    expect(screen.getByTestId("stress-start")).toBeDisabled();

    fireEvent.click(ack);
    expect(screen.getByTestId("stress-start")).not.toBeDisabled();

    // The mount-time adopt probe is still in flight. Drain it inside `act`
    // rather than leaving it to resolve after `cleanup()` — an unowned
    // setState landing in the next test's window is exactly the kind of
    // cross-test ordering this file must not depend on.
    await settle();
  });

  it("never presents a mode as crash-free, and shows honest estimates", async () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();

    // The warning is unconditional — not gated on which mode is selected.
    expect(screen.getByTestId("stress-disruption-warning")).toHaveTextContent(
      /may crash and be restarted/i,
    );
    expect(screen.getByTestId("stress-disruption-warning")).toHaveTextContent(
      /No mode can promise otherwise/i,
    );
    // v1 called the default mode "safe" and promised no crash. Nothing in the
    // confirm screen may say that, in any mode.
    const body = screen.getByTestId("stress-disruption-warning").textContent ?? "";
    expect(body).not.toMatch(/never crash|no crash|crash-free|safe mode/i);

    expect(screen.getByText("~15 min")).toBeInTheDocument();
    expect(screen.getByText("~30–45 min")).toBeInTheDocument();
    expect(screen.getByText("2.5–4 h")).toBeInTheDocument();

    await settle(); // see the note in the test above
  });

  it("POSTs the selected mode with acknowledge_disruption and force:false", async () => {
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();

    fireEvent.click(screen.getByTestId("stress-mode-thorough"));
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));

    // Monotone: with no matching run row the modal never leaves `running`.
    await settleUntil(present("stress-running"), "the running phase");
    const post = calls.find((c) => c.method === "POST" && c.url.endsWith("/stress"));
    expect(post).toBeDefined();
    expect(JSON.parse(post!.body!)).toEqual({
      mode: "thorough",
      acknowledge_disruption: true,
      force: false,
      // An ordinary start never wipes. `reset` is set only by "Wipe and run
      // again" on the results screen: the published record deliberately falls
      // back past a run that measured nothing, so a FAILED run must not
      // withdraw a good older measurement, and only the operator can tell that
      // case from a deliberate replacement.
      reset: false,
    });
  });

  it("reveals the force affordance only after the server refuses with 409", async () => {
    let refused = false;
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json(NO_RUNS);
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        if (!refused) {
          refused = true;
          return json({ detail: "model is serving traffic" }, 409);
        }
        return json({ run_id: "r-9" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();

    // `force` is not an ambient checkbox — it does not exist until refused.
    expect(screen.queryByTestId("stress-force")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));

    // Monotone: `conflict` is only ever cleared by another Start.
    await settleUntil(present("stress-conflict"), "the 409 conflict notice");
    expect(screen.getByTestId("stress-conflict")).toHaveTextContent(
      /serving traffic/i,
    );
    // Refused and not yet forced: Start must not fire the same request again.
    expect(screen.getByTestId("stress-start")).toBeDisabled();

    fireEvent.click(screen.getByTestId("stress-force"));
    fireEvent.click(screen.getByTestId("stress-start"));

    await settleUntil(present("stress-running"), "the running phase after force");
    const posts = calls.filter(
      (c) => c.method === "POST" && c.url.endsWith("/stress"),
    );
    expect(posts).toHaveLength(2);
    expect(JSON.parse(posts[1].body!).force).toBe(true);
  });
});

describe("StressTestModal — running phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function startRun(
    capabilitiesFor: (n: number) => unknown,
  ): Promise<Call[]> {
    let polls = 0;
    const calls = installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        const body = capabilitiesFor(polls);
        polls += 1;
        return json(body);
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await settleUntil(present("stress-running"), "the running phase");
    return calls;
  }

  it("polls the run row (no SSE) and renders probe outcomes as badges", async () => {
    const calls = await startRun((n) => ({
      runs: [
        {
          id: "r-1",
          status: "running",
          progress: { completed: n, total: 12, phase: "bracket" },
          observations: [
            { axis: "length", value: 1024, cls: "pass" },
            { axis: "length", value: 2048, cls: "degraded", gate_tripped: "abrupt" },
          ],
        },
      ],
    }));

    // Monotone: every poll returns the same two observations, so once the
    // rows are on screen they stay on screen.
    await settleUntil(
      () => screen.queryAllByTestId("stress-probe-row").length === 2,
      "both probe rows",
    );
    expect(screen.getByText("pass")).toBeInTheDocument();
    expect(screen.getByText("degraded")).toBeInTheDocument();
    expect(screen.getByTestId("stress-phase")).toHaveTextContent("bracket");

    // More than one poll, and every one of them a plain GET — the design
    // rejects SSE replay for progress outright. A call count only ever grows,
    // so this too is monotone.
    const pollsSoFar = () => calls.filter((c) => c.url.endsWith("/capabilities"));
    await settleUntil(() => pollsSoFar().length > 1, "a second poll");
    expect(pollsSoFar().every((c) => c.method === "GET")).toBe(true);
    expect(vi.isMockFunction(globalThis.EventSource)).toBe(false);
  });

  it("omits aria-valuenow while no probe total is known", async () => {
    await startRun(() => ({
      runs: [{ id: "r-1", status: "running", observations: [] }],
    }));
    await settleUntil(
      () => screen.queryByRole("progressbar") !== null,
      "the progress bar",
    );
    // Indeterminate ARIA: announcing 0% would be a lie, not a default. No
    // poll in this stub ever carries a total, so the absence is permanent —
    // this is not a snapshot of a passing moment.
    expect(screen.getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");

    cleanup();

    await startRun(() => ({
      runs: [
        {
          id: "r-1",
          status: "running",
          progress: { completed: 3, total: 12 },
          observations: [],
        },
      ],
    }));
    await settleUntil(
      () =>
        screen.queryByRole("progressbar")?.getAttribute("aria-valuenow") === "25",
      "aria-valuenow=25",
    );
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  });

  // ANCHORED ON THE POLLS, NOT ON A WALL CLOCK — IN BOTH DIRECTIONS.
  //
  // First failure (pipeline 21384): the recovery half waited for the banner to
  // CLEAR with `waitFor`, whose 1 s budget is real time, and real time on a CI
  // runner is not the test's to spend. That host carried three other jobs
  // (every phase 2-3x the green runs -- environment 140 s vs 47 s), the worker
  // was descheduled for seconds, and the budget expired while the successful
  // poll was still sitting in the timer queue. Anchoring on the poll and
  // yielding with `settle()` fixed that half, and it has held.
  //
  // Second failure: the ARRIVAL half, on a 503 that the fixture served to
  // exactly ONE poll. That made the banner true for exactly one poll interval
  // — 5 ms here — and put the assertion in a race against the modal's own
  // recovery. Measured: a deschedule of 5 ms or more between the failing poll
  // scheduling its retry and the timer phase running is enough for the
  // RECOVERY poll to fire inside the same `settle()`, so the banner arrives
  // and is wiped before the assertion ever reads it. That is a 5 ms
  // wall-clock deadline hidden in the fixture — two orders of magnitude
  // tighter than the 1 s budget the first fix removed, and hardening the wait
  // could not have fixed it: waiting LONGER makes it strictly more likely.
  //
  // So the failure is LATCHED. The stub keeps refusing until the test has
  // finished looking at the banner, and only then recovers. Neither state can
  // now expire under the test's feet, so no amount of scheduling delay can
  // turn "not yet" into "never" or "already gone".
  //
  // Which poll fails is a flag rather than an ordinal, too: the old
  // `polls === 2` silently assumed the mount-time adopt probe was poll 1, so
  // any change to how the modal opens would have moved the 503 onto a poll
  // this test was not talking about.
  it("keeps polling through a transient poll failure rather than declaring failure", async () => {
    let failing = false;
    let failedPolls = 0;
    let okPollsSinceFailure = 0;

    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        // The warden restarts the engine after a crash; the API can blip.
        if (failing) {
          failedPolls += 1;
          return json({ detail: "backend restarting" }, 503);
        }
        if (failedPolls > 0) okPollsSinceFailure += 1;
        return json({ runs: [{ id: "r-1", status: "running", observations: [] }] });
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await settleUntil(present("stress-running"), "the running phase");

    // ARRIVAL. Monotone while the latch is closed: nothing but a successful
    // poll clears `pollError`, and no poll can succeed until the test says so.
    failing = true;
    await settleUntil(present("stress-poll-error"), "the poll-error banner");
    expect(screen.getByTestId("stress-poll-error")).toBeInTheDocument();
    // Still running — a poll that could not be answered is not a run that
    // failed, which is the actual claim of this test.
    expect(screen.getByTestId("stress-running")).toBeInTheDocument();

    // KEEPS POLLING. The old fixture could not assert this at all: it served
    // one 503 and then recovered, so "it tried again" and "it gave up" looked
    // identical. A failing poll count only grows, so this is monotone too.
    await settleUntil(() => failedPolls >= 3, "three failing polls");
    expect(screen.getByTestId("stress-running")).toBeInTheDocument();
    expect(screen.queryByTestId("stress-failed")).not.toBeInTheDocument();
    expect(screen.getByTestId("stress-poll-error")).toBeInTheDocument();

    // RECOVERY. Monotone in the other direction once the latch opens: every
    // poll from here on succeeds, so `pollError` goes null and stays null.
    failing = false;
    await settleUntil(() => okPollsSinceFailure > 0, "a poll to succeed again");
    await settleUntil(
      () => screen.queryByTestId("stress-poll-error") === null,
      "the poll-error banner to clear",
    );
    expect(screen.queryByTestId("stress-poll-error")).not.toBeInTheDocument();
    expect(screen.getByTestId("stress-running")).toBeInTheDocument();
  });

  it("adopts a run that is already in flight instead of offering to start another", async () => {
    installFetchStub((url) => {
      if (url.endsWith("/capabilities")) {
        return json({
          runs: [{ id: "r-existing", status: "running", observations: [] }],
        });
      }
      throw new Error(`Unmocked: ${url}`);
    });
    renderModal();
    await settleUntil(present("stress-running"), "the adopted running phase");
    expect(screen.queryByTestId("stress-start")).not.toBeInTheDocument();
  });
});

describe("StressTestModal — done phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  function completedRun(overrides: Record<string, unknown> = {}) {
    return {
      id: "r-1",
      status: "completed",
      limits: {
        quality_context: {
          value: 122880,
          unit: "tokens",
          provenance: "measured",
          limited_by: "quality",
          gate_tripped: "abrupt",
          raw_confirmed: 136533,
          first_observed_failure: 147456,
          oracle: { n_consecutive: 3, contour_p: 0.21, confirm_m: 5, safety_factor: 0.9 },
        },
      },
      recommended_config: {
        max_model_len: 65536,
        limited_by: "load",
        next_failed_at: 98304,
      },
      observations: [],
      ...overrides,
    };
  }

  async function renderDone(run: Record<string, unknown>) {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json({ runs: [run] });
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    // Through the real flow: start, then let the first poll observe a run row
    // that is no longer `running`. `done` is terminal — the poll loop stops
    // rescheduling once it sees a non-running status — so this is monotone.
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await settleUntil(present("stress-done"), "the done phase");
  }

  it("leads with the setting to apply, and says the config is not the live one", async () => {
    await renderDone(completedRun());
    // The headline is an action, not a breaking point.
    expect(screen.getByTestId("stress-recommendation")).toHaveTextContent(
      "Set max_model_len to 65,536",
    );
    expect(screen.getByTestId("stress-not-live-caveat")).toHaveTextContent(
      /not currently running/i,
    );
    // limited_by drives the remedy: a memory wall is not a quality wall.
    expect(screen.getByTestId("stress-limited-by")).toHaveTextContent(
      "memory at load",
    );
    expect(screen.getByText(/could not allocate its KV pool/i)).toBeInTheDocument();
  });

  it("shows all three numbers for a limit, never a single scalar", async () => {
    await renderDone(completedRun());
    const row = screen.getByTestId("stress-limit-quality_context");
    expect(row).toHaveTextContent("122,880");
    expect(row).toHaveTextContent("136,533"); // raw_confirmed
    expect(row).toHaveTextContent("147,456"); // first_observed_failure
    expect(row).toHaveTextContent("3/3 consecutive");
  });

  it("falls back to the measured context when no reload sweep ran", async () => {
    // conservative/quick modes never sweep, so there is no better config to
    // recommend — and inventing one would be worse than saying so.
    await renderDone(completedRun({ recommended_config: null }));
    expect(screen.getByTestId("stress-recommendation")).toHaveTextContent(
      "Keep prompts under 122,880 tokens",
    );
    expect(screen.queryByTestId("stress-not-live-caveat")).not.toBeInTheDocument();
    expect(screen.getByText(/thorough/i)).toBeInTheDocument();
  });

  it("flags a completed-but-truncated run as a lower bound", async () => {
    await renderDone(completedRun({ truncated_by: "crash_budget" }));
    expect(screen.getByTestId("stress-truncated")).toHaveTextContent(
      /lower bound/i,
    );
  });
});

describe("StressTestModal — failure phase", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function renderFailed(run: Record<string, unknown>) {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json({ runs: [run] });
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    // Terminal, like `done`: the poll loop stops once the status is not
    // `running`.
    await settleUntil(present("stress-failed"), "the failed phase");
  }

  it("names a non-monotone verdict as unsound, not as a limit", async () => {
    await renderFailed({
      id: "r-1",
      status: "aborted",
      non_monotone: true,
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /unsound/i,
    );
    expect(screen.getByTestId("stress-failed")).toHaveTextContent(
      /not upward-closed/i,
    );
  });

  it("distinguishes a neighbour impact from a wedged engine", async () => {
    await renderFailed({
      id: "r-1",
      status: "aborted",
      truncated_by: "neighbour_impact",
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /co-resident/i,
    );
    cleanup();

    await renderFailed({
      id: "r-1",
      status: "aborted",
      truncated_by: "wedged",
      observations: [],
    });
    expect(screen.getByTestId("stress-failure-title")).toHaveTextContent(
      /wedged/i,
    );
  });

  it("says an interrupted run published nothing", async () => {
    await renderFailed({
      id: "r-1",
      status: "interrupted",
      observations: [],
      last_error: null,
    });
    expect(screen.getByTestId("stress-failed")).toHaveTextContent(
      /never published/i,
    );
  });
});

describe("StressTestModal — close", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("resets phase state so the next open is a fresh confirm screen", async () => {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) {
        return json({
          runs: [
            { id: "r-1", status: "completed", limits: {}, observations: [] },
          ],
        });
      }
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    function modal(open: boolean) {
      return (
        <StressTestModal
          open={open}
          onClose={() => {}}
          modelId="m-1"
          servedModelName="qwen3-8b"
          maxModelLen={32768}
          pollIntervalMs={5}
        />
      );
    }
    const { rerender } = render(modal(true));
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await settleUntil(present("stress-done"), "the done phase");

    // Scoped to the result body: the Modal chrome has its own "Close".
    fireEvent.click(
      within(screen.getByTestId("stress-done")).getByRole("button", {
        name: "Close",
      }),
    );
    rerender(modal(false));
    rerender(modal(true));
    // A finished run is not adopted on reopen — only a running one is — so the
    // operator lands back on the confirm screen with the box unticked.
    //
    // The reopen fires a fresh adopt probe, and the confirm screen is on
    // screen BEFORE that probe answers. Asserting immediately would therefore
    // pass even if adoption were broken and about to replace it, so drain the
    // probe first and assert on the state that survives it.
    await settleUntil(present("stress-acknowledge"), "the confirm screen");
    await settle();
    expect(screen.getByTestId("stress-acknowledge")).toBeInTheDocument();
    expect(screen.getByTestId("stress-acknowledge")).not.toBeChecked();
  });
});

// ---------------------------------------------------------------------------
// #262/#265 — "Apply and reload" goes through the model write path, which
// validates the WHOLE merged row, so this button can now answer 422 over a
// column the operator never touched. The modal has to say which one: this is
// the single most consequential button in the app, and it used to render a
// bare `HTTP 422`.
// ---------------------------------------------------------------------------

describe("StressTestModal — apply failures name the problem", () => {
  beforeEach(() => {
    setAccessToken("test-jwt", 900);
    setCsrfToken("test-csrf");
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  const RUN = {
    id: "r-1",
    status: "completed",
    limits: {
      quality_context: {
        value: 122880,
        unit: "tokens",
        provenance: "measured",
        limited_by: "quality",
        gate_tripped: "abrupt",
        raw_confirmed: 136533,
        first_observed_failure: 147456,
        oracle: { n_consecutive: 3, contour_p: 0.21, confirm_m: 5, safety_factor: 0.9 },
      },
    },
    recommended_config: { max_model_len: 65536, limited_by: "load", next_failed_at: 98304 },
    observations: [],
  };

  async function applyAgainst(failure: Response) {
    installFetchStub((url, init) => {
      if (url.endsWith("/capabilities")) return json({ runs: [RUN] });
      if (url === "/api/models/m-1/stress" && init?.method === "POST") {
        return json({ run_id: "r-1" }, 202);
      }
      if (url === "/api/models/m-1/stress/apply") return failure.clone();
      throw new Error(`Unmocked: ${init?.method} ${url}`);
    });
    renderModal();
    fireEvent.click(screen.getByTestId("stress-acknowledge"));
    fireEvent.click(screen.getByTestId("stress-start"));
    await settleUntil(present("stress-done"), "the done phase");
    fireEvent.click(screen.getByTestId("stress-apply"));
    await settleUntil(present("stress-apply-error"), "the apply error");
    return screen.getByTestId("stress-apply-error");
  }

  it("names the offending column when the merged row is refused", async () => {
    const err = await applyAgainst(
      new Response(
        JSON.stringify({
          detail: [
            {
              loc: [],
              msg: "Value error, extra_env key 'FOO_BAR' is not in the allowlist for backend 'vllm'",
              type: "value_error",
            },
          ],
        }),
        { status: 422 },
      ),
    );
    expect(err.textContent).toMatch(/FOO_BAR/);
    expect(err.textContent).not.toMatch(/HTTP 422/);
    expect(err.textContent).not.toMatch(/object Object/);
  });

  it("still renders the structured {message} refusals", async () => {
    const err = await applyAgainst(
      new Response(
        JSON.stringify({
          detail: {
            error_code: "run_already_active",
            message: "a stress run holds this model",
          },
        }),
        { status: 409 },
      ),
    );
    expect(err.textContent).toMatch(/a stress run holds this model/);
  });

  it("falls back to the status code only when there is no detail at all", async () => {
    const err = await applyAgainst(new Response("not json", { status: 500 }));
    expect(err.textContent).toMatch(/HTTP 500/);
  });
});
