// The live-stream mock's mutable handle for the merged /stats page tests.
//
// NOT a test file. This module is deliberately a LEAF: it imports nothing
// but types, so the `vi.mock('@/lib/live-stats-stream', ...)` factory in
// each stats-merged-*.test.tsx can `await import()` it without pulling in
// the page.
//
// Why it is separate from stats-merged-harness.tsx: the harness imports
// `@/app/stats/page`, which imports `@/lib/live-stats-stream` — the module
// the factory is replacing. When the factory imported the harness for this
// handle, the harness was already mid-evaluation (test file → harness →
// page → live-stats-stream → factory), so the factory awaited the harness's
// pending module promise while the harness's own imports awaited the
// factory. Neither side could progress: an idle-CPU deadlock during test
// collection, before any test starts, which no `--testTimeout` can
// interrupt and which prints nothing. All six merged-page suites hung
// every run (41 min in CI against a ~40 s baseline) until this split.

import type { LiveEngineFrame } from "@/lib/live-stats";
import type { LiveStatsState } from "@/lib/live-stats-stream";

export const mockStream: { state: LiveStatsState } = {
  state: { status: "connecting", frame: null, errorCode: null },
};

export function setFrame(
  frame: LiveEngineFrame | null,
  status: LiveStatsState["status"] = "connected",
) {
  mockStream.state = { status, frame, errorCode: null };
}
