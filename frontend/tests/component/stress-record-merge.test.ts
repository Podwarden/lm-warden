// Attaching the published record to the run it came from.
//
// Found in production on 2026-09-04: a completed stress run rendered
// "No recommendation. Nothing met the bar for publication." even when it had
// measured something. `GET /capabilities` returns `limits` and
// `recommended_config` at the TOP level; `runs[]` carries only the summary
// (`_run_summary` in app/stress/routes_api.py omits both). `StressResult`
// reads them off the run object, so they were always undefined.
//
// The id guard is the load-bearing part. The published record is keyed on the
// FINGERPRINT, not on the latest run, so after a run that published nothing
// the record still holds an EARLIER run's numbers. Merging unconditionally
// would present those as the finished run's own findings — the precise class
// of confident-but-wrong output this whole feature is built to avoid.
import { describe, it, expect } from "vitest";
import { withRecord } from "@/components/stress/stress-test-modal";
import type { CapabilitiesResponse, StressRun } from "@/components/stress/types";

const run = { id: "run-1", status: "completed" } as StressRun;
const limits = { quality_context: { value: 32768 } } as unknown as CapabilitiesResponse["limits"];
const rec = { max_model_len: 32768 } as unknown as CapabilitiesResponse["recommended_config"];

describe("withRecord", () => {
  it("attaches the record to the run that produced it", () => {
    const got = withRecord(run, { record_run_id: "run-1", limits, recommended_config: rec });
    expect(got.limits).toEqual(limits);
    expect(got.recommended_config).toEqual(rec);
  });

  it("never attributes an earlier run's numbers to this one", () => {
    const got = withRecord(run, { record_run_id: "run-0", limits, recommended_config: rec });
    expect(got.limits ?? null).toBeNull();
    expect(got.recommended_config ?? null).toBeNull();
  });

  it("leaves the run alone when nothing is published for this fingerprint", () => {
    const got = withRecord(run, { record_run_id: null, limits: null, recommended_config: null });
    expect(got).toBe(run);
  });

  it("does not mutate the run it was given", () => {
    const got = withRecord(run, { record_run_id: "run-1", limits, recommended_config: rec });
    expect(got).not.toBe(run);
    expect(run.limits ?? null).toBeNull();
  });
});
