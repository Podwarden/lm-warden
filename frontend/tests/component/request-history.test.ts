// Pure helpers behind the requests chart: what colour encodes, mark sizing,
// log ticks, and the wording of what a window covers.

import { describe, it, expect } from "vitest";
import {
  GROUP_CLASSES,
  OTHER_CLASS,
  OTHER_LABEL,
  classFor,
  clientOf,
  colourGroups,
  coverageNote,
  finishClassOf,
  formatSpan,
  logTicks,
  markRadius,
  pickColourField,
  type RequestHistoryRow,
} from "@/lib/request-history";

function row(over: Partial<RequestHistoryRow> = {}): RequestHistoryRow {
  return {
    id: "r", finished_at: 1000, model_id: "id-model-a", model: "model-a",
    token_name: "key-a", token_id: "tok-a", client_ip: "10.0.0.1", prompt_tokens: 10,
    completion_tokens: 5, duration_s: 1, ttft_s: 0.1, finish_reason: "stop",
    orphan: false, started_iso: "x", queued_s: 0,
    ...over,
  };
}

describe("pickColourField", () => {
  it("is nothing for no rows", () => {
    expect(pickColourField([])).toBe("none");
  });
  it("is client when at least two share the rows", () => {
    expect(pickColourField([row(), row({ token_name: "key-b" })])).toBe("client");
  });
  it("is not client when one client dominates (95%+)", () => {
    const rows = [...Array(40)].map(() => row());
    rows.push(row({ token_name: "key-b" }));
    // 40 of 41 = 97.6%: colour by client would paint one series with noise.
    expect(pickColourField(rows)).toBe("none");
    expect(pickColourField([...rows, row({ model: "model-b" })])).toBe("model");
  });
  it("is model when clients are one but models are several", () => {
    expect(pickColourField([row(), row({ model: "model-b" })])).toBe("model");
  });
  it("groups an anonymous request by IP, then by 'anonymous'", () => {
    expect(clientOf({ token_name: null, client_ip: "10.0.0.9" })).toBe("10.0.0.9");
    expect(clientOf({ token_name: null, client_ip: null })).toBe("anonymous");
  });
});

describe("colourGroups", () => {
  it("orders by count then name and folds the tail into 'other'", () => {
    const rows: RequestHistoryRow[] = [];
    for (let i = 0; i < 8; i++) {
      for (let k = 0; k <= i; k++) rows.push(row({ token_name: `key-${i}` }));
    }
    const groups = colourGroups(rows, "client");
    expect(groups.map((g) => g.key)).toEqual([
      "key-7", "key-6", "key-5", "key-4", "key-3", "key-2", OTHER_LABEL,
    ]);
    expect(groups.slice(0, 6).map((g) => g.className)).toEqual([...GROUP_CLASSES]);
    expect(groups[6].className).toBe(OTHER_CLASS);
    expect(groups[6].count).toBe(1 + 2); // key-0 + key-1
    expect(classFor(groups, "key-0")).toBe(OTHER_CLASS);
    expect(classFor(groups, "key-7")).toBe(GROUP_CLASSES[0]);
  });
  it("is one accent group when colour encodes nothing", () => {
    expect(colourGroups([row()], "none")).toEqual([
      { key: "all", count: 1, className: GROUP_CLASSES[0] },
    ]);
    expect(colourGroups([], "none")).toEqual([]);
  });
});

describe("marks", () => {
  it("folds finish reasons to three shapes", () => {
    expect(finishClassOf("stop")).toBe("stop");
    expect(finishClassOf("length")).toBe("length");
    expect(finishClassOf("runaway")).toBe("other");
    expect(finishClassOf(null)).toBe("other");
  });
  it("sizes by the square root of generated tokens within 2..6 px", () => {
    expect(markRadius(0, 100)).toBe(2);
    expect(markRadius(100, 100)).toBe(6);
    expect(markRadius(25, 100)).toBe(4);
    expect(markRadius(500, 100)).toBe(6); // clamped
    expect(markRadius(10, 0)).toBe(2);
  });
  it("log ticks cover the range in decades", () => {
    expect(logTicks(0.3, 45)).toEqual([0.1, 1, 10, 100]);
    expect(logTicks(0.001, 0.5)).toEqual([0.01, 0.1, 1]); // floor at 10 ms
    expect(logTicks(5, 0)).toEqual([1]);
  });
});

describe("wording", () => {
  it("formats spans", () => {
    expect(formatSpan(45)).toBe("45 s");
    expect(formatSpan(600)).toBe("10 min");
    expect(formatSpan(3 * 3600 + 12 * 60)).toBe("3 h 12 min");
    expect(formatSpan(6 * 86400 + 4 * 3600)).toBe("6 d 4 h");
    expect(formatSpan(null)).toBe("—");
  });
  const cov = { earliest_epoch: 1000, retention_days: 30, max_rows: 1, covers_window: true };
  it("says nothing when the window is covered", () => {
    expect(coverageNote(cov, 500, 2000, "last hour")).toBeNull();
    expect(coverageNote(undefined, 500, 2000, "last hour")).toBeNull();
  });
  it("blames retention when it is shorter than the window", () => {
    const c = { ...cov, retention_days: 1, covers_window: false };
    const now = 10 * 86400;
    expect(coverageNote(c, now - 7 * 86400, now, "last 7 days")).toMatch(
      /retention is 1 d, shorter than the last 7 days/,
    );
  });
  it("says when history begins otherwise", () => {
    const c = { ...cov, earliest_epoch: 2000 - 3 * 3600, covers_window: false };
    expect(coverageNote(c, 2000 - 86400, 2000, "last 24 hours")).toBe(
      "history begins 3 h ago — covers 3 h of the last 24 hours",
    );
  });
  it("says so when nothing was ever recorded", () => {
    const c = { ...cov, earliest_epoch: null, covers_window: false };
    expect(coverageNote(c, 1000, 2000, "last hour")).toBe("no requests recorded yet");
  });
});
