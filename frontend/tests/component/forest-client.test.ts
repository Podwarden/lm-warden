import { afterEach, describe, expect, it, vi } from "vitest";
import { applyResponse, forestFetcher, ForestAuthError, forestUrl, pollForest, type ForestResponse } from "@/lib/forest/client";

const tree = (id: string, start: number, last_at: number) => ({
  id, key: "k", start, end: start + 10, last: last_at - 1000, last_at,
  traits: { sys0: 1, ctx_gen: 1, mean_gen: 1, thrash: 0, fail: 0, fanout: 0 }, sessions: [],
});
const resp = (o: Partial<ForestResponse>): ForestResponse => ({
  range: "6h", t0: 1000, now: 1100, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 1090, ...o,
});

describe("applyResponse", () => {
  it("full response replaces state", () => {
    const { state, refetchFull } = applyResponse(null, resp({ trees: [tree("a", 5, 1050)], ids: ["a"] }));
    expect(refetchFull).toBe(false);
    expect([...state.trees.keys()]).toEqual(["a"]);
  });

  it("delta replaces by id, drops ids not listed, rebases by Δt0", () => {
    const s1 = applyResponse(null, resp({ trees: [tree("a", 5, 1050), tree("b", 20, 1060)], ids: ["a", "b"] })).state;
    const r2 = resp({ t0: 1010, full: false, trees: [tree("b", 10, 1095)], ids: ["b"] });
    const { state, refetchFull } = applyResponse(s1, r2);
    expect(refetchFull).toBe(false);
    expect([...state.trees.keys()]).toEqual(["b"]);
    expect(state.t0).toBe(1010);
    expect(state.trees.get("b")!.start).toBe(10);
  });

  it("rebases a held tree by Δt0", () => {
    const s1 = applyResponse(null, resp({ trees: [tree("a", 25, 1050)], ids: ["a"] })).state;
    const { state } = applyResponse(s1, resp({ t0: 1010, full: false, ids: ["a"] }));
    expect(state.trees.get("a")!.start).toBe(15);
    expect(state.trees.get("a")!.end).toBe(25);
  });

  it("refetches when full or unknown id", () => {
    const s1 = applyResponse(null, resp({ trees: [tree("a", 5, 1050)], ids: ["a"] })).state;
    expect(applyResponse(s1, resp({ full: false, ids: ["a", "zz"] })).refetchFull).toBe(true);
  });

  it("flowers starting after since-600 are replaced, older ones kept", () => {
    const prev = { ...applyResponse(null, resp({ flowers: [[10, 1, 1], [80, 2, 2]] })).state, cursor: 1650 };
    const r2 = resp({ t0: 1000, full: false, cursor: 1655, flowers: [[85, 3, 3]] });
    expect(applyResponse(prev, r2).state.flowers).toEqual([[10, 1, 1], [85, 3, 3]]);
  });

  it("kept flowers are rebased when t0 moves", () => {
    const prev = { ...applyResponse(null, resp({ flowers: [[10, 1, 1], [80, 2, 2]] })).state, cursor: 1650 };
    const r2 = resp({ t0: 1010, full: false, cursor: 1655, flowers: [[75, 3, 3]] });
    expect(applyResponse(prev, r2).state.flowers).toEqual([[0, 1, 1], [75, 3, 3]]);
  });
});

describe("applyResponse: now is relative to t0 (the wire sends it absolute)", () => {
  it("full response: now = r.now − r.t0", () => {
    expect(applyResponse(null, resp({ t0: 1000, now: 1100 })).state.now).toBe(100);
  });
  it("delta whose t0 moved: now is relative to the new t0", () => {
    const s1 = applyResponse(null, resp({ t0: 1000, now: 1100, trees: [tree("a", 5, 1050)], ids: ["a"] })).state;
    const s2 = applyResponse(s1, resp({ t0: 1010, now: 1102, full: false, ids: ["a"], cursor: 1095 })).state;
    expect(s2.t0).toBe(1010);
    expect(s2.now).toBe(92);
  });
});

describe("forestFetcher forest-token", () => {
  afterEach(() => { vi.unstubAllGlobals(); sessionStorage.clear(); });
  it.each([401, 403])("throws ForestAuthError on %i", async (status) => {
    sessionStorage.setItem("vw-forest-token", "t");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() + 60_000));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ status, ok: false }));
    await expect(forestFetcher("forest-token")("/api/stats/forest")).rejects.toBeInstanceOf(ForestAuthError);
  });
});

describe("pollForest", () => {
  it("first poll is a full fetch of the full window's default range (24h); later polls pass the since cursor", async () => {
    const get = vi.fn(async (_url: string) => resp({ range: "24h", trees: [tree("a", 5, 1050)], ids: ["a"] }));
    const s1 = await pollForest(null, get);
    expect(get).toHaveBeenLastCalledWith("/api/stats/forest?range=24h");
    get.mockResolvedValueOnce(resp({ range: "24h", full: false, ids: ["a"], cursor: 1095 }));
    const s2 = await pollForest(s1, get);
    expect(get).toHaveBeenLastCalledWith("/api/stats/forest?range=24h&since=1090");
    expect(s2.cursor).toBe(1095);
    expect(forestUrl()).toBe("/api/stats/forest?range=24h");
  });

  it("a given range (the Stats card's 6h) is used for the full fetch, the delta and the refetch", async () => {
    const s1 = applyResponse(null, resp({ trees: [tree("a", 5, 1050)], ids: ["a"] })).state;
    const get = vi.fn(async (url: string) =>
      url.includes("since") ? resp({ full: false, ids: ["a", "zz"], cursor: 1095 }) : resp({ trees: [], ids: [] }),
    );
    await pollForest(s1, get, "6h");
    expect(get.mock.calls.map((c) => c[0])).toEqual([
      "/api/stats/forest?range=6h&since=1090",
      "/api/stats/forest?range=6h",
    ]);
    expect(forestUrl(undefined, "6h")).toBe("/api/stats/forest?range=6h");
  });

  it("refetchFull runs exactly one full fetch, never a loop", async () => {
    const s1 = applyResponse(null, resp({ range: "24h", trees: [tree("a", 5, 1050)], ids: ["a"] })).state;
    const get = vi.fn(async (url: string) =>
      url.includes("since")
        ? resp({ full: false, ids: ["a", "zz"], cursor: 1095 })
        // even a (malformed) full refetch that again names unknown ids must not trigger another fetch
        : resp({ full: false, trees: [tree("zz", 5, 1050)], ids: ["a", "zz", "yy"], cursor: 1096 }),
    );
    const s2 = await pollForest(s1, get);
    expect(get.mock.calls.map((c) => c[0])).toEqual([
      "/api/stats/forest?range=24h&since=1090",
      "/api/stats/forest?range=24h",
    ]);
    expect(s2.cursor).toBe(1096);
    expect([...s2.trees.keys()]).toEqual(["zz"]);
  });
});
