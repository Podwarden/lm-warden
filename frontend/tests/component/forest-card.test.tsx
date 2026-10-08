import { StrictMode } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

// The scene mock follows forest-view.test.tsx (Plan 2), plus Task 1's pause/resume/snapshot/exportView/importView.
type Scene = Record<string, ReturnType<typeof vi.fn> | unknown> & {
  pause: ReturnType<typeof vi.fn>;
  resume: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  snapshot: ReturnType<typeof vi.fn>;
  setCameraMode: ReturnType<typeof vi.fn>;
  resetView: ReturnType<typeof vi.fn>;
  exportView: ReturnType<typeof vi.fn>;
};
const scene = vi.hoisted(() => ({
  current: null as null | Scene,
  all: [] as Scene[],
  /** Overrides the next scenes' snapshot (a rejection, a bad URL). */
  snapshot: null as null | (() => Promise<string>),
}));
const router = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => router, usePathname: () => "/stats" }));
const VIEW = { pos: { x: 1, y: 2, z: 3 }, target: { x: 0, y: 1, z: 0 }, mode: "follow", frontRel: true };
vi.mock("@/components/forest/engine/scene", () => ({
  createForestScene: vi.fn(() => {
    const s = {
      setState: vi.fn(),
      shownAbs: vi.fn(() => 9100),
      jumpAbs: vi.fn(),
      pick: () => null,
      highlight: vi.fn(),
      resize: vi.fn(),
      dispose: vi.fn(),
      stats: () => ({ trees: 0, cubes: 0, flora: 0, frameMs: 0 }),
      setCameraMode: vi.fn(),
      resetView: vi.fn(),
      setPanels: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      snapshot: vi.fn(scene.snapshot ?? (async () => "data:image/png;base64,AAAA")),
      exportView: vi.fn(() => VIEW),
      importView: vi.fn(),
      camera: { focusSession: vi.fn(), userInteracted: vi.fn() },
      debug: { girthAt: vi.fn(), heightAt: vi.fn(), rel: () => 0, flights: () => [], cameraPos: () => ({ x: 0, y: 0, z: 0 }) },
    } as unknown as Scene;
    scene.current = s;
    scene.all.push(s);
    return s;
  }),
}));
import { createForestScene } from "@/components/forest/engine/scene";
import ForestCard from "@/components/forest/ForestCard";
import ForestPage from "@/app/forest/page";
import { encodeHandoff } from "@/components/forest/engine/handoff";

const empty = { range: "6h", t0: 0, now: 100, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 0 };

const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
/** Stubs fetch; the returned spy is called only for forest polls (never for the in-flight endpoint, which the card does not poll). */
function stubForestFetch(forest: unknown = empty) {
  const forestSpy = vi.fn();
  const all = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/auth/refresh") return json({ access_token: "t", expires_in: 900 });
    if (url.startsWith("/api/stats/forest")) {
      forestSpy(url);
      // the server answers with the range asked for
      const range = new URL(url, "http://x").searchParams.get("range");
      return json(forest && typeof forest === "object" && "range" in forest && range ? { ...forest, range } : forest);
    }
    if (url.startsWith("/api/stats/requests")) return json({ ts: "", count: 0, requests: [], by_token: [], by_ip: [] });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", all);
  return Object.assign(forestSpy, { all });
}

/** A controllable IntersectionObserver: `io.trigger(visible)` reports to every live observer. */
const io = {
  observers: [] as FakeIO[],
  trigger(visible: boolean) {
    act(() => {
      for (const o of io.observers) {
        if (!o.targets.length) continue;
        o.cb(o.targets.map((target) => ({ isIntersecting: visible, target }) as unknown as IntersectionObserverEntry), o as unknown as IntersectionObserver);
      }
    });
  },
};
class FakeIO {
  targets: Element[] = [];
  constructor(public cb: IntersectionObserverCallback, public opts?: IntersectionObserverInit) {
    io.observers.push(this);
  }
  observe(el: Element) {
    this.targets.push(el);
  }
  unobserve(el: Element) {
    this.targets = this.targets.filter((t) => t !== el);
  }
  disconnect() {
    this.targets = [];
  }
  takeRecords() {
    return [];
  }
}

function mockMatchMedia(query: string, matches: boolean) {
  vi.stubGlobal("matchMedia", vi.fn((q: string) => ({
    matches: q === query ? matches : false,
    media: q,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })));
}

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}

/** The card as the Stats page renders it, with the page's range; `rerender(range)` is the page's range switch. */
const show = (range: "1h" | "6h" | "24h" | "7d" = "6h") => {
  const cfg = { provider: () => new Map(), dedupingInterval: 0, revalidateOnFocus: false };
  const r = render(
    <SWRConfig value={cfg}>
      <ForestCard range={range} />
    </SWRConfig>,
  );
  return {
    ...r,
    switchTo: (next: "1h" | "6h" | "24h" | "7d") =>
      r.rerender(
        <SWRConfig value={cfg}>
          <ForestCard range={next} />
        </SWRConfig>,
      ),
  };
};

describe("ForestCard", () => {
  // warm the module next/dynamic loads: under a loaded machine the first (cold) import took longer than waitFor's 1 s
  // and the first tests that wait for the scene failed (a test-harness flake, not a product timing)
  beforeAll(async () => {
    await import("@/components/forest/ForestView");
  }, 30_000);
  beforeEach(() => {
    io.observers = [];
    scene.all = [];
    scene.current = null;
    scene.snapshot = null;
    router.push.mockReset();
    router.replace.mockReset();
    sessionStorage.clear();
    (createForestScene as ReturnType<typeof vi.fn>).mockClear();
    vi.stubGlobal("IntersectionObserver", FakeIO);
    vi.stubGlobal("ResizeObserver", RO);
    mockMatchMedia("(pointer: coarse)", false);
    Object.defineProperty(window, "innerWidth", { value: 1280, configurable: true, writable: true });
    Object.defineProperty(document, "hidden", { value: false, configurable: true });
  });
  afterEach(() => {
    cleanup();
    Object.defineProperty(document, "hidden", { value: false, configurable: true });
  });

  it("does not load the view before intersecting", async () => {
    stubForestFetch();
    show();
    expect(screen.getByTestId("forest-card-placeholder")).toBeInTheDocument();
    expect(createForestScene).not.toHaveBeenCalled();
    io.trigger(true);
    await waitFor(() =>
      expect(createForestScene).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ compact: true })),
    );
  });

  it("lets the Stats page scroll over it: the scene captures the wheel only after a click into the card (Plan 4 T4)", async () => {
    stubForestFetch();
    show();
    io.trigger(true);
    await waitFor(() =>
      expect(createForestScene).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ wheelNeedsFocus: true })),
    );
  });

  it("shows a focus ring while the card captures the trackpad, and hides it when released (Plan 4 T5)", async () => {
    stubForestFetch();
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const opts = (createForestScene as ReturnType<typeof vi.fn>).mock.calls[0][1] as { onWheelEngaged?: (e: boolean) => void };
    expect(typeof opts.onWheelEngaged).toBe("function");
    expect(screen.queryByTestId("forest-card-focus")).toBeNull();
    act(() => opts.onWheelEngaged!(true));
    const ring = screen.getByTestId("forest-card-focus");
    expect(ring).toHaveAttribute("aria-hidden", "true");
    expect(ring.className).toMatch(/pointer-events-none/);
    act(() => opts.onWheelEngaged!(false)); // Esc, a blur or a pointer-down elsewhere (the scene's WheelGate)
    expect(screen.queryByTestId("forest-card-focus")).toBeNull();
  });

  it("observes with threshold 0", () => {
    stubForestFetch();
    show();
    expect(io.observers.some((o) => o.opts?.threshold === 0)).toBe(true);
  });

  it("pauses offscreen and resumes on return", async () => {
    stubForestFetch();
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    expect(s.pause).not.toHaveBeenCalled();
    io.trigger(false);
    expect(s.pause).toHaveBeenCalled();
    expect(s.resume).not.toHaveBeenCalled();
    io.trigger(true);
    expect(s.resume).toHaveBeenCalled();
    // stays mounted while scrolled away: one scene, never disposed
    expect(createForestScene).toHaveBeenCalledTimes(1);
    expect(s.dispose).not.toHaveBeenCalled();
  });

  it("pauses while the document is hidden", async () => {
    stubForestFetch();
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    Object.defineProperty(document, "hidden", { value: true, configurable: true });
    act(() => void document.dispatchEvent(new Event("visibilitychange")));
    expect(s.pause).toHaveBeenCalled();
    Object.defineProperty(document, "hidden", { value: false, configurable: true });
    act(() => void document.dispatchEvent(new Event("visibilitychange")));
    expect(s.resume).toHaveBeenCalled();
  });

  it("does not poll while paused", async () => {
    const fetchSpy = stubForestFetch(empty);
    show();
    io.trigger(true);
    await screen.findByText(/No traffic in the last 6 h/);
    const n = fetchSpy.mock.calls.length;
    io.trigger(false);
    await new Promise((r) => setTimeout(r, 2500));
    expect(fetchSpy.mock.calls.length).toBe(n);
  }, 10_000);

  it("never polls the in-flight endpoint", async () => {
    const fetchSpy = stubForestFetch(empty);
    show();
    io.trigger(true);
    await screen.findByText(/No traffic in the last 6 h/);
    expect(fetchSpy.all.mock.calls.some(([u]) => String(u).startsWith("/api/stats/requests"))).toBe(false);
  });

  it("coarse pointer renders a still and releases the context", async () => {
    mockMatchMedia("(pointer: coarse)", true);
    const fetchSpy = stubForestFetch(empty);
    show();
    io.trigger(true);
    expect(await screen.findByRole("img", { name: /session forest/i })).toHaveAttribute(
      "src",
      expect.stringMatching(/^data:image\/png/),
    );
    expect(scene.current!.dispose).toHaveBeenCalled();
    // no polling after the still
    const n = fetchSpy.mock.calls.length;
    await new Promise((r) => setTimeout(r, 2500));
    expect(fetchSpy.mock.calls.length).toBe(n);
    // a tap opens the full window the same way, with no hand-off (the still has no live camera to continue)
    sessionStorage.setItem("forest.handoff", "left over");
    fireEvent.click(screen.getByRole("img", { name: /session forest/i }));
    expect(router.push).toHaveBeenCalledWith("/forest?range=6h"); // the range is always in the query (review M4)
    expect(sessionStorage.getItem("forest.handoff")).toBeNull();
  }, 10_000);

  it("a narrow window renders a still", async () => {
    window.innerWidth = 500;
    stubForestFetch(empty);
    show();
    io.trigger(true);
    expect(await screen.findByRole("img", { name: /session forest/i })).toHaveAttribute("src", expect.stringMatching(/^data:image\/png/));
    expect(scene.current!.dispose).toHaveBeenCalled();
  });

  it("shows the chip and requests range=6h", async () => {
    const fetchSpy = stubForestFetch(empty);
    show();
    io.trigger(true);
    expect(await screen.findByText("6 h · live · 30 s delay")).toBeInTheDocument();
    expect(fetchSpy.mock.calls.some(([u]) => String(u).includes("range=6h"))).toBe(true);
  });

  it("fetches the Stats page's range (1h, the page's default) and names it in the chip", async () => {
    const fetchSpy = stubForestFetch(empty);
    show("1h");
    io.trigger(true);
    expect(await screen.findByText("1 h · live · 30 s delay")).toBeInTheDocument();
    await waitFor(() => expect(fetchSpy.mock.calls.map(([u]) => u)).toContain("/api/stats/forest?range=1h"));
    expect(fetchSpy.mock.calls.every(([u]) => String(u).includes("range=1h"))).toBe(true);
  });

  it("follows the page's switch 6h → 24h: one full fetch of 24h then deltas, no 6h poll, the same scene, a new chip", async () => {
    const fetchSpy = stubForestFetch(empty);
    const { switchTo } = show("6h");
    io.trigger(true);
    await waitFor(() => expect(fetchSpy.mock.calls.map(([u]) => u)).toContain("/api/stats/forest?range=6h"));
    const before = fetchSpy.mock.calls.length;
    switchTo("24h");
    expect(await screen.findByText("24 h · live · 30 s delay")).toBeInTheDocument();
    const after = () => fetchSpy.mock.calls.slice(before).map(([u]) => String(u));
    await waitFor(() => expect(after().some((u) => u.startsWith("/api/stats/forest?range=24h&since="))).toBe(true), { timeout: 5000 });
    expect(after().filter((u) => u === "/api/stats/forest?range=24h")).toHaveLength(1);
    expect(after().some((u) => u.includes("range=6h"))).toBe(false);
    expect(createForestScene).toHaveBeenCalledTimes(1);
    const fed = scene.current!.setState as ReturnType<typeof vi.fn>;
    expect(fed.mock.calls.at(-1)![0]).toMatchObject({ range: "24h" });
  }, 10_000);

  it("keeps its own SWR entry per range, never the full window's (review M6)", async () => {
    stubForestFetch(empty);
    const cache = new Map();
    render(
      <SWRConfig value={{ provider: () => cache, dedupingInterval: 0, revalidateOnFocus: false }}>
        <ForestCard range="24h" />
      </SWRConfig>,
    );
    io.trigger(true);
    await waitFor(() => expect([...cache.keys()]).toContain("forest-view:session:card:24h"));
    expect([...cache.keys()]).not.toContain("forest-view:session");
  });

  it("hands its range to the full window with the camera view", async () => {
    stubForestFetch(empty);
    show("24h");
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(JSON.parse(sessionStorage.getItem("forest.handoff")!)).toMatchObject({ pos: VIEW.pos, range: "24h" });
  });

  it("has a Reset view button (follow-up B) that resets the scene's camera", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    fireEvent.click(await screen.findByRole("button", { name: "Reset view" }));
    expect(scene.current!.resetView).toHaveBeenCalledTimes(1);
  });

  it("starts in follow mode, with no toggle, panels, description or legend", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await screen.findByText(/No traffic in the last 6 h/);
    expect(scene.current!.setCameraMode).toHaveBeenCalledWith("follow");
    expect(screen.queryByText("Session forest")).toBeNull();
    expect(screen.queryByText("Legend")).toBeNull();
    expect(screen.queryByText(/In flight/)).toBeNull();
    expect(screen.queryByRole("button", { name: /wide|cinematic|follow/i })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fullscreen" })).toBeNull();
  });

  it("handoff disposes card scene before navigating", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(s.dispose).toHaveBeenCalled();
    expect(router.push).toHaveBeenCalledWith("/forest?range=6h");
    expect(s.dispose.mock.invocationCallOrder[0]).toBeLessThan(router.push.mock.invocationCallOrder[0]);
    expect(s.exportView.mock.invocationCallOrder[0]).toBeLessThan(s.dispose.mock.invocationCallOrder[0]);
    expect(sessionStorage.getItem("forest.handoff")).toContain('"mode":"follow"');
    expect(JSON.parse(sessionStorage.getItem("forest.handoff")!)).toMatchObject({ pos: VIEW.pos, target: VIEW.target });
  });

  it("?debug=1 on /stats carries over to /forest (the e2e debug hooks)", async () => {
    window.history.replaceState(null, "", "/stats?debug=1");
    try {
      stubForestFetch(empty);
      show();
      io.trigger(true);
      await waitFor(() => expect(createForestScene).toHaveBeenCalled());
      fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
      expect(router.push).toHaveBeenCalledWith("/forest?range=6h&debug=1");
    } finally {
      window.history.replaceState(null, "", "/");
    }
  });

  it("a fast double click hands off once: one push, and click 1's view stays stored", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    const btn = screen.getByRole("button", { name: "Open full window" });
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(router.push).toHaveBeenCalledTimes(1);
    expect(s.dispose).toHaveBeenCalledTimes(1);
    expect(JSON.parse(sessionStorage.getItem("forest.handoff")!)).toMatchObject({ pos: VIEW.pos, target: VIEW.target, mode: "follow" });
  });

  it("shows an opening state after the click, in place of the blank canvas", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(screen.getByText("Opening the full window…")).toBeInTheDocument();
  });

  it("a storage failure still disposes and navigates", async () => {
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(s.dispose).toHaveBeenCalled();
    expect(router.push).toHaveBeenCalledWith("/forest?range=6h");
  });

  // ---- Task 2 review fold-ins (M1, M2, M4, M5) ---------------------------------------------------------------
  it("a rejected snapshot disposes the scene, shows the fallback and stops polling", async () => {
    mockMatchMedia("(pointer: coarse)", true);
    scene.snapshot = async () => {
      throw new Error("no frame");
    };
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const fetchSpy = stubForestFetch(empty);
    show();
    io.trigger(true);
    expect(await screen.findByText("3D view unavailable.")).toBeInTheDocument();
    expect(scene.current!.dispose).toHaveBeenCalled();
    expect(screen.queryByRole("img", { name: /session forest/i })).toBeNull();
    expect(screen.getByRole("button", { name: "Open full window" })).toBeInTheDocument();
    expect(warn).toHaveBeenCalled();
    const n = fetchSpy.mock.calls.length;
    await new Promise((r) => setTimeout(r, 2500));
    expect(fetchSpy.mock.calls.length).toBe(n);
  }, 10_000);

  it("a still that fails to load falls back to the placeholder with the full-window button", async () => {
    mockMatchMedia("(pointer: coarse)", true);
    stubForestFetch(empty);
    show();
    io.trigger(true);
    const img = await screen.findByRole("img", { name: /session forest/i });
    fireEvent.error(img);
    expect(screen.queryByRole("img", { name: /session forest/i })).toBeNull();
    expect(screen.getByText("3D view unavailable.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(router.push).toHaveBeenCalledWith("/forest?range=6h");
  });

  it("a phone whose first fetch fails releases the scene, stops polling and shows the placeholder (review M1)", async () => {
    mockMatchMedia("(pointer: coarse)", true);
    const forestCalls = vi.fn();
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/auth/refresh") return json({ access_token: "t", expires_in: 900 });
      if (url.startsWith("/api/stats/forest")) return forestCalls(url), json({ detail: "boom" }, 500);
      return json({}, 404);
    }));
    show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const s = scene.current!;
    await waitFor(() => expect(s.dispose).toHaveBeenCalled());
    expect(screen.getByText("3D view unavailable.")).toBeInTheDocument();
    expect(document.querySelector("[data-testid=forest-canvas-host] canvas")).toBeNull();
    const n = forestCalls.mock.calls.length;
    // past POLL_MS and SWR's first error retry (errorRetryInterval 5 s, jittered ±50 %)
    await new Promise((r) => setTimeout(r, 8000));
    expect(forestCalls.mock.calls.length).toBe(n); // no polling (nor error retries) after the release
    fireEvent.click(screen.getByRole("button", { name: "Open full window" }));
    expect(router.push).toHaveBeenCalledWith("/forest?range=6h");
  }, 15_000);

  it("removes its observer and visibility listener on unmount", async () => {
    stubForestFetch(empty);
    const removed = vi.spyOn(document, "removeEventListener");
    const view = show();
    io.trigger(true);
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    const obs = io.observers.filter((o) => o.opts?.threshold === 0);
    expect(obs.length).toBeGreaterThan(0);
    const disconnects = obs.map((o) => vi.spyOn(o, "disconnect"));
    const onVis = removed.mock.calls.length;
    view.unmount();
    for (const d of disconnects) expect(d).toHaveBeenCalled();
    expect(removed.mock.calls.slice(onVis).some(([type]) => type === "visibilitychange")).toBe(true);
    expect(scene.current!.dispose).toHaveBeenCalled();
  });

  it("runs no 1 Hz token-bar tick, and stops the hover re-pick while paused", async () => {
    const set = vi.spyOn(window, "setInterval");
    const clear = vi.spyOn(window, "clearInterval");
    stubForestFetch(empty);
    show();
    io.trigger(true);
    await screen.findByText(/No traffic in the last 6 h/);
    // the card has no token bars: no 1 s interval at all
    expect(set.mock.calls.some(([, ms]) => ms === 1000)).toBe(false);
    const repicks = () => set.mock.results.filter((_, i) => set.mock.calls[i][1] === 200).map((r) => r.value);
    expect(repicks()).toHaveLength(1);
    const [id] = repicks();
    io.trigger(false);
    expect(clear.mock.calls.some(([x]) => x === id)).toBe(true);
    io.trigger(true);
    expect(repicks()).toHaveLength(2); // restarted on resume
  });

  it("removes only its own debug hooks", async () => {
    window.history.replaceState(null, "", "/stats?debug=1");
    try {
      stubForestFetch(empty);
      const view = show();
      io.trigger(true);
      await waitFor(() => expect(window.__forest).toBeDefined());
      // another view (the full window) has since installed its own hooks
      const other = { ...window.__forest! };
      window.__forest = other;
      view.unmount();
      expect(window.__forest).toBe(other);
    } finally {
      delete window.__forest;
      window.history.replaceState(null, "", "/");
    }
  });

  it("is full width and 320 px high", () => {
    stubForestFetch();
    show();
    const root = screen.getByTestId("forest-card");
    expect(root.className).toMatch(/w-full/);
    expect(root.className).toMatch(/h-\[320px\]/);
  });
});

// ---- /forest: the receiving side of the hand-off -------------------------------------------------------------
describe("/forest hand-off", () => {
  const full = { ...empty, range: "24h" };
  beforeEach(() => {
    scene.all = [];
    scene.current = null;
    scene.snapshot = null;
    sessionStorage.clear();
    (createForestScene as ReturnType<typeof vi.fn>).mockClear();
    vi.stubGlobal("ResizeObserver", RO);
    window.history.replaceState(null, "", "/forest");
  });
  afterEach(() => cleanup());
  const page = () =>
    render(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, revalidateOnFocus: false }}>
        <ForestPage />
      </SWRConfig>,
    );

  it("a fresh hand-off is imported into the scene and deleted", async () => {
    stubForestFetch(full);
    sessionStorage.setItem("forest.handoff", encodeHandoff(VIEW as never, Date.now() - 2000));
    page();
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    await waitFor(() => expect(scene.current!.importView).toHaveBeenCalledWith(VIEW));
    expect(scene.current!.importView).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem("forest.handoff")).toBeNull();
  });

  it("a fresh hand-off survives StrictMode's double effect run", async () => {
    stubForestFetch(full);
    sessionStorage.setItem("forest.handoff", encodeHandoff(VIEW as never, Date.now()));
    render(
      <StrictMode>
        <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, revalidateOnFocus: false }}>
          <ForestPage />
        </SWRConfig>
      </StrictMode>,
    );
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    await waitFor(() => expect(scene.current!.importView).toHaveBeenCalledWith(VIEW));
    expect(sessionStorage.getItem("forest.handoff")).toBeNull();
  });

  it("a stale hand-off is not imported, and is deleted", async () => {
    stubForestFetch(full);
    sessionStorage.setItem("forest.handoff", encodeHandoff(VIEW as never, Date.now() - 11_000));
    page();
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    await screen.findByTestId("forest-view");
    expect(scene.current!.importView).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("forest.handoff")).toBeNull();
  });

  const polled = (spy: ReturnType<typeof vi.fn>) => spy.mock.calls.map(([u]) => String(u));
  const rangeGroup = () => screen.findByRole("group", { name: "Range" });

  it("opens with the hand-off's range (the card's)", async () => {
    const spy = stubForestFetch(full);
    sessionStorage.setItem("forest.handoff", encodeHandoff({ ...VIEW, range: "6h" } as never, Date.now()));
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=6h"));
    expect(polled(spy).every((u) => u.includes("range=6h"))).toBe(true);
    expect(within(await rangeGroup()).getByRole("button", { name: "6h" })).toHaveAttribute("aria-pressed", "true");
  });

  it("defaults to 24h; the controls bar's switch refetches once, reframes, and is remembered (forest.range)", async () => {
    localStorage.removeItem("forest.range");
    const spy = stubForestFetch(full);
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=24h"));
    const group = await rangeGroup();
    expect(within(group).getAllByRole("button").map((b) => b.textContent)).toEqual(["1h", "6h", "24h", "7d"]);
    expect(within(group).getByRole("button", { name: "24h" })).toHaveAttribute("aria-pressed", "true");
    const before = spy.mock.calls.length;
    fireEvent.click(within(group).getByRole("button", { name: "7d" }));
    await waitFor(() => expect(polled(spy).slice(before)).toContain("/api/stats/forest?range=7d"));
    await waitFor(() => expect(polled(spy).slice(before).some((u) => u.startsWith("/api/stats/forest?range=7d&since="))).toBe(true), { timeout: 5000 });
    expect(polled(spy).slice(before).filter((u) => u === "/api/stats/forest?range=7d")).toHaveLength(1);
    expect(within(group).getByRole("button", { name: "7d" })).toHaveAttribute("aria-pressed", "true");
    expect(localStorage.getItem("forest.range")).toBe("7d");
    expect(createForestScene).toHaveBeenCalledTimes(1);
  }, 10_000);

  it("?range= and then forest.range choose the range; the hand-off wins over both", async () => {
    localStorage.setItem("forest.range", "7d");
    let spy = stubForestFetch(full);
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=7d"));
    cleanup();
    window.history.replaceState(null, "", "/forest?range=1h");
    spy = stubForestFetch(full);
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=1h"));
    cleanup();
    sessionStorage.setItem("forest.handoff", encodeHandoff({ ...VIEW, range: "6h" } as never, Date.now()));
    spy = stubForestFetch(full);
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=6h"));
    localStorage.removeItem("forest.range");
  });

  it("a switch updates a ?range= in the address, so a reload keeps it", async () => {
    window.history.replaceState(null, "", "/forest?range=1h&debug=1");
    const spy = stubForestFetch(full);
    page();
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=1h"));
    fireEvent.click(within(await rangeGroup()).getByRole("button", { name: "7d" }));
    expect(window.location.search).toBe("?range=7d&debug=1");
    localStorage.removeItem("forest.range");
  });

  it("an unreadable localStorage falls back to 24h", async () => {
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation((k: string) => {
      if (k === "forest.range") throw new Error("denied");
      return null;
    });
    try {
      const spy = stubForestFetch(full);
      page();
      await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=24h"));
    } finally {
      get.mockRestore();
    }
  });

  it("the key holder's window has the same switch", async () => {
    const spy = stubForestFetch(full);
    sessionStorage.setItem("vw-forest-token", "ft");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() + 3_600_000));
    page();
    const group = await rangeGroup();
    fireEvent.click(within(group).getByRole("button", { name: "1h" }));
    await waitFor(() => expect(polled(spy)).toContain("/api/stats/forest?range=1h"));
  });

  it("forest-token mode ignores the hand-off and deletes it", async () => {
    stubForestFetch(full);
    sessionStorage.setItem("vw-forest-token", "ft");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() + 3_600_000));
    sessionStorage.setItem("forest.handoff", encodeHandoff(VIEW as never, Date.now()));
    page();
    await waitFor(() => expect(createForestScene).toHaveBeenCalled());
    expect(scene.current!.importView).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("forest.handoff")).toBeNull();
  });
});
