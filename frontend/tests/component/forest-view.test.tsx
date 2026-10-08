import { StrictMode } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { SWRConfig } from "swr";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Made = { canvas: HTMLCanvasElement; opts: { onContextLost?: () => void; onCameraMode?: (m: string) => void }; s: Record<string, unknown> };
const scene = vi.hoisted(() => ({
  current: null as null | Record<string, unknown>,
  make: null as null | ((canvas: HTMLCanvasElement, opts: never) => Record<string, unknown>),
  all: [] as Made[],
  /** Canvases whose scene was disposed: like a real WebGL context after forceContextLoss, they cannot be reused. */
  dead: new Set<HTMLCanvasElement>(),
}));
vi.mock("@/components/forest/engine/scene", () => ({
  createForestScene: vi.fn((canvas: HTMLCanvasElement, opts: never) => scene.make!(canvas, opts)),
}));
const baseMake = (canvas: HTMLCanvasElement, opts: never) => {
    if (scene.dead.has(canvas)) throw new TypeError("context lost: getShaderPrecisionFormat returned null");
    const s = {
      setState: vi.fn(),
      shownAbs: vi.fn(() => 9100),
      jumpAbs: vi.fn(),
      pick: () => null,
      highlight: vi.fn(),
      resize: vi.fn(),
      dispose: vi.fn(() => void scene.dead.add(canvas)),
      stats: () => ({ trees: 0, cubes: 0, flora: 0, frameMs: 0 }),
      setCameraMode: vi.fn(),
      resetView: vi.fn(),
      setPanels: vi.fn(),
      snapshot: vi.fn(async () => "data:image/png;base64,AAAA"),
      // like the controller: a row click switches to cinematic and the scene reports it
      camera: { focusSession: vi.fn((id: string | null) => id !== null && (opts as Made["opts"]).onCameraMode?.("cinema")), userInteracted: vi.fn() },
      debug: {
        girthAt: vi.fn(() => 0.5),
        heightAt: vi.fn(() => 1),
        rel: () => 0,
        flights: () => [{ treeId: "a", dx: -4 }],
        cameraPos: () => ({ x: 1, y: 2, z: 3 }),
      },
    };
    scene.current = s;
    scene.all.push({ canvas, opts, s });
    return s;
};
scene.make = baseMake;
// the tooltip text is built once per new pick result: counted to prove an unchanged pick re-renders nothing (N4)
const describeCalls = vi.hoisted(() => ({ n: 0 }));
vi.mock("@/components/forest/panels/Tooltip", async (orig) => {
  const m = await orig<typeof import("@/components/forest/panels/Tooltip")>();
  return { ...m, describePick: (...a: Parameters<typeof m.describePick>) => (describeCalls.n++, m.describePick(...a)) };
});
import ForestView from "@/components/forest/ForestView";
import type { LiveRequestRow } from "@/lib/live-stats";

const empty = { range: "24h", t0: 0, now: 100, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 0 };
const turn = (t: number) => [t, 1000, 200, 400, [], [], "stop", 1];
const withTree = {
  ...empty,
  now: 10_000,
  cursor: 9990,
  ids: ["a"],
  trees: [{
    id: "a", key: "k", start: 9000, end: 9050, last: 9050, last_at: 9050,
    traits: { sys0: 1, ctx_gen: 1, mean_gen: 1, thrash: 0, fail: 0, fanout: 0 },
    sessions: [{ id: "sess-1", variant: null, children: [], turns: [turn(9000), turn(9030)] }],
  }],
};
const row: LiveRequestRow = {
  id: "r1", token_id: null, token_name: "dev", client_ip: null, model: "m", path: "/v1/chat/completions",
  prompt_tokens: 10, completion_tokens: 0, context_tokens: 10, max_model_len: 100, context_pct: 0.1,
  elapsed_s: 1, phase: "prefill", orphan: false, session_id: "raw-client-session", forest_session: "sess-1",
  cache_est_tokens: null, cache_est_pct: null,
};

const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
function stubFetch(forest: unknown = empty, requests: LiveRequestRow[] = []) {
  const urls: string[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    urls.push(url);
    if (url === "/api/auth/refresh") return json({ access_token: "t", expires_in: 900 });
    if (url.startsWith("/api/stats/forest")) return json(forest);
    if (url.startsWith("/api/stats/requests")) return json({ ts: "", count: requests.length, requests, by_token: [], by_ip: [] });
    return json({}, 404);
  }));
  return urls;
}

type Props = Partial<React.ComponentProps<typeof ForestView>>;
const show = (p: Props = {}) =>
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, revalidateOnFocus: false }}>
      <ForestView mode="session" compact={false} {...p} />
    </SWRConfig>,
  );

class RO {
  static all: RO[] = [];
  constructor(public cb: ResizeObserverCallback) { RO.all.push(this); }
  observe() {}
  unobserve() {}
  disconnect() {}
}

describe("ForestView", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    RO.all = [];
    scene.all = [];
    scene.dead.clear();
    vi.stubGlobal("ResizeObserver", RO);
    Object.defineProperty(window, "innerWidth", { value: 1024, configurable: true, writable: true });
  });
  afterEach(() => {
    cleanup();
    delete (window as unknown as { __forest?: unknown }).__forest;
    window.history.replaceState(null, "", "/");
  });

  it("renders empty state", async () => {
    stubFetch();
    show();
    expect(await screen.findByText(/No traffic yet/)).toBeInTheDocument();
  });

  it("collapses description and remembers it", async () => {
    window.innerWidth = 1400;
    stubFetch();
    show();
    const summary = await screen.findByText("Session forest");
    expect(summary.closest("details")).toHaveAttribute("open");
    fireEvent.click(summary);
    expect(localStorage.getItem("forest.desc.open")).toBe("false");
    expect(summary.closest("details")).not.toHaveAttribute("open");
  });

  it("description and legend start closed below 1100 px", async () => {
    stubFetch();
    show();
    expect((await screen.findByText("Session forest")).closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText("Legend").closest("details")).not.toHaveAttribute("open");
  });

  it("the full window captures every gesture, and the legend has the trackpad help (Plan 4 T4)", async () => {
    stubFetch();
    show();
    expect(
      await screen.findByText("Trackpad: two fingers pan · pinch zoom · Shift+two fingers (or twist in Safari) rotate"),
    ).toBeInTheDocument();
    expect((scene.all[0].opts as { wheelNeedsFocus?: boolean }).wheelNeedsFocus).toBe(false);
  });

  it("no webgl shows fallback", async () => {
    const { createForestScene } = await import("@/components/forest/engine/scene");
    (createForestScene as ReturnType<typeof vi.fn>).mockImplementationOnce(() => { throw new Error("no webgl"); });
    stubFetch();
    show();
    expect(await screen.findByText(/3D view unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/In flight/)).toBeInTheDocument();
  });

  it("a scene that throws on new data falls back to the still message; panels keep working", async () => {
    const { createForestScene } = await import("@/components/forest/engine/scene");
    (createForestScene as ReturnType<typeof vi.fn>).mockImplementationOnce((canvas: HTMLCanvasElement, opts: never) => {
      const s = scene.make!(canvas, opts);
      s.setState = vi.fn(() => { throw new RangeError("Array buffer allocation failed"); });
      return s;
    });
    stubFetch(withTree, [row]);
    show();
    expect(await screen.findByText(/3D view unavailable/)).toBeInTheDocument();
    expect(scene.current!.dispose).toHaveBeenCalled();
    expect(await screen.findByTestId("live-row")).toBeInTheDocument();
  });

  it("feeds the scene, polls the forest with a since cursor and live in-flight rows", async () => {
    const urls = stubFetch(withTree, [row]);
    show();
    await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
    expect(urls).toContain("/api/stats/forest?range=24h");
    await waitFor(() => expect(urls.some((u) => u.startsWith("/api/stats/requests"))).toBe(true));
    expect(screen.queryByText(/No traffic yet/)).toBeNull();
  });

  it("in-flight row hover highlights its forest session (not the raw client id), click focuses the camera", async () => {
    stubFetch(withTree, [row]);
    show();
    const tr = await screen.findByTestId("live-row");
    fireEvent.mouseEnter(tr);
    expect(scene.current!.highlight).toHaveBeenLastCalledWith("sess-1");
    fireEvent.mouseLeave(tr);
    expect(scene.current!.highlight).toHaveBeenLastCalledWith(null);
    fireEvent.click(tr);
    const cam = scene.current!.camera as { focusSession: ReturnType<typeof vi.fn> };
    expect(cam.focusSession).toHaveBeenCalledWith("sess-1");
  });

  it("token bars follow the scene's shown time, not wall-clock now", async () => {
    stubFetch(withTree);
    const { container } = show();
    await waitFor(() => expect(scene.current!.shownAbs).toHaveBeenCalled());
    // shownAbs 9100 → minutes 121..150 hold the turns at 9000 and 9030 (minute 150)
    await waitFor(() => expect(container.querySelector("[data-testid=token-bars]")?.getAttribute("data-total")).toBe("2400"));
  });

  it("a row without a forest session neither highlights nor focuses", async () => {
    stubFetch(withTree, [{ ...row, forest_session: null }]);
    show();
    const tr = await screen.findByTestId("live-row");
    fireEvent.mouseEnter(tr);
    expect(scene.current!.highlight).toHaveBeenLastCalledWith(null);
    fireEvent.click(tr);
    expect((scene.current!.camera as { focusSession: ReturnType<typeof vi.fn> }).focusSession).not.toHaveBeenCalled();
  });

  it("the toggle shows the mode a row click switched the camera to (I3)", async () => {
    stubFetch(withTree, [row]);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /Wide/ }));
    expect(screen.getByRole("button", { name: /Wide/ })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(await screen.findByTestId("live-row"));
    await waitFor(() => expect(screen.getByRole("button", { name: /Cinematic/ })).toHaveAttribute("aria-pressed", "true"));
    expect(screen.getByRole("button", { name: /Wide/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("Reset view (follow-up B) asks the scene to reset, in either mode; the mode stays", async () => {
    stubFetch();
    show();
    fireEvent.click(await screen.findByRole("button", { name: "Reset view" }));
    expect(scene.current!.resetView).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Wide/ }));
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));
    expect(scene.current!.resetView).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: /Wide/ })).toHaveAttribute("aria-pressed", "true");
  });

  it("camera toggle switches cinematic and wide", async () => {
    stubFetch();
    show();
    fireEvent.click(await screen.findByRole("button", { name: /Wide/ }));
    expect(scene.current!.setCameraMode).toHaveBeenLastCalledWith("follow");
    fireEvent.click(screen.getByRole("button", { name: /Cinematic/ }));
    expect(scene.current!.setCameraMode).toHaveBeenLastCalledWith("cinema");
  });

  it("reports open panel rects to the scene and re-reports when a panel resizes", async () => {
    window.innerWidth = 1400;
    stubFetch();
    show();
    await waitFor(() => expect(scene.current!.setPanels).toHaveBeenCalled());
    const calls = (scene.current!.setPanels as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(RO.all.length).toBeGreaterThan(0);
    act(() => RO.all.forEach((o) => o.cb([], o as unknown as ResizeObserver)));
    expect((scene.current!.setPanels as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(calls);
    const rects = (scene.current!.setPanels as ReturnType<typeof vi.fn>).mock.lastCall![0];
    expect(Array.isArray(rects)).toBe(true);
    expect(rects[0]).toEqual(expect.objectContaining({ left: expect.any(Number), right: expect.any(Number) }));
  });

  it("Esc and ✕ close the view", async () => {
    stubFetch();
    const onClose = vi.fn();
    show({ onClose });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Close/ }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("a forest-token auth failure reaches onAuthError and shows a message", async () => {
    stubFetch();
    sessionStorage.clear();
    const onAuthError = vi.fn();
    show({ mode: "forest-token", onAuthError });
    await waitFor(() => expect(onAuthError).toHaveBeenCalled());
    expect(await screen.findByText(/sign in again/i)).toBeInTheDocument();
  });

  it("?debug=1 exposes window.__forest hooks", async () => {
    window.history.replaceState(null, "", "/?debug=1");
    stubFetch();
    show();
    await waitFor(() => expect((window as unknown as { __forest?: unknown }).__forest).toBeDefined());
    const f = (window as unknown as { __forest: {
      stats: () => unknown; cameraPos: () => unknown; girthAt: (id: string) => number | null;
      flights: () => { treeId: string; fromX: number; toX: number }[];
    } }).__forest;
    expect(f.cameraPos()).toEqual({ x: 1, y: 2, z: 3 });
    expect(f.girthAt("a")).toBe(0.5);
    const [fl] = f.flights();
    expect(fl.treeId).toBe("a");
    expect(fl.toX - fl.fromX).toBe(4); // dx = from − to = −4: moved right
    await waitFor(() => expect(scene.current).not.toBeNull());
    expect(await (f as unknown as { snapshot: () => Promise<string | null> }).snapshot()).toBe("data:image/png;base64,AAAA");
  });

  it("no debug hooks without ?debug=1", async () => {
    stubFetch();
    show();
    await screen.findByText("Session forest");
    expect((window as unknown as { __forest?: unknown }).__forest).toBeUndefined();
  });

  it("StrictMode: each mount gets its own fresh canvas; the first scene is disposed; no fallback", async () => {
    stubFetch(withTree);
    render(
      <StrictMode>
        <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0, revalidateOnFocus: false }}>
          <ForestView mode="session" compact={false} />
        </SWRConfig>
      </StrictMode>,
    );
    await waitFor(() => expect(scene.all.length).toBe(2));
    const [a, b] = scene.all;
    expect(a.canvas).not.toBe(b.canvas);
    expect(a.s.dispose).toHaveBeenCalled();
    expect(b.s.dispose).not.toHaveBeenCalled();
    expect(a.canvas.isConnected).toBe(false);
    expect(b.canvas.isConnected).toBe(true);
    await waitFor(() => expect(b.s.setState).toHaveBeenCalled());
    expect(screen.queryByText(/3D view unavailable/)).toBeNull();
  });

  it("a lost context drops the scene (disposed, its canvas removed) and shows the fallback", async () => {
    stubFetch(withTree, [row]);
    show();
    await waitFor(() => expect(scene.all.length).toBe(1));
    const [{ s, opts, canvas }] = scene.all;
    act(() => opts.onContextLost!());
    expect(await screen.findByText(/3D view unavailable/)).toBeInTheDocument();
    expect(s.dispose).toHaveBeenCalledTimes(1);
    expect(canvas.isConnected).toBe(false);
    // the panels keep working, and no longer reach the dead scene
    fireEvent.mouseEnter(await screen.findByTestId("live-row"));
    expect(s.highlight).not.toHaveBeenCalled();
  });

  it("the first ForestAuthError stops both pollers and reaches onAuthError exactly once", async () => {
    sessionStorage.setItem("vw-forest-token", "expired");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() + 60_000)); // held as valid: the server refuses it
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      urls.push(String(input));
      return new Response("{}", { status: 401 });
    }));
    const onAuthError = vi.fn();
    show({ mode: "forest-token", onAuthError });
    await waitFor(() => expect(onAuthError).toHaveBeenCalled());
    await screen.findByText(/sign in again/i);
    const seen = urls.length;
    await new Promise((r) => setTimeout(r, 2600)); // longer than one 2 s poll
    expect(urls.length).toBe(seen);
    expect(onAuthError).toHaveBeenCalledTimes(1);
  }, 10_000);

  it("the in-flight SWR entry is keyed by mode, never the Stats page's plain key", async () => {
    stubFetch(withTree, [row]);
    const cache = new Map();
    render(
      <SWRConfig value={{ provider: () => cache, dedupingInterval: 0, revalidateOnFocus: false }}>
        <ForestView mode="session" compact={false} />
      </SWRConfig>,
    );
    await screen.findByTestId("live-row");
    const keys = [...cache.keys()].map(String);
    expect(keys).not.toContain("/api/stats/requests");
    expect(keys.some((k) => k.includes("/api/stats/requests") && k.includes("session"))).toBe(true);
  });

  it("unmount disposes the scene, clears intervals and observers, removes hooks, and stops fetching", async () => {
    window.history.replaceState(null, "", "/?debug=1");
    const urls = stubFetch(withTree, [row]);
    const clear = vi.spyOn(window, "clearInterval");
    const disconnect = vi.spyOn(RO.prototype, "disconnect");
    const { unmount } = show();
    await screen.findByTestId("live-row");
    await waitFor(() => expect(window.__forest).toBeDefined());
    const [{ s }] = scene.all;
    unmount();
    expect(s.dispose).toHaveBeenCalledTimes(1);
    expect(clear).toHaveBeenCalled();
    expect(disconnect).toHaveBeenCalled();
    expect(window.__forest).toBeUndefined();
    const seen = urls.length;
    await new Promise((r) => setTimeout(r, 2600));
    expect(urls.length).toBe(seen);
  }, 10_000);

  it("the description names the server", async () => {
    window.innerWidth = 1400;
    stubFetch();
    show();
    expect(await screen.findByText(/LM Warden · Last 24 h/)).toBeInTheDocument();
  });

  describe("canvas hover detail (I4)", () => {
    const toolTree = {
      ...withTree,
      trees: [{
        ...withTree.trees[0],
        key: "farm-key",
        sessions: [{ id: "0123456789abcdef", variant: null, children: [], turns: [
          turn(9000),
          [9030, 52_000, 340, 48_000, [["read", 120, false], ["bash", 40, true]], ["edit"], "tool_calls", 1],
        ] }],
      }],
    };
    const pickWith = (result: unknown) => {
      const pick = vi.fn(() => result);
      scene.make = ((orig) => (canvas: HTMLCanvasElement, opts: never) => ({ ...orig(canvas, opts), pick }))(baseMake);
      return pick;
    };
    afterEach(() => void (scene.make = baseMake));
    // jsdom has no PointerEvent: a MouseEvent carries clientX/Y and buttons the same way
    beforeEach(() => {
      if (!("PointerEvent" in window)) vi.stubGlobal("PointerEvent", class extends MouseEvent {});
    });
    const hover = async (x = 300, y = 200) => {
      const canvas = await screen.findByLabelText("Session forest");
      fireEvent.pointerMove(canvas, { clientX: x, clientY: y });
      return canvas;
    };
    const clock = (abs: number) => new Date(abs * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    it("a turn shows its key, short session id, local time, tokens and tool families; leaving hides it", async () => {
      const pick = pickWith({ kind: "turn", part: "wood", treeId: "a", sessionId: "0123456789abcdef", turn: 1, time: 9030 });
      stubFetch(toolTree);
      show();
      await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
      const canvas = await hover();
      const tip = await screen.findByRole("tooltip");
      expect(pick).toHaveBeenCalledWith(300, 200);
      expect(tip).toHaveTextContent("farm-key");
      expect(tip).toHaveTextContent("01234567");
      expect(tip).not.toHaveTextContent("0123456789abcdef");
      expect(tip).toHaveTextContent(clock(9030));
      expect(tip).toHaveTextContent(/context\s*52,000/);
      expect(tip).toHaveTextContent(/generated\s*340/);
      expect(tip).toHaveTextContent(/cached\s*48,000/);
      expect(tip).toHaveTextContent(/tools\s*bash, edit, read/);
      fireEvent.pointerLeave(canvas);
      expect(screen.queryByRole("tooltip")).toBeNull();
    });

    it("a quiet-hours post shows how long the quiet stretch was ('6 h quiet') and its ends", async () => {
      pickWith({ kind: "quiet", time: 3600, to: 3600 + 6 * 3600 + 200 });
      stubFetch(withTree);
      show();
      await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
      await hover();
      const tip = await screen.findByRole("tooltip");
      expect(tip).toHaveTextContent("6 h quiet");
      expect(tip).toHaveTextContent(clock(3600));
    });

    it("the legend names the quiet-hours post", async () => {
      stubFetch(withTree);
      show();
      expect(await screen.findByText(/quiet hours, shortened/)).toBeInTheDocument();
    });

    it("the legend describes the voxel look: blocks, the cyan tint, pink and olive blocks, stones; no bark or tool leaves (Plan 4 T5)", async () => {
      stubFetch(withTree);
      show();
      const legend = await screen.findByTestId("forest-legend");
      for (const t of [/Brown blocks: trunk and limbs/, /Green leaf blocks: the crown/, /Cyan tint: blocks growing now/, /Pink block: a turn that answered/, /Olive block: a failed tool/, /Ghost blocks/, /Daisy bed/, /mushrooms/, /stone blocks on the shore/])
        expect(within(legend).getByText(t)).toBeInTheDocument();
      // the old look's rows: per-tool leaf colours, cache-coloured bark, the blue pulse, compaction rings, aerial roots
      for (const t of [/Bark/, /Blue pulse/, /Read \/ search/, /Edit \/ write/, /Shell \/ test/, /withered/, /Pale ring/, /aerial roots/])
        expect(within(legend).queryByText(t)).toBeNull();
    });

    it("the legend's ghost swatch is drawn in the ghost's own colours: the pale crown, soil and stone blocks (final review)", async () => {
      const { GHOST_CROWN, GHOST_SOIL, GHOST_STONE } = await import("@/components/forest/engine/ghostColors");
      stubFetch(withTree);
      show();
      const legend = await screen.findByTestId("forest-legend");
      const row = within(legend).getByText(/Ghost blocks/).closest("li")!;
      const sw = [...row.querySelectorAll<HTMLElement>("span[aria-hidden] > span")].map((e) => e.style.background);
      const rgb = (h: string) => { const n = parseInt(h.slice(1), 16); return `rgb(${n >> 16}, ${(n >> 8) & 255}, ${n & 255})`; };
      expect(sw).toEqual([GHOST_CROWN, GHOST_SOIL[0], GHOST_STONE[0]].map(rgb));
      expect(within(legend).queryByText(/fallen leaf/)).toBeNull();
    });

    it("a flower and a mushroom show their time and tokens; nothing picked hides the tooltip", async () => {
      let result: unknown = { kind: "flower", time: 9050, ctx: 900, gen: 350 };
      pickWith(undefined);
      scene.make = ((orig) => (canvas: HTMLCanvasElement, opts: never) => ({ ...orig(canvas, opts), pick: () => result }))(baseMake);
      stubFetch(withTree);
      show();
      await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
      await hover();
      let tip = await screen.findByRole("tooltip");
      expect(tip).toHaveTextContent(/Flower/);
      expect(tip).toHaveTextContent(clock(9050));
      expect(tip).toHaveTextContent(/prompt\s*900/);
      expect(tip).toHaveTextContent(/answer\s*350/);
      result = { kind: "mushroom", time: 9060, ctx: 2400 };
      await new Promise((r) => setTimeout(r, 60)); // past the 20 Hz throttle
      await hover(310, 200);
      tip = await screen.findByRole("tooltip");
      await waitFor(() => expect(tip).toHaveTextContent(/Mushroom/));
      expect(tip).toHaveTextContent(/prompt\s*2,400/);
      result = null;
      await new Promise((r) => setTimeout(r, 60));
      await hover(320, 200);
      await waitFor(() => expect(screen.queryByRole("tooltip")).toBeNull());
    });

    it("picks at most 20 times a second, and the last position always wins", async () => {
      const pick = pickWith(null);
      stubFetch(withTree);
      show();
      const canvas = await screen.findByLabelText("Session forest");
      for (let i = 0; i < 10; i++) fireEvent.pointerMove(canvas, { clientX: 100 + i, clientY: 50 });
      expect(pick).toHaveBeenCalledTimes(1);
      await waitFor(() => expect(pick).toHaveBeenLastCalledWith(109, 50));
      expect(pick).toHaveBeenCalledTimes(2);
    });

    it("re-picks about 5 times a second while the pointer rests, so the tooltip follows a moving camera (N4)", async () => {
      let result: unknown = { kind: "flower", time: 9050, ctx: 900, gen: 350 };
      const pick = vi.fn((_x: number, _y: number) => result);
      scene.make = ((orig) => (canvas: HTMLCanvasElement, opts: never) => ({ ...orig(canvas, opts), pick }))(baseMake);
      stubFetch(withTree);
      show();
      await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
      const canvas = await hover();
      expect(await screen.findByRole("tooltip")).toHaveTextContent(/Flower/);
      result = { kind: "mushroom", time: 9060, ctx: 2400 }; // the camera moved: something else is under the pointer
      await waitFor(() => expect(screen.getByRole("tooltip")).toHaveTextContent(/Mushroom/), { timeout: 600 });
      expect(pick.mock.calls.every((c) => c[0] === 300 && c[1] === 200)).toBe(true); // no pointermove since
      fireEvent.pointerLeave(canvas);
      const n = pick.mock.calls.length;
      await new Promise((r) => setTimeout(r, 450));
      expect(pick.mock.calls.length).toBe(n); // stops when the pointer leaves
    });

    it("an unchanged pick result re-renders nothing (N4)", async () => {
      const pick = vi.fn(() => ({ kind: "flower", time: 9050, ctx: 900, gen: 350 }));
      scene.make = ((orig) => (canvas: HTMLCanvasElement, opts: never) => ({ ...orig(canvas, opts), pick }))(baseMake);
      stubFetch(withTree);
      show();
      await waitFor(() => expect(scene.current!.setState).toHaveBeenCalled());
      await hover();
      await screen.findByRole("tooltip");
      const before = describeCalls.n, picks = pick.mock.calls.length;
      await new Promise((r) => setTimeout(r, 650)); // ≥ 3 re-picks, same result
      expect(pick.mock.calls.length).toBeGreaterThanOrEqual(picks + 2);
      expect(describeCalls.n).toBe(before);
    });

    it("stays inside the viewport and off the side panels", async () => {
      const { tooltipPosition } = await import("@/components/forest/panels/Tooltip");
      const vw = { width: 1000, height: 700 };
      // near the bottom-right corner: flipped inside the viewport
      const a = tooltipPosition({ x: 990, y: 690 }, { w: 200, h: 100 }, vw, []);
      expect(a.left + 200).toBeLessThanOrEqual(1000);
      expect(a.top + 100).toBeLessThanOrEqual(700);
      expect(a.left).toBeGreaterThanOrEqual(0);
      // a panel to the right of the cursor: placed on the other side, not under it
      const panel = { left: 620, top: 0, right: 1000, bottom: 500 };
      const b = tooltipPosition({ x: 560, y: 100 }, { w: 200, h: 100 }, vw, [panel]);
      const overlaps = b.left < panel.right && b.left + 200 > panel.left && b.top < panel.bottom && b.top + 100 > panel.top;
      expect(overlaps).toBe(false);
    });
  });
});

describe("ForestView poll cadence (SWR defaults, Plan 3 Task 1 follow-up)", () => {
  beforeEach(() => {
    scene.all = [];
    scene.dead.clear();
    vi.stubGlobal("ResizeObserver", RO);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("polls the forest every POLL_MS (2 s): 5 ± 1 fetches in 10 s with SWR's default dedupingInterval", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const urls = stubFetch(withTree);
    // the app's SWR defaults: only a fresh cache, no dedupingInterval override (the other tests set it to 0)
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <ForestView mode="session" compact={false} />
      </SWRConfig>,
    );
    const at: number[] = [];
    let seen = 0;
    for (let t = 0; t <= 10_000; t += 50) {
      await vi.advanceTimersByTimeAsync(50);
      const n = urls.filter((u) => u.startsWith("/api/stats/forest")).length;
      for (; seen < n; seen++) at.push(t);
    }
    const gaps = at.slice(1).map((t, i) => t - at[i]);
    // 5 ± 1 in 10 s, and no gap twice POLL_MS: SWR's interval timer starts at mount, so with the default 2 s dedupe the
    // first refresh fell inside the mount fetch's dedupe window and was dropped (a 4 s first gap)
    expect(at.length, `fetch times ${at.join(", ")}`).toBeGreaterThanOrEqual(4);
    expect(at.length).toBeLessThanOrEqual(6);
    expect(Math.max(...gaps), `gaps ${gaps.join(", ")}`).toBeLessThan(3000);
  });
});
