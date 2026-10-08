/**
 * Key-holder login (Task 8): a key holder exchanges an inference API key for a forest-only token at /forest/login and
 * sees their own forest at /forest, with no admin session.
 *
 * What must hold:
 *   - only the token and its expiry are stored, never the key (no storage, cookie or URL)
 *   - 401 / 403 / 429 / other failures each read clearly
 *   - a forest auth error clears the token and routes to /forest/login exactly once
 *   - an expired token is treated as absent: straight to login, no failing request first
 *   - the full cycle auth error → login → back to /forest does not bounce again off SWR's cached error
 */
import { useSyncExternalStore } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// ---- next/navigation: a tiny reactive router ----------------------------------------------------------------
const nav = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  const state = { path: "/forest" };
  const go = (to: string) => {
    state.path = to.split("?")[0];
    listeners.forEach((l) => l());
  };
  return {
    state,
    listeners,
    replace: vi.fn((to: string) => go(to)),
    push: vi.fn((to: string) => go(to)),
  };
});
vi.mock("next/navigation", () => ({
  usePathname: () =>
    useSyncExternalStore(
      (l) => {
        nav.listeners.add(l);
        return () => nav.listeners.delete(l);
      },
      () => nav.state.path,
    ),
  useRouter: () => ({ replace: nav.replace, push: nav.push }),
}));

// ---- next/dynamic: load the view synchronously enough for a test --------------------------------------------
vi.mock("next/dynamic", async () => {
  const React = await import("react");
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    default: (loader: () => Promise<{ default: React.ComponentType<any> }>) => {
      const Lazy = React.lazy(loader);
      return (props: object) => React.createElement(React.Suspense, { fallback: null }, React.createElement(Lazy, props));
    },
  };
});

// ---- the three.js scene: a stub (as in forest-view.test.tsx) ------------------------------------------------
vi.mock("@/components/forest/engine/scene", () => ({
  createForestScene: vi.fn(() => ({
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
    camera: { focusSession: vi.fn(), userInteracted: vi.fn() },
    debug: { girthAt: vi.fn(), heightAt: vi.fn(), rel: () => 0, flights: () => [], cameraPos: () => ({ x: 0, y: 0, z: 0 }) },
  })),
}));
vi.mock("@/components/header-metrics", () => ({ HeaderMetrics: () => null }));

import LoginPage from "@/app/forest/login/page";
import ForestPage from "@/app/forest/page";
import { SessionGate } from "@/components/session-gate";
import { setAccessToken, __resetLoginRedirectInFlightForTests } from "@/lib/auth-fetch";

const KEY = "vw_secret_key_123";
const empty = { range: "24h", t0: 0, now: 100, models: {}, trees: [], flowers: [], ids: [], full: true, cursor: 0 };
const json = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
const urlOf = (input: RequestInfo | URL) => (typeof input === "string" ? input : input.toString());
const bearer = (init?: RequestInit) => new Headers(init?.headers).get("Authorization");

function storeToken(token: string, expMs = Date.now() + 3_600_000) {
  sessionStorage.setItem("vw-forest-token", token);
  sessionStorage.setItem("vw-forest-exp", String(expMs));
}

/** The app as the root layout composes it, routed by the mocked pathname, with one SWR cache for the whole run. */
function App() {
  const path = useSyncExternalStore(
    (l) => {
      nav.listeners.add(l);
      return () => nav.listeners.delete(l);
    },
    () => nav.state.path,
  );
  return <SessionGate>{path === "/forest/login" ? <LoginPage /> : <ForestPage />}</SessionGate>;
}
/** The SWR cache of the last render (one per render: entries never leak across tests). */
let swrCache = new Map();
const cache = () => (swrCache = new Map());
const forestEntries = () =>
  [...swrCache.entries()].filter(([, v]) => {
    const k = (v as { _k?: unknown })._k;
    return k === "forest-view:forest-token" || (Array.isArray(k) && k[1] === "forest-token");
  }).filter(([, v]) => (v as { data?: unknown }).data !== undefined || (v as { error?: unknown }).error !== undefined);
const showApp = () =>
  render(
    <SWRConfig value={{ provider: cache, dedupingInterval: 0, revalidateOnFocus: false }}>
      <App />
    </SWRConfig>,
  );
const showLogin = () =>
  render(
    <SWRConfig value={{ provider: cache, dedupingInterval: 0 }}>
      <LoginPage />
    </SWRConfig>,
  );

function submitKey(key = KEY) {
  fireEvent.change(screen.getByLabelText(/API key/i), { target: { value: key } });
  fireEvent.click(screen.getByRole("button", { name: /open forest/i }));
}

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}

describe("forest login", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    nav.state.path = "/forest/login";
    nav.replace.mockClear();
    nav.push.mockClear();
    setAccessToken(null);
    __resetLoginRedirectInFlightForTests();
    vi.stubGlobal("ResizeObserver", RO);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("stores the forest token, never the key", async () => {
    const fetchMock = vi.fn(async () => json({ token: "jwt", expires_in: 43200 }));
    vi.stubGlobal("fetch", fetchMock);
    const before = Date.now();
    showLogin();
    const input = screen.getByLabelText(/API key/i);
    expect(input).toHaveAttribute("type", "password");
    expect(input).toHaveAttribute("autocomplete", "off");
    submitKey();
    await waitFor(() => expect(sessionStorage.getItem("vw-forest-token")).toBe("jwt"));
    const exp = Number(sessionStorage.getItem("vw-forest-exp"));
    expect(exp).toBeGreaterThanOrEqual(before + 43200_000);
    expect(exp).toBeLessThanOrEqual(Date.now() + 43200_000);
    // the key goes only as a Bearer header to the same-origin login endpoint
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/forest/login");
    expect(init.method).toBe("POST");
    expect(bearer(init)).toBe(`Bearer ${KEY}`);
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest?key=1"));
    // never stored, never in the URL
    expect(JSON.stringify({ ...sessionStorage })).not.toContain(KEY);
    expect(JSON.stringify({ ...localStorage })).not.toContain(KEY);
    expect(document.cookie).not.toContain(KEY);
    for (const c of [...nav.replace.mock.calls, ...nav.push.mock.calls]) expect(String(c[0])).not.toContain(KEY);
    expect(window.location.href).not.toContain(KEY);
  });

  it.each([
    [401, /Key not accepted/],
    [403, /Key is paused/],
    [429, /Too many attempts — wait and try again/],
    [500, /Could not sign in/],
  ])("shows a clear error on %i", async (status, text) => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status })));
    showLogin();
    submitKey("vw_bad");
    expect(await screen.findByText(text)).toBeInTheDocument();
    expect(sessionStorage.getItem("vw-forest-token")).toBeNull();
    expect(nav.replace).not.toHaveBeenCalled();
    expect(screen.getByRole("alert").textContent).not.toContain("vw_bad");
  });

  it("shows a generic error when the network fails, without the key", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError(`fetch failed for ${KEY}`); }));
    const warn = vi.spyOn(console, "error").mockImplementation(() => {});
    showLogin();
    submitKey();
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/Could not sign in/);
    expect(alert.textContent).not.toContain(KEY);
    for (const c of warn.mock.calls) expect(JSON.stringify(c)).not.toContain(KEY);
    warn.mockRestore();
  });

  it("401 sends key holder to login once", async () => {
    nav.state.path = "/forest";
    storeToken("jwt");
    const fetchMock = vi.fn(async () => new Response("{}", { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    showApp();
    // both pollers (forest + in-flight) get their 401
    // (the first lazy import of the view compiles it: allow it time)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2), { timeout: 8000 });
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest/login"));
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
    expect(nav.replace).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem("vw-forest-token")).toBeNull();
    expect(sessionStorage.getItem("vw-forest-exp")).toBeNull();
    // no admin session check: a token holder never touches /api/auth/refresh
    for (const [u] of fetchMock.mock.calls as unknown as [string][]) expect(u).not.toContain("/api/auth");
  }, 15_000);

  it("an expired token goes straight to login, with no request", async () => {
    nav.state.path = "/forest";
    storeToken("jwt", Date.now() - 1000);
    const fetchMock = vi.fn(async () => json(empty));
    vi.stubGlobal("fetch", fetchMock);
    showApp();
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest/login"));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("vw-forest-token")).toBeNull();
    expect(sessionStorage.getItem("vw-forest-exp")).toBeNull();
  });

  it("with neither a token nor an admin session /forest goes to /forest/login, not /login", async () => {
    nav.state.path = "/forest";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      urlOf(input).includes("/api/auth/refresh") ? new Response("no", { status: 401 }) : json({}, 404),
    );
    vi.stubGlobal("fetch", fetchMock);
    const locReplace = vi.fn();
    const realLocation = window.location;
    Object.defineProperty(window, "location", {
      value: { ...realLocation, replace: locReplace, pathname: "/forest", search: "", href: "http://localhost/forest" },
      configurable: true,
      writable: true,
    });
    try {
      showApp();
      await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest/login"));
      expect(locReplace).not.toHaveBeenCalled();
      expect(screen.queryByTestId("forest-view")).toBeNull();
    } finally {
      Object.defineProperty(window, "location", { value: realLocation, configurable: true, writable: true });
    }
  });

  it("with an admin session /forest uses the session mode", async () => {
    nav.state.path = "/forest";
    setAccessToken("admin-access");
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      urlOf(input).startsWith("/api/stats/forest") ? json(empty) : json({ ts: "", count: 0, requests: [], by_token: [], by_ip: [] }),
    );
    vi.stubGlobal("fetch", fetchMock);
    showApp();
    expect(await screen.findByText(/No traffic yet/)).toBeInTheDocument();
    const forestCall = (fetchMock.mock.calls as unknown as [string, RequestInit][]).find(([u]) => u.startsWith("/api/stats/forest"));
    expect(bearer(forestCall![1])).toBe("Bearer admin-access");
    expect(nav.replace).not.toHaveBeenCalled();
  });

  it("auth error → login → back to /forest: no second redirect off the stale cached error", async () => {
    nav.state.path = "/forest";
    storeToken("old");
    let tokenIssued = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const u = urlOf(input);
      if (u === "/api/forest/login") {
        tokenIssued = true;
        return json({ token: "new", expires_in: 43200 });
      }
      if (bearer(init) !== "Bearer new") return new Response("{}", { status: 401 });
      if (u.startsWith("/api/stats/forest")) return json(empty);
      if (u.startsWith("/api/stats/requests")) return json({ ts: "", count: 0, requests: [], by_token: [], by_ip: [] });
      return json({}, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    showApp();

    // 1. the old token is refused: once to login
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest/login"));
    expect(await screen.findByLabelText(/API key/i)).toBeInTheDocument();

    // 2. sign in again
    submitKey();
    await waitFor(() => expect(tokenIssued).toBe(true));
    await waitFor(() => expect(nav.state.path).toBe("/forest"));

    // 3. the fresh view loads and stays: no immediate bounce from the cached ForestAuthError
    expect(await screen.findByText(/No traffic yet/)).toBeInTheDocument();
    await act(async () => { await new Promise((r) => setTimeout(r, 100)); });
    expect(nav.replace.mock.calls.filter(([p]) => p === "/forest/login")).toHaveLength(1);
    expect(nav.state.path).toBe("/forest");
    expect(sessionStorage.getItem("vw-forest-token")).toBe("new");
    expect(JSON.stringify({ ...sessionStorage })).not.toContain(KEY);
  });

  // ---- fix round 1 ----------------------------------------------------------------------------------------------
  function okFetch() {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const u = urlOf(input);
      if (u.startsWith("/api/stats/forest")) return json(empty);
      if (u.startsWith("/api/stats/requests")) return json({ ts: "", count: 0, requests: [], by_token: [], by_ip: [] });
      return json({}, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("token mode: Esc and leaving fullscreen keep the key holder signed in and on /forest", async () => {
    nav.state.path = "/forest";
    storeToken("jwt");
    okFetch();
    showApp();
    expect(await screen.findByText(/No traffic yet/, undefined, { timeout: 8000 })).toBeInTheDocument();
    // no ✕ in token mode: an explicit Sign out instead
    expect(screen.queryByRole("button", { name: /Close the full window/ })).toBeNull();
    expect(screen.getByRole("button", { name: /Sign out/ })).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
    // the browser leaving fullscreen
    const view = screen.getByTestId("forest-view");
    Object.defineProperty(document, "fullscreenElement", { value: view, configurable: true });
    fireEvent(document, new Event("fullscreenchange"));
    Object.defineProperty(document, "fullscreenElement", { value: null, configurable: true });
    fireEvent(document, new Event("fullscreenchange"));
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });

    expect(nav.replace).not.toHaveBeenCalled();
    expect(nav.push).not.toHaveBeenCalled();
    expect(nav.state.path).toBe("/forest");
    expect(sessionStorage.getItem("vw-forest-token")).toBe("jwt");
    expect(sessionStorage.getItem("vw-forest-exp")).not.toBeNull();
    expect(screen.getByTestId("forest-view")).toBeInTheDocument();
  }, 15_000);

  it("token mode: Sign out clears the token and the forest cache, and goes to login once", async () => {
    nav.state.path = "/forest";
    storeToken("jwt");
    okFetch();
    showApp();
    expect(await screen.findByText(/No traffic yet/, undefined, { timeout: 8000 })).toBeInTheDocument();
    await waitFor(() => expect(forestEntries().length).toBeGreaterThan(0));
    const out = screen.getByRole("button", { name: /Sign out/ });
    fireEvent.click(out);
    fireEvent.click(out);
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/forest/login"));
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    expect(nav.replace).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem("vw-forest-token")).toBeNull();
    expect(sessionStorage.getItem("vw-forest-exp")).toBeNull();
    expect(forestEntries()).toHaveLength(0);
  }, 15_000);

  it("session mode: Esc still goes to /stats", async () => {
    nav.state.path = "/forest";
    setAccessToken("admin-access");
    okFetch();
    showApp();
    expect(await screen.findByText(/No traffic yet/, undefined, { timeout: 8000 })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Sign out/ })).toBeNull();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(nav.push).toHaveBeenCalledWith("/stats");
  }, 15_000);

  it("shows a storage message (and keeps nothing) when the token cannot be stored", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ token: "jwt", expires_in: 43200 })));
    // jsdom's Storage cannot be spied on: a Map-backed one whose token write fails (quota)
    const m = new Map<string, string>();
    vi.stubGlobal("sessionStorage", {
      getItem: (k: string) => m.get(k) ?? null,
      setItem: (k: string, v: string) => {
        if (k === "vw-forest-token") throw new DOMException("quota", "QuotaExceededError");
        m.set(k, v);
      },
      removeItem: (k: string) => void m.delete(k),
      clear: () => m.clear(),
    });
    showLogin();
    submitKey();
    expect(await screen.findByText(/blocked tab storage/i)).toBeInTheDocument();
    expect([...m.keys()]).toEqual([]);
    expect(nav.replace).not.toHaveBeenCalled();
  });
});
