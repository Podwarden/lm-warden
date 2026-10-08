"use client";

// The forest full window (spec §6.4, live view only per §10.6): the three.js scene behind glass panels.
// - Left: description and legend (collapsible, remembered). Right: token bars and requests in flight.
//   Bottom centre: the camera toggle and "Reset view". Bottom right: fullscreen and close.
// - Data: SWR polls `/api/stats/forest` every 2 s with a since cursor (`pollForest`); the scene runs its own clock,
//   30 s behind the server's now. Token bars follow the scene's shown time, never wall-clock now.
// - Range (follow-up C): `range` is fetched and framed. Each range is its own SWR key; a switch is one full fetch of the
//   new range (a state of another range is never polled with its cursor), and only the latest poll of the current range
//   reaches the scene. The full window shows a 1h / 6h / 24h / 7d switch beside the camera toggle (`onRange`); the card
//   follows the Stats page's range.
// - Open panels report their screen rects to the scene so the follow fit leaves them out.
// - In-flight rows match the forest by `forest_session` (the hashed key the forest's sessions use). Hover marks the
//   session's joints; a click focuses the camera on it, in cinematic mode: the toggle follows the scene's mode.
// - Hover detail: the canvas is picked at most 20 times a second (`scene.pick`) and shown in a tooltip (panels/Tooltip).
// - No WebGL (or a lost context): "3D view unavailable" in place of the canvas; the panels keep working.
// - `variant="card"` (the Stats page card, spec §6.3): the same poller, scene wiring and auth handling, rendered small.
//   Compact engine, wide follow camera with no toggle, no side panels, description or legend (so no in-flight poll);
//   a status chip, "Reset view" and an "Open full window" button. `paused` (offscreen or hidden tab) pauses the scene and stops
//   polling. `still` (phones): after the first state, one snapshot replaces the canvas, the scene is disposed and
//   polling stops. "Open full window" hands the camera view over and releases this scene first (`onOpenFull`).
// - `initialView` (the full window): a camera view handed over by the card, imported as soon as the scene exists.
// This module imports three.js (through the scene): load it only with next/dynamic, ssr: false.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { type ForestScene, type ForestView as CameraView, type PickResult, createForestScene } from "@/components/forest/engine/scene";
import type { CameraMode, PanelRect } from "@/components/forest/engine/camera";
import { FOREST_RANGE, ForestAuthError, forestFetcher, pollForest } from "@/lib/forest/client";
import { type ForestRange, type ForestState, type Range, isForestRange } from "@/lib/forest/types";
import type { LiveRequestRow } from "@/lib/live-stats";
import { cn } from "@/lib/utils";
import { Controls, RangeSwitch } from "./panels/Controls";
import { Description } from "./panels/Description";
import { GLASS } from "./panels/glass";
import { type ForestMode, InFlight } from "./panels/InFlight";
import { Legend } from "./panels/Legend";
import { TokenBars } from "./panels/TokenBars";
import { ForestTooltip, type Rect, describePick, tooltipPosition } from "./panels/Tooltip";

/** The view runs this far behind the server's now (the camera's look-ahead). */
const DELAY_S = 30;
const POLL_MS = 2000;
/** SWR dedupe window for the forest poll: well under POLL_MS (see the useSWR options). */
const POLL_DEDUPE_MS = 500;
const BARS_MS = 1000;
/** Canvas hover picks at most this often (ms): 20 Hz. */
const PICK_MS = 50;
/** While the pointer rests on the canvas, it is re-picked this often (ms, ~5 Hz): the camera keeps moving. */
const REPICK_MS = 200;

/** One pick's identity: an unchanged result is not a new tooltip (no re-render). */
const pickKey = (r: PickResult): string =>
  r.kind === "turn" ? `t:${r.treeId}:${r.sessionId}:${r.turn}` : r.kind === "ghost" ? `g:${r.treeId}:${r.plantedAt}` : r.kind === "quiet" ? `q:${r.time}:${r.to}` : `${r.kind}:${r.time}:${r.ctx}`;

export interface ForestViewProps {
  mode: ForestMode;
  compact?: boolean;
  /** Esc, ✕ or leaving browser fullscreen. The page routes back to /stats. Without it, Esc only leaves fullscreen. */
  onClose?: () => void;
  /** Given (the key holder's forest-token mode): an explicit "Sign out" button replaces ✕. */
  onSignOut?: () => void;
  /** The forest token was missing, expired or revoked (401/403). Task 8 wires the redirect. */
  onAuthError?: (err: ForestAuthError) => void;
  /** Ask for browser fullscreen on open (tolerating a refusal: the fixed overlay stays). */
  fullscreen?: boolean;
  /** "full" (default): the full window. "card": the Stats page card (always compact, follow camera, no panels). */
  variant?: "full" | "card";
  /** The fetch range, fetched and framed (default: the full window's 24 h). */
  range?: Range;
  /** The full window: the range switch was pressed (the page keeps and remembers the range). Without it, no switch. */
  onRange?: (range: ForestRange) => void;
  /** Offscreen or a hidden tab: the scene is paused and nothing is polled. */
  paused?: boolean;
  /** Phones: show one still frame after the first state, then dispose the scene and stop polling. */
  still?: boolean;
  /**
   * The card's "Open full window" (or a tap on the still): the camera view to hand over (null on a still or without a
   * scene) and `release`, which disposes this view's scene. Call `release` before navigating: the full window's scene
   * must never share the page with this one (two live WebGL contexts).
   */
  onOpenFull?: (view: CameraView | null, release: () => void) => void;
  /** The full window: a view handed over by the card, imported into the scene as soon as it exists. */
  initialView?: CameraView | null;
}

interface DebugHooks {
  stats: () => ReturnType<ForestScene["stats"]>;
  cameraPos: () => { x: number; y: number; z: number };
  /** The orbit target as shown; null without a scene. */
  cameraTarget: () => { x: number; y: number; z: number } | null;
  /** The camera is held by the user (a drag, a gesture, or their 9 s grace). */
  userHold: () => boolean;
  /** The card was clicked into: its wheel drives the camera (before that the page scrolls over it). */
  wheelEngaged: () => boolean;
  /** The ground's height at (x, z): the camera stays ≥ this + 0.5 (follow-up A). */
  groundAt: (x: number, z: number) => number;
  cameraLook: () => { x: number; y: number; z: number } | null;
  girthAt: (treeId: string) => number | null;
  heightAt: (treeId: string) => number | null;
  /** Fine voxel cells of a tree's build on screen, and the level of detail it is drawn at. */
  cellsAt: (treeId: string) => number | null;
  levelOf: (treeId: string) => number | null;
  levelCubes: () => number[];
  holdPose: (pos: { x: number; y: number; z: number }, target: { x: number; y: number; z: number }) => void;
  /** In-progress transplant flights. x is relative to the destination (toX = 0): toX − fromX > 0 is a move right. */
  flights: () => { treeId: string; fromX: number; toX: number }[];
  shownAbs: () => number;
  treeIds: () => string[];
  /** The scene's snapshot (a PNG data URL; it works while paused); null without a scene. */
  snapshot: () => Promise<string | null>;
  /** The camera, front-relative (what the hand-off carries); null before the first data. */
  exportView: () => CameraView | null;
  /** Where the follow fit wants the camera (world); null without a fit or outside follow mode. */
  followGoal: () => { x: number; y: number; z: number } | null;
  /** Every built tree's height on screen (px), and the spit's length over the shown range (units). */
  treePx: () => number[];
  spanLength: () => number;
  pickWorld: (x: number, y: number, z: number) => PickResult | null;
  treeDist: () => { id: string; d: number; top: number; scale: number; x: number; z: number; r: number }[];
}

declare global {
  interface Window {
    __forest?: DebugHooks;
  }
}

const debugEnabled = () => {
  try {
    return new URLSearchParams(window.location.search).get("debug") === "1";
  } catch {
    return false;
  }
};

export default function ForestView({
  mode,
  compact = false,
  onClose,
  onSignOut,
  onAuthError,
  fullscreen = false,
  variant = "full",
  range = FOREST_RANGE,
  onRange,
  paused = false,
  still = false,
  onOpenFull,
  initialView = null,
}: ForestViewProps) {
  const isCard = variant === "card";
  const compactMode = compact || isCard;
  const rootRef = useRef<HTMLDivElement>(null);
  /** Holds the canvas: the scene effect creates a fresh <canvas> per mount (StrictMode mounts twice, and a disposed
   *  scene's canvas has had its context force-lost). */
  const hostRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ForestScene | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [camMode, setCamMode] = useState<CameraMode>(isCard ? "follow" : "cinema");
  /** The card was clicked into and captures the trackpad (the scene's WheelGate): a subtle focus ring shows it. */
  const [engaged, setEngaged] = useState(false);
  /** The still frame (`still`), once taken. */
  const [stillUrl, setStillUrl] = useState<string | null>(null);
  /** "Open full window" was clicked: the scene is released and the card waits for /forest (one-shot). */
  const [opening, setOpening] = useState(false);
  const openedRef = useRef(false);
  const stillTaken = useRef(false);
  /** Hover detail: what is under the pointer and where the pointer was when it was picked. */
  const [tip, setTip] = useState<{ x: number; y: number; result: PickResult } | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  /** Places the tooltip for a pointer at (x, y) by writing its style: inside the viewport, off the open panels. */
  const placeTip = useCallback((x: number, y: number) => {
    const el = tipRef.current, root = rootRef.current;
    if (!el || !root) return;
    const panels: Rect[] = [...root.querySelectorAll<HTMLElement>("[data-forest-panel]")].map((p) => p.getBoundingClientRect());
    const at = tooltipPosition({ x, y }, { w: el.offsetWidth, h: el.offsetHeight }, { width: window.innerWidth, height: window.innerHeight }, panels);
    el.style.left = `${at.left}px`;
    el.style.top = `${at.top}px`;
  }, []);
  const [authError, setAuthError] = useState<ForestAuthError | null>(null);
  /** Set by the first ForestAuthError: both pollers stop, and onAuthError fires exactly once. */
  const authLatch = useRef(false);

  // ---- data: one SWR key per range, the cursor lives in a ref (no key churn while polling) --------------------
  /** The latest poll of the current range (a poll of another range that lands after a switch is dropped). */
  const stateRef = useRef<ForestState | null>(null);
  const rangeRef = useRef(range);
  rangeRef.current = range;
  const get = useMemo(() => forestFetcher(mode), [mode]);
  // the still is taken (or cannot be): nothing more to poll
  const stopped = still && (stillUrl !== null || unavailable);
  /** Read by SWR before every revalidation: true stops the fetch even before the new interval applies. */
  const haltRef = useRef(false);
  /** The scene was released for the hand-off to the full window: this view is about to unmount, poll nothing more. */
  const released = useRef(false);
  haltRef.current = paused || stopped || released.current;
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const { data: state, error } = useSWR<ForestState>(
    // the full window's key is unchanged (token.ts clears it by name); another range is its own cache entry
    // the card has its own entries (review M6): it never shares one with the full window's same range
    isCard ? `forest-view:${mode}:card:${range}` : range === FOREST_RANGE ? `forest-view:${mode}` : `forest-view:${mode}:${range}`,
    async () => {
      // a held state of another range is not polled with its cursor: one full fetch of this range (pollForest)
      const s = await pollForest(stateRef.current, get, range);
      if (rangeRef.current === range) stateRef.current = s;
      return s;
    },
    {
      refreshInterval: authError || paused || stopped ? 0 : POLL_MS,
      // below POLL_MS: SWR's interval timer starts at mount, so with the default 2 s window the first refresh fell
      // inside the mount fetch's dedupe window and was dropped (a 4 s first gap). It still joins a fetch in flight.
      dedupingInterval: POLL_DEDUPE_MS,
      isPaused: () => authLatch.current || haltRef.current,
      shouldRetryOnError: (e: unknown) => !(e instanceof ForestAuthError),
      revalidateOnFocus: false,
      keepPreviousData: true,
    },
  );

  // latched: the first auth failure (from either poller) stops both and is reported once
  const onAuthErrorRef = useRef(onAuthError);
  onAuthErrorRef.current = onAuthError;
  const authFailed = useCallback((e: ForestAuthError) => {
    if (authLatch.current) return;
    authLatch.current = true;
    setAuthError(e);
    onAuthErrorRef.current?.(e);
  }, []);
  useEffect(() => {
    if (error instanceof ForestAuthError) authFailed(error);
  }, [error, authFailed]);

  // ---- panels → scene: the follow fit leaves the open panels out ---------------------------------------------
  const reportPanels = useCallback(() => {
    const root = rootRef.current, scene = sceneRef.current;
    if (!root || !scene) return;
    const rects: PanelRect[] = [...root.querySelectorAll<HTMLElement>("[data-forest-panel]")].map((el) => {
      const r = el.getBoundingClientRect();
      return { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
    });
    scene.setPanels(rects);
  }, []);

  // ---- the scene ------------------------------------------------------------------------------------------
  /** The paused state last applied to the scene: `pause()` / `resume()` are called on changes only. */
  const appliedPause = useRef(false);
  /** The scene failed (a lost context, or it threw): drop it and show the still message; the panels keep working. */
  const dropScene = useCallback(() => {
    const scene = sceneRef.current;
    sceneRef.current = null;
    setUnavailable(true);
    try {
      scene?.dispose(); // stops its render loop and controls
    } catch {
      /* already broken */
    }
    hostRef.current?.replaceChildren();
  }, []);
  /** The hand-off: dispose the scene (and its WebGL context) now, before the full window mounts its own. */
  const release = useCallback(() => {
    released.current = haltRef.current = true;
    const scene = sceneRef.current;
    sceneRef.current = null;
    try {
      scene?.dispose();
    } catch {
      /* already broken */
    }
    hostRef.current?.replaceChildren();
  }, []);
  /** The hover re-pick timer's controls (the scene effect owns it): stopped while paused. */
  const repickRef = useRef<{ start: () => void; stop: () => void } | null>(null);
  const initialViewRef = useRef(initialView);
  const feed = useCallback(
    (s: ForestState) => {
      try {
        sceneRef.current?.setState(s);
      } catch (e) {
        console.warn("forest: the 3D scene failed; showing the panels only", e);
        dropScene();
      }
    },
    [dropScene],
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    // a fresh canvas per mount: each scene owns its canvas and context (dispose force-loses the context)
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", "Session forest");
    canvas.className = "absolute inset-0 block h-full w-full";
    host.appendChild(canvas);
    let scene: ForestScene | null = null;
    try {
      scene = createForestScene(canvas, {
        compact: compactMode,
        wheelNeedsFocus: isCard,
        onContextLost: () => {
          if (scene && sceneRef.current === scene) {
            console.warn("forest: the WebGL context was lost; showing the panels only");
            dropScene();
          }
        },
        onCameraMode: (m) => setCamMode(m),
        onWheelEngaged: isCard ? setEngaged : undefined,
      });
    } catch (e) {
      console.warn("forest: WebGL unavailable; showing the panels only", e);
      canvas.remove();
      setUnavailable(true);
      return;
    }
    const live: ForestScene = scene;
    sceneRef.current = live;
    stillTaken.current = false;
    if (isCard) live.setCameraMode("follow"); // the card's wide camera: the whole recent forest in the thumbnail
    appliedPause.current = pausedRef.current;
    if (pausedRef.current) live.pause();
    // a view handed over by the card: before the first data it replaces the front placement (used once)
    if (initialViewRef.current) live.importView(initialViewRef.current);
    if (stateRef.current) feed(stateRef.current);
    reportPanels();
    const onResize = () => {
      sceneRef.current?.resize();
      reportPanels();
    };
    window.addEventListener("resize", onResize);
    // hover detail: one pick per PICK_MS at most, the latest pointer position always picked last; none while dragging.
    // While the pointer rests on the canvas it is re-picked every REPICK_MS (the camera moves under it). An unchanged
    // result only moves the tooltip (no state update, no re-render); a new one is one state update.
    let lastPick = -Infinity, pending: { x: number; y: number } | null = null, timer: ReturnType<typeof setTimeout> | undefined;
    let pointer: { x: number; y: number } | null = null, shownKey: string | null = null;
    const show = (key: string | null, next: { x: number; y: number; result: PickResult } | null) => {
      if (key === shownKey) return void (next && placeTip(next.x, next.y));
      shownKey = key;
      setTip(next);
    };
    const pickAt = (x: number, y: number) => {
      lastPick = performance.now();
      const result = sceneRef.current === live ? live.pick(x, y) : null;
      show(result ? pickKey(result) : null, result ? { x, y, result } : null);
    };
    let repick: ReturnType<typeof setInterval> | undefined;
    const startRepick = () => {
      repick ??= setInterval(() => {
        if (pointer && performance.now() - lastPick >= PICK_MS) pickAt(pointer.x, pointer.y);
      }, REPICK_MS);
    };
    const stopRepick = () => {
      if (repick !== undefined) clearInterval(repick);
      repick = undefined;
    };
    const repickCtl = { start: startRepick, stop: stopRepick };
    repickRef.current = repickCtl;
    if (!pausedRef.current) startRepick();
    const onPointerMove = (e: PointerEvent) => {
      if (e.buttons) {
        pending = pointer = null;
        return show(null, null);
      }
      pointer = { x: e.clientX, y: e.clientY };
      const wait = lastPick + PICK_MS - performance.now();
      if (wait <= 0) return pickAt(e.clientX, e.clientY);
      pending = { x: e.clientX, y: e.clientY };
      timer ??= setTimeout(() => {
        timer = undefined;
        const p = pending;
        pending = null;
        if (p) pickAt(p.x, p.y);
      }, wait);
    };
    const onPointerLeave = () => {
      pending = pointer = null;
      show(null, null);
    };
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    return () => {
      window.removeEventListener("resize", onResize);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      if (timer !== undefined) clearTimeout(timer);
      stopRepick();
      if (repickRef.current === repickCtl) repickRef.current = null;
      if (sceneRef.current === live) {
        sceneRef.current = null;
        live.dispose();
      }
      canvas.remove();
      setEngaged(false);
    };
    // the scene lives as long as the view; `compact` is fixed for a mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // only the latest poll of the current range: a switch's cached entry of that range (an earlier visit) is stale, and a
  // poll of the previous range that lands after the switch is not this range's
  useEffect(() => {
    const s = stateRef.current;
    if (state && s && s.range === range) feed(s);
  }, [state, feed, range]);

  // offscreen / hidden tab: the scene stops rendering and its clock (and the pollers stop, above)
  useEffect(() => {
    if (paused) repickRef.current?.stop();
    else repickRef.current?.start();
    const s = sceneRef.current;
    if (!s || appliedPause.current === paused) return;
    appliedPause.current = paused;
    if (paused) s.pause();
    else s.resume();
  }, [paused]);

  // phones: a failed first fetch (no state, so no still to take) releases the scene too; the placeholder and its
  // "Open full window" button show, and `stopped` (unavailable) ends polling and the error retries
  useEffect(() => {
    if (still && error && state === undefined && sceneRef.current) dropScene();
  }, [still, error, state, dropScene]);

  // phones: once the first state is in, one frame becomes an <img> and the scene (its WebGL context) is released
  useEffect(() => {
    if (!still || !state || stillTaken.current) return;
    const live = sceneRef.current;
    if (!live) return;
    stillTaken.current = true;
    live.snapshot().then(
      (url) => {
        if (sceneRef.current !== live) {
          stillTaken.current = false; // that scene is gone (a remount): the next state retries with the new one
          return;
        }
        sceneRef.current = null;
        live.dispose();
        hostRef.current?.replaceChildren();
        setStillUrl(url);
      },
      (e: unknown) => {
        if (sceneRef.current !== live) return;
        console.warn("forest: the still frame failed", e);
        dropScene();
      },
    );
  }, [still, state, dropScene]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => reportPanels());
    root.querySelectorAll("[data-forest-panel]").forEach((el) => ro.observe(el));
    return () => ro.disconnect();
  }, [reportPanels]);

  // ---- token bars follow the shown forest time (polled ~1 Hz) ---------------------------------------------
  const [atRel, setAtRel] = useState<number | null>(null);
  useEffect(() => {
    if (isCard) return; // the card has no token bars
    const tick = () => {
      const s = stateRef.current;
      if (!s) return;
      const shown = sceneRef.current?.shownAbs();
      // without a scene (no WebGL) the panels still follow the server's now − 30 s (state.now is relative to t0),
      // never the browser clock
      setAtRel(shown !== undefined && Number.isFinite(shown) ? shown - s.t0 : s.now - DELAY_S);
    };
    tick();
    const id = setInterval(tick, BARS_MS);
    return () => clearInterval(id);
  }, [state, isCard]);

  // ---- in-flight rows ↔ scene -------------------------------------------------------------------------------
  // matched on forest_session: the forest's session ids are the hashed keys, never the raw client id
  const onRowHover = useCallback((row: LiveRequestRow | null) => sceneRef.current?.highlight(row?.forest_session ?? null), []);
  const onRowClick = useCallback((row: LiveRequestRow) => {
    if (row.forest_session) sceneRef.current?.camera.focusSession(row.forest_session);
  }, []);

  // ---- hover detail: placed after measuring, inside the viewport and off the open panels ---------------------
  const tipContent = useMemo(() => (tip ? describePick(tip.result, stateRef.current) : null), [tip]);
  // placed by writing the element's style after it is measured: no second render
  useLayoutEffect(() => {
    if (tip) placeTip(tip.x, tip.y);
  }, [tip, placeTip]);

  const changeMode = useCallback((m: CameraMode) => {
    setCamMode(m);
    sceneRef.current?.setCameraMode(m);
  }, []);
  // ends any user override (drag or zoom hold, a focused row) and glides back to the mode's default view
  const resetView = useCallback(() => sceneRef.current?.resetView(), []);

  // ---- leaving: Esc, ✕, or the browser leaving fullscreen -----------------------------------------------------
  useEffect(() => {
    if (isCard) return; // the card is part of the Stats page: Esc is not its own
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (onClose) onClose();
      else if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, isCard]);

  const wasFullscreen = useRef(false);
  const enterFullscreen = useCallback(() => {
    const el = rootRef.current;
    if (!el || typeof el.requestFullscreen !== "function" || !document.fullscreenEnabled) return;
    // a refusal (no user gesture, a policy) leaves the fixed overlay, which already fills the window
    el.requestFullscreen().catch(() => {});
  }, []);
  useEffect(() => {
    if (isCard) return;
    const onChange = () => {
      if (document.fullscreenElement) wasFullscreen.current = true;
      else if (wasFullscreen.current) {
        // Esc in browser fullscreen never reaches the page as a keydown: treat leaving fullscreen as Esc
        wasFullscreen.current = false;
        onClose?.();
      }
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [onClose, isCard]);
  useEffect(() => {
    if (isCard) return;
    // only with a live user gesture: a request without one is refused anyway
    const active = (navigator as Navigator & { userActivation?: { isActive: boolean } }).userActivation?.isActive;
    if (fullscreen && active) enterFullscreen();
    return () => {
      if (document.fullscreenElement && document.exitFullscreen) {
        wasFullscreen.current = false;
        document.exitFullscreen().catch(() => {});
      }
    };
  }, [fullscreen, enterFullscreen, isCard]);

  // ---- ?debug=1 hooks for the e2e motion test --------------------------------------------------------------
  useEffect(() => {
    if (!debugEnabled()) return;
    const scene = () => sceneRef.current;
    const hooks: DebugHooks = {
      stats: () => scene()?.stats() ?? { trees: 0, cubes: 0, flora: 0, frameMs: 0 },
      cameraPos: () => scene()?.debug.cameraPos() ?? { x: 0, y: 0, z: 0 },
      cameraTarget: () => scene()?.debug.cameraTarget() ?? null,
      userHold: () => scene()?.camera.paused ?? false,
      wheelEngaged: () => scene()?.debug.wheelEngaged() ?? false,
      groundAt: (x, z) => scene()?.debug.groundAt(x, z) ?? NaN,
      // the look direction (target − position) of the pose the camera controller applies each frame
      cameraLook: () => {
        const p = scene()?.camera.pose();
        return p ? { x: p.target.x - p.pos.x, y: p.target.y - p.pos.y, z: p.target.z - p.pos.z } : null;
      },
      girthAt: (id) => scene()?.debug.girthAt(id) ?? null,
      heightAt: (id) => scene()?.debug.heightAt(id) ?? null,
      cellsAt: (id) => scene()?.debug.cellsAt(id) ?? null,
      levelOf: (id) => scene()?.debug.levelOf(id) ?? null,
      levelCubes: () => scene()?.debug.levelCubes() ?? [],
      holdPose: (pos, target) => scene()?.debug.holdPose(pos, target),
      // the engine reports a flight as dx = from.x − to.x
      flights: () => (scene()?.debug.flights() ?? []).map((f) => ({ treeId: f.treeId, fromX: f.dx, toX: 0 })),
      shownAbs: () => scene()?.shownAbs() ?? NaN,
      treeIds: () => [...(stateRef.current?.trees.keys() ?? [])],
      snapshot: async () => {
        const s = scene();
        return s ? s.snapshot() : null;
      },
      exportView: () => scene()?.exportView() ?? null,
      followGoal: () => scene()?.debug.followGoal() ?? null,
      treePx: () => scene()?.debug.treePx() ?? [],
      spanLength: () => scene()?.debug.spanLength() ?? 0,
      treeDist: () => scene()?.debug.treeDist() ?? [],
      pickWorld: (x, y, z) => scene()?.debug.pickWorld(x, y, z) ?? null,
    };
    window.__forest = hooks;
    return () => {
      // the card and the full window may overlap during a route change: remove only the hooks this view installed
      if (window.__forest === hooks) delete window.__forest;
    };
  }, []);

  const loaded = state !== undefined;
  const trees = state ? state.trees.size : null;
  const flowers = state ? state.flowers.length : null;
  const empty = loaded && trees === 0 && flowers === 0;
  const dataError = authError || (error && !(error instanceof ForestAuthError));
  const errorText = authError
    ? "Your forest access has ended. Please sign in again."
    : `Forest data unavailable${error instanceof Error ? `: ${error.message}` : "."}`;

  if (isCard) {
    const span = range.replace(/^(\d+)/, "$1 "); // "6h" → "6 h": the Stats page's range
    // a still has no live camera to continue: it opens the full window with no hand-off
    const openFull = () => {
      // one-shot: a second click would find no scene (view null) and wipe the hand-off the first one wrote
      if (openedRef.current) return;
      openedRef.current = true;
      setOpening(true);
      let view: CameraView | null = null;
      try {
        view = stillUrl === null ? (sceneRef.current?.exportView() ?? null) : null;
      } catch {
        /* a broken scene: open without a view */
      }
      onOpenFull?.(view, release);
    };
    return (
      <div
        ref={rootRef}
        data-theme="retro-dark"
        data-testid="forest-view"
        data-variant="card"
        className="relative h-full w-full overflow-hidden bg-[#0e1210] text-[#eef2ec]"
      >
        <div ref={hostRef} data-testid="forest-canvas-host" className="absolute inset-0" hidden={unavailable || stillUrl !== null} />
        {engaged && stillUrl === null && !unavailable && (
          // the focus cue: the card captures two-finger slides and the wheel until Esc, a click elsewhere or a blur
          <div
            aria-hidden="true"
            data-testid="forest-card-focus"
            className="pointer-events-none absolute inset-0 z-10 rounded-lg ring-2 ring-inset ring-sky-300/50"
          />
        )}
        {stillUrl !== null && (
          <button type="button" title="Open full window" onClick={openFull} className="absolute inset-0 block h-full w-full">
            {/* a data URL from the scene: next/image has nothing to optimise */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              alt="Session forest"
              src={stillUrl}
              className="h-full w-full object-cover"
              onError={() => {
                // a bad or empty data URL: the placeholder (with the full-window button) instead of a broken image
                setStillUrl(null);
                setUnavailable(true);
              }}
            />
          </button>
        )}
        {unavailable && !opening && (
          <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-[#7aa5c8] to-[#3d5a44]">
            <p className={cn(GLASS, "px-3 py-2 text-xs")} role="status">
              3D view unavailable.
            </p>
          </div>
        )}
        {opening && (
          <div className="absolute inset-0 grid place-items-center bg-[#0e1210]">
            <p className={cn(GLASS, "px-3 py-2 text-xs")} role="status">
              Opening the full window…
            </p>
          </div>
        )}
        <span className={cn(GLASS, "pointer-events-none absolute left-3 top-3 px-2 py-1 text-[11px]")}>
          {span} · live · {DELAY_S} s delay
        </span>
        {stillUrl === null && !unavailable && !opening && (
          <button
            type="button"
            aria-label="Reset view"
            title="Reset view"
            onClick={resetView}
            className={cn(GLASS, "absolute right-12 top-3 grid h-8 w-8 place-items-center text-sm hover:bg-white/10")}
          >
            ⟲
          </button>
        )}
        <button
          type="button"
          aria-label="Open full window"
          title="Open full window"
          onClick={openFull}
          className={cn(GLASS, "absolute right-3 top-3 grid h-8 w-8 place-items-center text-sm hover:bg-white/10")}
        >
          ⛶
        </button>
        {empty && !dataError && !opening && (
          <p role="status" className={cn(GLASS, "pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1.5 text-xs")}>
            No traffic in the last {span}
          </p>
        )}
        {dataError && (
          <p role="alert" className={cn(GLASS, "absolute bottom-3 left-3 right-3 px-3 py-1.5 text-xs text-[#f8b4a0]")}>
            {errorText}
          </p>
        )}
        {tip && tipContent && <ForestTooltip ref={tipRef} tip={tipContent} left={tip.x + 14} top={tip.y + 14} />}
      </div>
    );
  }

  return (
    <div
      ref={rootRef}
      data-theme="retro-dark"
      data-testid="forest-view"
      className="fixed inset-0 z-50 overflow-hidden bg-[#0e1210] text-[#eef2ec]"
    >
      <div ref={hostRef} data-testid="forest-canvas-host" className="absolute inset-0" hidden={unavailable} />
      {unavailable && (
        <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-[#7aa5c8] to-[#3d5a44]">
          <p className={cn(GLASS, "px-4 py-3 text-sm")} role="status">
            3D view unavailable. The panels still work.
          </p>
        </div>
      )}

      <div className="pointer-events-none absolute inset-x-4 top-4 bottom-20 flex justify-between gap-4">
        <div className="pointer-events-auto flex w-[min(320px,40vw)] min-h-0 flex-col gap-1.5 self-start">
          <div data-forest-panel>
            <Description trees={trees} flowers={flowers} loaded={loaded} empty={empty} range={range} />
          </div>
          <div data-forest-panel>
            <Legend />
          </div>
          {dataError && (
            <p role="alert" className={cn(GLASS, "px-3 py-2 text-xs text-[#f8b4a0]")}>
              {errorText}
            </p>
          )}
        </div>
        <div className="pointer-events-auto flex w-[min(560px,45vw)] min-h-0 flex-col gap-1.5">
          <div data-forest-panel className="flex-none">
            <TokenBars state={state ?? null} atRel={atRel} />
          </div>
          <div data-forest-panel className="flex min-h-0 flex-col">
            <InFlight
              mode={mode}
              paused={authError !== null}
              isPaused={() => authLatch.current}
              onRowHover={onRowHover}
              onRowClick={onRowClick}
              onAuthError={authFailed}
            />
          </div>
        </div>
      </div>

      <div data-forest-panel className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-1.5">
        {onRange && isForestRange(range) && <RangeSwitch range={range} onRange={onRange} />}
        <Controls mode={camMode} onMode={changeMode} />
        <button
          type="button"
          aria-label="Reset view"
          title="Reset view"
          onClick={resetView}
          className={cn(GLASS, "grid h-9 w-9 place-items-center text-sm hover:bg-white/10")}
        >
          ⟲
        </button>
      </div>

      {tip && tipContent && (
        <ForestTooltip ref={tipRef} tip={tipContent} left={tip.x + 14} top={tip.y + 14} />
      )}

      <div data-forest-panel className="absolute bottom-4 right-4 flex gap-1.5">
        <button
          type="button"
          aria-label="Fullscreen"
          title="Fullscreen"
          onClick={enterFullscreen}
          className={cn(GLASS, "grid h-9 w-9 place-items-center text-sm hover:bg-white/10")}
        >
          ⛶
        </button>
        {onSignOut ? (
          <button
            type="button"
            title="Sign out of your forest"
            onClick={onSignOut}
            className={cn(GLASS, "h-9 px-3 text-sm hover:bg-white/10")}
          >
            Sign out
          </button>
        ) : (
          <button
            type="button"
            aria-label="Close the full window"
            title="Close (Esc)"
            onClick={() => onClose?.()}
            className={cn(GLASS, "grid h-9 w-9 place-items-center text-sm hover:bg-white/10")}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
