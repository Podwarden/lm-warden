/**
 * The forest scene (spec §6.1 `engine/scene.ts`): renderer, composer, sky, land, trees and the camera, drawn on demand.
 * Live view only (spec §10.6). The pure orchestration (scene time, the clock, the camera step, data intake, rebuild
 * planning) lives in frame.ts; this file applies it to three.js. Ported from forest-real.html (renderer and composer
 * setup, `resize`, the render-on-demand `loop`, the OrbitControls wiring) in the voxel look of forest-styles.html: no
 * tone mapping, one whole-frame saturation pass, always day.
 * - Scene time is internal (origin: the server's now in the first state; frame.ts); a server re-anchor changes nothing
 *   on screen. The public API speaks absolute time (unix seconds): `shownAbs()`, `jumpAbs()`, pick times.
 * - The scene owns the clock (30 s behind the server's now).
 * - Pixel ratio ≤ 1.5 (≤ 1 compact); the composer follows the renderer's pixel ratio (README §4.6). No bloom: the voxel
 *   look has no glow (the style mockup renders without one).
 * - The shadow map (2048) is redrawn only when something moved: a swap, a flight, an emergence, the key light, the shore;
 *   growing flora at most twice a second.
 * - Growth between rebuilds is only in the shaders (`uNow`: new cubes scale in from their born; voxels.ts).
 * - One CameraController for the scene's life: stepped every frame (its step ends a drag pause), `pose()` applied every
 *   frame unless the user holds it, `setEvents` only when the trees changed, `setPose` only during a drag or a gesture.
 * - Trackpad gestures (Plan 4, gestures.ts): two fingers pan, pinch zooms to the pointer, Shift + two fingers or a
 *   Safari twist orbit, a mouse wheel zooms; click-drag stays OrbitControls'. Each one holds the camera like a drag
 *   (`userInteracted`, 9 s) and is clamped above ground. The card lets the page scroll until it is clicked into; Esc,
 *   a window blur or a pointer-down elsewhere lets it go (`onWheelEngaged`: the card's focus ring).
 * - `pause()` (the Stats card offscreen) stops the render loop (cancelAnimationFrame) and freezes the clock; new data is
 *   queued. `resume()` applies it and follows the hidden-tab rule (frame.ts), then restarts the loop.
 * - `snapshot()` waits (≤ SNAPSHOT_WAIT_MS) for the builds outstanding at the call, then renders once and reads the
 *   canvas in the same task (the drawing buffer is not preserved), paused or running.
 * - `exportView()` / `importView()` hand the camera pose and mode between surfaces, front-relative (x − X(shown time);
 *   handoff.ts). Imported before the first data, the view waits for it: the receiving scene's front is known only then.
 * - Bigger when far (farscale.ts): every frame the trees' and flora's render scale follows the camera's distance (1 in
 *   every 1h/6h/24h view, larger beyond), capped so no scaled crown overlaps a neighbour; the camera's crown checks use
 *   the scaled crowns, the follow fit the true-size points.
 * - Ground (follow-up A): the controller is clamped to the land's own height (`groundHeight` over the timeline's width
 *   at the shown time). OrbitControls cannot orbit below the horizon (a drag that starts lower keeps that limit, so it
 *   never snaps), and every drag or zoom pose is clamped by the controller and written back to the three.js camera.
 * - `resetView()` ("Reset view", follow-up B): ends the user's hold or focus and glides to the mode's default view.
 * - Range (follow-up C): the state's `range` is shown and framed. A switch keeps the fixed scene origin (frame.ts: the
 *   trees it brings in are history, known turns keep their times), keeps the trees near the front in place
 *   (frozen places and X: `SceneLayout`), reaches the land back to the range's start, frames the whole span in follow mode and glides there
 *   (≤ 3 s). The card's tree budget follows the range (`cardTreeBudget`); a far fit thins the fog and pushes the far
 *   plane out (`viewDepth`), and lets OrbitControls zoom that far out.
 */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type GestureIntent, type Pose, WheelGate, applyOrbit, applyPan, applyZoom, twistOrbit, wheelIntent } from "./gestures";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { type Place, type Timeline, TREE_SCALE } from "@/lib/forest/timeline";
const placeIdx = (ps: readonly Place[], cut: number) => Math.max(0, ps.findLastIndex((q) => q.from <= cut));
import { type ForestState, RANGE_S } from "@/lib/forest/types";
import { CameraController, type CameraMode, ISO, type PanelRect, type SafeRect, framePoints, safeRectFromPanels } from "./camera";
import { createTrail } from "./flights";
import type { ForestView } from "./handoff";
import {
  BEHIND, FAR_MIN, FrameCore, MAX_BUILT, MAX_ZOOM_OUT, PixelRatioGovernor, SceneLayout, type TreeInfo, cardTreeBudget, cutOnJump, treeInfo, viewDepth, zoomOutLimit,
} from "./frame";
import { FAR_D0, effectiveDistance, treeBudget } from "./farscale";
import { cameraGround, createLand, groundHeight, terraceHeight } from "./land";
import { SaturationShader, createMaterials } from "./materials";
import { createSky } from "./sky";
import { type BuildJob, type BuildReply, handleBuild } from "@/lib/forest/geometry/worker";
import { type FloraPick, createGround } from "./ground";
import { type QuietPick, createQuietMarks } from "./quiet";
import { type Builder, CUBE_CAP, type TreePickResult, createTreeLayer } from "./trees";

export { type ForestView, HANDOFF_TTL_MS, decodeHandoff, encodeHandoff } from "./handoff";

/** Hover detail: a turn of a tree, a ghost, a flower bed or a mushroom. */
export type PickResult = TreePickResult | FloraPick | QuietPick;

const FOV = 36;
/** Trees are (re)selected and rebuild checks run this often (ms). */
const TICK_MS = 250;
/** The shore is re-measured at most this often (ms): the spit widens over minutes. */
const LAND_MS = 5000;
/** Growing flora redraws the shadow map at most this often (ms). */
const FLORA_SHADOW_MS = 500;
/** A snapshot waits at most this long (ms) for the builds outstanding when it was asked for. */
const SNAPSHOT_WAIT_MS = 5000;

export interface ForestSceneOptions {
  compact: boolean;
  /** The WebGL context was lost: the shell shows its "3D view unavailable" fallback. */
  onContextLost?: () => void;
  /** The camera's mode changed (the toggle, or a row click switching to cinematic): the shell's toggle follows it. */
  onCameraMode?: (mode: CameraMode) => void;
  /**
   * The Stats card: a wheel or two-finger slide is the page's (it scrolls) until the canvas is clicked or tapped into;
   * a pinch zooms the card regardless (gestures.ts `WheelGate`). The full window captures every gesture.
   */
  wheelNeedsFocus?: boolean;
  /** The card was clicked into (true) or let go (false: Esc, a window blur, a pointer-down elsewhere, a pause): the
   * shell shows a focus ring while it captures the trackpad. */
  onWheelEngaged?: (engaged: boolean) => void;
}

export interface ForestScene {
  /** New forest data (any t0 anchor); the scene's own clock shows the server's now − 30 s. */
  setState(state: ForestState): void;
  /** The shown history time, absolute (unix seconds): token bars and the in-flight panel follow it. */
  shownAbs(): number;
  /** Show absolute time `abs` now, as one jump (never past now − 30 s). Only moves the shown time; never rebuilds. */
  jumpAbs(abs: number): void;
  /**
   * Hover detail at client coordinates. Times are absolute (unix seconds); a late turn's time is when it was shown to
   * grow (its arrival), not its request start.
   */
  pick(x: number, y: number): PickResult | null;
  /** Marks a session's joints (in-flight row hover, by its `forest_session`); null clears. */
  highlight(sessionId: string | null): void;
  resize(): void;
  dispose(): void;
  /** `cubes`: tree and ghost cubes drawn; `flora`: the voxel cubes of flower beds and mushrooms; `levels`: built trees
   * per level of detail (0 fine … 2 coarsest). */
  /** `pixelRatio`: the render resolution chosen by the adaptive governor (R26); always set by the scene (optional
   * only so the shell's no-scene fallback stays valid). */
  /** `paused`: the render loop is stopped (`pause()`); for the debug hooks. */
  /** `drawCalls`, `triangles`: what the last rendered frame drew, every pass (shadow map, scene, post) counted. */
  stats(): { trees: number; cubes: number; flora: number; frameMs: number; pixelRatio?: number; paused?: boolean; levels?: number[]; drawCalls?: number; triangles?: number };
  /** Stops the render loop and the shown time (the card scrolled offscreen); new data is queued until `resume`. */
  pause(): void;
  /** Applies the queued data, moves the clock as after a hidden tab (no late burst), restarts the render loop. */
  resume(): void;
  /** One frame rendered now, as a PNG data URL, once the trees present now are built (waits at most 5 s). */
  snapshot(): Promise<string>;
  /** The camera as shown, front-relative (x − X(shown time)), with its mode; null before the first data. */
  exportView(): ForestView | null;
  /**
   * Puts the camera at this front-relative pose (mapped with this scene's own front) and mode, with no animation; it
   * continues from there. Before the first data it is held and replaces the front placement when the data lands.
   * Ignored during a live drag, and for a view that is not front-relative.
   */
  importView(v: ForestView): void;
  readonly camera: CameraController;
  setCameraMode(mode: CameraMode): void;
  /** "Reset view": ends any user override (drag or zoom hold, focus) and glides (≤ 3 s) to the mode's default view. */
  resetView(): void;
  /** Open panels' client rects: the follow fit leaves them out. */
  setPanels(rects: readonly PanelRect[]): void;
  /** For the `?debug=1` hooks (Task 7/9). */
  debug: {
    /** Trunk base girth on screen now: the wood cells' cross-section, each at its scale-in. */
    girthAt(treeId: string): number | null;
    /** Top of a tree's fully grown cubes on screen now (units above its base). */
    heightAt(treeId: string): number | null;
    /** Fine cells of a tree's build on screen (grows as turns land). */
    cellsAt(treeId: string): number | null;
    /** The level of detail a tree is drawn at (null: not built). */
    levelOf(treeId: string): number | null;
    /** The cubes the built trees would draw at each level of detail (0 … 2). */
    levelCubes(): number[];
    /** Puts the camera at this pose as a user drag would (held, the automatic camera resumes later): screenshots. */
    holdPose(pos: { x: number; y: number; z: number }, target: { x: number; y: number; z: number }): void;
    /** The shown history time, scene time. */
    rel(): number;
    flights(): { treeId: string; dx: number }[];
    cameraPos(): { x: number; y: number; z: number };
    /** The orbit target (OrbitControls' target) as shown. */
    cameraTarget(): { x: number; y: number; z: number };
    /** The card was clicked into: its wheel and two-finger slide drive the camera (gestures.ts `WheelGate`). */
    wheelEngaged(): boolean;
    /** The ground's height at (x, z): the camera's floor is this + GROUND_CLEARANCE. */
    groundAt(x: number, z: number): number;
    /** Where the follow fit wants the camera (world), or null (no fit yet, or cinema). */
    followGoal(): { x: number; y: number; z: number } | null;
    /** Every built tree's height on screen now (CSS px, base to its highest grown wood). */
    treePx(): number[];
    /** The spit's length from the shown range's start to the front (units). */
    spanLength(): number;
    /** What the hover pick finds at the screen point of world (x, y, z). */
    pickWorld(x: number, y: number, z: number): PickResult | null;
    /** Every built tree's distance from the camera and its grown height (units). */
    treeDist(): { id: string; d: number; top: number; scale: number; x: number; z: number; r: number }[];
  };
}

/** Builds on the calling thread, replying in a microtask (tests, and browsers without module workers). */
export function createDirectBuilder(): Builder {
  let alive = true;
  const b: Builder = {
    onReply: null,
    post: (job) => queueMicrotask(() => alive && b.onReply?.(handleBuild(job).reply)),
    dispose: () => void (alive = false),
  };
  return b;
}

/** A small pool of geometry workers; if a worker fails to load, its jobs (and later ones) are built directly. */
export function createWorkerBuilder(size = 2): Builder {
  const direct = createDirectBuilder();
  const pool: { w: Worker; open: Map<number, BuildJob> }[] = [];
  let next = 0, broken = typeof Worker === "undefined";
  const b: Builder = {
    onReply: null,
    post(job) {
      if (broken || !pool.length) return direct.post(job);
      const p = pool[next++ % pool.length];
      p.open.set(job.id, job);
      p.w.postMessage(job);
    },
    dispose() {
      for (const p of pool) p.w.terminate();
      direct.dispose();
    },
  };
  direct.onReply = (r) => b.onReply?.(r);
  if (!broken) {
    try {
      for (let i = 0; i < size; i++) {
        const w = new Worker(new URL("@/lib/forest/geometry/worker.ts", import.meta.url), { type: "module" });
        const p = { w, open: new Map<number, BuildJob>() };
        w.onmessage = (e: MessageEvent<BuildReply>) => (p.open.delete(e.data.id), b.onReply?.(e.data));
        w.onerror = w.onmessageerror = () => {
          broken = true;
          for (const job of p.open.values()) direct.post(job);
          p.open.clear();
        };
        pool.push(p);
      }
    } catch {
      broken = true;
    }
  }
  return b;
}

export function createForestScene(canvas: HTMLCanvasElement, opts: ForestSceneOptions): ForestScene {
  const shadows = !opts.compact;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  // adaptive render resolution (R26): starts at min(DPR, 1.5) (≤ 1 compact), steps by 0.25 with the frame time
  const governor = new PixelRatioGovernor(window.devicePixelRatio || 1, opts.compact);
  renderer.setPixelRatio(governor.ratio);
  // the style mockup renders linear with no tone mapping, then saturates the whole frame (the pass below)
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = shadows;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;
  // the draw-call count covers a whole frame: the composer renders several passes, each of which would reset it
  renderer.info.autoReset = false;
  let drawn = { calls: 0, triangles: 0 };

  const scene = new THREE.Scene();
  const cam3 = new THREE.PerspectiveCamera(FOV, 1, 0.1, FAR_MIN);
  cam3.position.set(-12, 13, 46);
  // trackpad gestures (gestures.ts), registered before OrbitControls: every wheel event stops here, so OrbitControls'
  // own wheel dolly (which always prevents the default, and so swallowed the Stats page's scroll) never runs
  const gate = new WheelGate(opts.wheelNeedsFocus === true, (e) => opts.onWheelEngaged?.(e));
  const onWheel = (e: WheelEvent) => {
    e.stopImmediatePropagation();
    if (!gate.captures(e)) return; // the page scrolls over an unengaged card
    e.preventDefault(); // a pinch never zooms the page
    gesture(wheelIntent(e, canvas.clientHeight), e.clientX, e.clientY);
  };
  canvas.addEventListener("wheel", onWheel, { passive: false });
  /** Safari's GestureEvent (not in the DOM typings): the twist's rotation in degrees. */
  type GestureLike = Event & { rotation?: number };
  let twist = 0;
  const onGestureStart = (e: Event) => {
    e.preventDefault();
    twist = (e as GestureLike).rotation ?? 0;
  };
  const onGestureChange = (e: Event) => {
    e.preventDefault();
    const r = (e as GestureLike).rotation ?? 0;
    if (gate.capturesTwist()) gesture(twistOrbit(twist, r), 0, 0);
    twist = r;
  };
  const onGestureEnd = (e: Event) => e.preventDefault();
  canvas.addEventListener("gesturestart", onGestureStart);
  canvas.addEventListener("gesturechange", onGestureChange);
  canvas.addEventListener("gestureend", onGestureEnd);
  // the full window covers the page: a Safari pinch over its panels must not zoom the page either (not the card's page)
  const onDocGesture = (e: Event) => e.preventDefault();
  // …nor a Chrome/Firefox pinch (Ctrl + wheel) over them
  const onDocPinch = (e: WheelEvent) => void (e.ctrlKey && e.preventDefault());
  if (!gate.needsEngage) {
    document.addEventListener("gesturestart", onDocGesture);
    document.addEventListener("wheel", onDocPinch, { passive: false });
  }
  // the card is clicked or tapped into (a pointer-down on its canvas) or left (a pointer-down anywhere else)
  const onDocPointerDown = (e: PointerEvent) => gate.pointerDown(e.target === canvas);
  document.addEventListener("pointerdown", onDocPointerDown, true);
  // Esc or leaving the window lets the card go (and its focus ring with it)
  const onDocKey = (e: KeyboardEvent) => gate.key(e.key);
  const onWinBlur = () => gate.blur();
  if (gate.needsEngage) {
    document.addEventListener("keydown", onDocKey);
    window.addEventListener("blur", onWinBlur);
  }
  const controls = new OrbitControls(cam3, canvas);
  controls.target.set(3, 8.5, 0);
  controls.enableDamping = false; // no inertia: the drag pose handed to the camera controller is the final one
  controls.minDistance = 6;
  controls.maxDistance = MAX_ZOOM_OUT;
  /** Never below the horizon (a drag that starts lower, e.g. from a low cinematic orbit, keeps its own polar angle as the
   * limit for that drag, so OrbitControls does not snap it up). */
  const HORIZON = Math.PI / 2;
  controls.maxPolarAngle = HORIZON;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, cam3));
  const saturate = new ShaderPass(SaturationShader);
  composer.addPass(saturate);
  const output = new OutputPass();
  composer.addPass(output);

  // always day (Plan 4): a flat cyan sky and the mockup's day lights at every hour
  const sky = createSky(scene, { shadows });
  const mats = createMaterials();
  const land = createLand(scene, { compact: opts.compact, shadows, toon: mats.toon });
  const trail = createTrail(scene);
  /** What stands on the land stands on its terraces: a pure function of the world position (never of the data). */
  const baseAt = (x: number, z: number) => terraceHeight(x, z);
  const quiet = createQuietMarks(scene, { shadows, baseAt, toon: mats.toon });
  let camInputsDirty = false, shadowDirty = true, dirty = true;
  // budgets (spec §6.5): one cube budget shared by the trees, their ghosts and the ground flora (card: 15k)
  const instanceCap = opts.compact ? 15_000 : CUBE_CAP;
  const ground = createGround(scene, { shadows, cap: Math.floor(instanceCap / 5), toon: mats.toon, baseAt });
  const layer = createTreeLayer(scene, mats, trail, {
    shadows,
    builder: typeof Worker === "undefined" ? createDirectBuilder() : createWorkerBuilder(Math.max(1, Math.min(3, (navigator.hardwareConcurrency || 2) - 1))),
    onChange: () => void (camInputsDirty = shadowDirty = dirty = true),
    cubeCap: () => instanceCap - ground.count(),
    // the card: 12 trees for 1h / 6h, 30 for 24h / 7d
    // a far view (7 d) builds every tree it can (re-review 3, N3), at coarser levels of detail within the cube budget
    // (capped by the cubes left after the flora, so the card's budget always has a coarser level to fall back to)
    maxTrees: () => treeBudget(opts.compact ? (state ? cardTreeBudget(state.range) : 12) : MAX_BUILT, farView, instanceCap - ground.count()),
    // a tree anywhere in the shown range may be built (the fit frames all of it)
    behind: () => Math.max(BEHIND, controls.target.x - spanX0() + 20),
    baseAt,
  });

  /** The scene's one camera controller, for its whole lifetime (first data resets it, never replaces it). */
  const ctl = new CameraController({ fov: FOV, aspect: 1, pos: cam3.position, target: controls.target });
  ctl.onModeChange = (m) => opts.onCameraMode?.(m);
  const core = new FrameCore(ctl);
  let state: ForestState | null = null, timeline: Timeline | null = null;
  /** The camera stands far off (bigger-when-far territory): the build budget rises (farscale.ts treeBudget). */
  let farView = false;
  /** The layout: frozen places, and the x shift a range switch keeps the front with (frame.ts `SceneLayout`). */
  const layout = new SceneLayout();
  let infos = new Map<string, TreeInfo>();
  let rel = 0, primed = false;
  let panels: readonly PanelRect[] = [];
  let safe: SafeRect = { l: -1, r: 1, t: 1, b: -1 };
  let lastTick = -Infinity, lastLand = -Infinity, lastFlora = -Infinity, lastT: number | null = null, frameMs = 0;
  /** The previous animation frame rendered: only intervals between two rendered frames measure the render cost. */
  let renderedPrev = false;
  let raf = 0, alive = true, lost = false, paused = false;
  /** Wall time of the pause: a paused snapshot animates at it, so flights and emergences hold still. */
  let pausedAt = 0;
  // the land's own height at the shown time (flat until the first data), from the widths the strip is written from,
  // with its terraces; set here, after `timeline` and `rel` exist: setGround clamps at once
  const groundAt = (x: number, z: number) => {
    const tl = timeline;
    return tl ? groundHeight(x, z, (xx, side) => tl.width(xx, side, rel), land.startX(), land.endX()) : 0.04;
  };
  /** The camera's floor: the ground with a continuous envelope over the terraces' steps (land.ts `cameraGround`). */
  const floorAt = (x: number, z: number) => {
    const tl = timeline;
    return tl ? cameraGround(x, z, (xx, side) => tl.width(xx, side, rel), land.startX(), land.endX()) : 0.04;
  };
  /** Where the land must start (x): before the span's oldest tree and the oldest flower shown. */
  function landStart(): number {
    const tl = timeline, s = state;
    let x = spanX0();
    if (tl && s) for (const f of s.flowers) x = Math.min(x, tl.X(f[0]) - 6);
    return x;
  }
  /**
   * Where the shown range's span starts (x): its oldest tree's first place (6 units before it), never before
   * X(now − range). The trees in [now − range, now] are what the fit frames: an idle stretch before the oldest of them
   * (a 7 d range over two busy days) is not. −Infinity before the data; X(now − range) with no tree.
   */
  function spanX0(): number {
    const tl = timeline, s = state;
    if (!tl || !s) return -Infinity;
    const edge = tl.X(s.now - (RANGE_S[s.range] ?? 0));
    let oldest = Infinity;
    for (const info of infos.values()) if (info.places.length) oldest = Math.min(oldest, info.places[0].x - 6);
    return Number.isFinite(oldest) ? Math.max(edge, oldest) : edge;
  }
  ctl.setGround(floorAt);

  const applyPose = (p = ctl.pose()) => {
    cam3.position.set(p.pos.x, p.pos.y, p.pos.z);
    controls.target.set(p.target.x, p.target.y, p.target.z);
    cam3.lookAt(controls.target);
  };

  /** A view handed over before the first data (front-relative): applied when the data lands (placeAtFront). */
  let pendingView: ForestView | null = null;
  const fromFrontRel = (v: ForestView, front: number) => ({
    pos: { ...v.pos, x: v.pos.x + front },
    target: { ...v.target, x: v.target.x + front },
  });

  /** The follow view's fit: the trees growing near the front plus a stretch of shore, inside the free area. */
  const fit = (frame: ReturnType<typeof layer.cameraInputs>["frame"]) => {
    const tl = timeline;
    // the whole shown range, [now − range, now] (follow-up C)
    if (tl) ctl.follow(framePoints(frame, tl.X(rel), (x, side) => tl.width(x, side, rel), spanX0()), safe);
  };
  const refollow = () => fit(layer.cameraInputs(rel).frame);

  /** First data: start near the forest front (the newest trees), not at the planner's fallback. */
  const placeAtFront = (s: ForestState, tl: Timeline) => {
    let best: Place | null = null;
    for (const id of s.trees.keys()) {
      const ps = tl.places(id).filter((p) => p.from <= rel);
      const p = ps.at(-1);
      if (p && (!best || p.x > best.x)) best = p;
    }
    const target = best ? { x: best.x, y: 3, z: best.z } : { x: tl.X(rel), y: 3, z: 0 };
    const d = 42;
    const pos = { x: target.x + ISO.x * d, y: target.y + ISO.y * d, z: target.z + ISO.z * d };
    // a view imported before the first data: mapped with this scene's front now that it is known; it replaces the
    // front placement (placeInitial), unless the import is refused (a live drag)
    const v = pendingView;
    pendingView = null;
    let imported = false;
    if (v) {
      const w = fromFrontRel(v, tl.X(rel));
      imported = ctl.importView(w.pos, w.target, v.mode);
    }
    // in follow mode (the card) the first placement is the fit itself: the capped follow camera would otherwise
    // glide for seconds from the front placement to its own framing (an imported view still wins: placeInitial)
    let first = { pos, target };
    if (!imported && ctl.cameraMode === "follow") {
      fit(layer.cameraInputs(rel).frame);
      first = ctl.followPose() ?? first;
    }
    ctl.placeInitial(first.pos, first.target);
    // a hand-off in follow mode: from the sender's view to this window's own fit (another aspect, panels and range),
    // by a gentle planned glide (≤ 0.1 per frame at first, there within SETTLE_MAX_S)
    if (imported && ctl.cameraMode === "follow") {
      fit(layer.cameraInputs(rel).frame);
      ctl.settleToFit();
    }
    applyPose();
  };

  const syncCamera = (nowMs: number) => {
    const inp = layer.cameraInputs(rel);
    ctl.setCrowns(inp.crowns);
    ctl.setTips(inp.tips);
    ctl.setGrowth(inp.newest, nowMs);
    fit(inp.frame);
  };

  const tickTrees = (nowMs: number) => {
    lastTick = nowMs;
    layer.tick(rel, core.takeStep(), controls.target.x);
    if (ground.tick(rel, controls.target.x)) shadowDirty = dirty = true; // the trees' budget follows at the next view
    if (state && timeline && nowMs - lastLand > LAND_MS) {
      lastLand = nowMs;
      const tl = timeline;
      if (land.update((x, side) => tl.width(x, side, rel), tl.X(state.now), landStart())) shadowDirty = dirty = true;
    }
  };

  // OrbitControls 'start'/'end' (drag, wheel, touch) pause the active camera; the pose is handed over only while
  // the user is dragging, so the camera resumes 9 s later from where the user left it
  let dragging = false;
  const onStart = () => {
    dragging = true;
    const off = cam3.position.clone().sub(controls.target);
    // out to 1.5× the fit's distance (a wide range's fit stands far off); a camera already further out keeps its distance
    // for this drag only, so it does not snap in, and repeated wheel-outs cannot ratchet the limit (review M2)
    controls.maxDistance = Math.max(zoomOutLimit(ctl.fitDistance()), Math.min(off.length(), controls.maxDistance));
    controls.maxPolarAngle = Math.max(HORIZON, Math.acos(THREE.MathUtils.clamp(off.y / Math.max(1e-9, off.length()), -1, 1)));
    ctl.setDragging(true, performance.now());
  };
  // the controller clamps the pose to the ground (camera and target); written back, OrbitControls continues from it
  const adoptUserPose = () => {
    if (ctl.setPose(cam3.position, controls.target)) applyPose();
  };
  const onChange = () => {
    if (!dragging) return;
    adoptUserPose();
    dirty = true;
  };
  const onEnd = () => {
    adoptUserPose();
    dragging = false;
    controls.maxPolarAngle = HORIZON;
    ctl.setDragging(false, performance.now());
  };
  controls.addEventListener("start", onStart);
  controls.addEventListener("change", onChange);
  controls.addEventListener("end", onEnd);
  /**
   * A trackpad gesture or a wheel step (gestures.ts): a user interaction like a drag. It holds the automatic camera
   * for FOLLOW_PAUSE_MS from now; the pose goes through the controller (`setPose`: the ground clamp) and back to the
   * three.js camera. Ignored while a drag is in progress (the drag owns the camera). `cx`, `cy`: the pointer (zoom).
   */
  function gesture(i: GestureIntent, cx: number, cy: number): void {
    if (!alive || dragging) return;
    ctl.userInteracted(performance.now());
    const pos = { x: cam3.position.x, y: cam3.position.y, z: cam3.position.z };
    const target = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
    let p: Pose;
    if (i.kind === "pan") p = applyPan(pos, target, i.dx, i.dy);
    else if (i.kind === "orbit") p = applyOrbit(pos, target, i.dTheta, i.dPhi);
    else {
      const rect = canvas.getBoundingClientRect(), d = cam3.position.distanceTo(controls.target);
      const nx = ((cx - rect.left) / Math.max(1, rect.width)) * 2 - 1, ny = 1 - ((cy - rect.top) / Math.max(1, rect.height)) * 2;
      // the drag's limits: out to the fit's zoom-out limit (a camera already further out may only come in), in to
      // minDistance (a camera already closer may only go out)
      p = applyZoom(pos, target, nx, ny, i.factor, {
        fovDeg: cam3.fov,
        aspect: cam3.aspect,
        minDist: Math.min(controls.minDistance, d),
        maxDist: Math.max(zoomOutLimit(ctl.fitDistance()), d),
      });
    }
    if (ctl.setPose(p.pos, p.target)) applyPose();
    dirty = true;
  }
  const onLost = (e: Event) => {
    e.preventDefault();
    lost = true;
    opts.onContextLost?.();
  };
  canvas.addEventListener("webglcontextlost", onLost);
  // shown again after a hidden stretch: one jump to the bound, and the gap's turns are history (final review I2)
  const onVisible = () => {
    if (document.hidden || paused) return; // a paused scene resumes through resume()
    core.resumeAfterHidden(performance.now());
    dirty = true;
  };
  document.addEventListener("visibilitychange", onVisible);

  const frame = (t: number) => {
    if (!alive || paused) return;
    raf = requestAnimationFrame(frame);
    if (lost || document.hidden) return void ((lastT = null), (renderedPrev = false));
    const t0 = performance.now();
    const interval = lastT === null ? null : t - lastT;
    const dt = interval === null ? 0 : Math.min(0.1, interval / 1000);
    lastT = t;
    if (interval !== null && renderedPrev) {
      frameMs = frameMs ? frameMs * 0.9 + interval * 0.1 : interval;
      const r = governor.frame(t, interval);
      if (r !== null) {
        renderer.setPixelRatio(r); // keeps the canvas's CSS size; the composer follows in resize()
        resize();
      }
    }
    renderedPrev = false;
    const out = core.frame(t0, dt);
    if (out) {
      rel = out.rel;
      if (mats.uniforms.uNow.value !== rel) (mats.uniforms.uNow.value = rel), (dirty = true);
      if (out.pose) applyPose(out.pose); // every frame unless the user holds the camera
      if (out.moved) dirty = true;
    }
    fitDepth();
    // the sea under the view, out to the far plane: no edge of it is ever in frame (review I2)
    if (land.fitSea(cam3.position.x, cam3.position.z, cam3.far)) shadowDirty = dirty = true;
    if (state && t0 - lastTick > TICK_MS) tickTrees(t0);
    if (camInputsDirty) {
      camInputsDirty = false;
      syncCamera(t0);
    }
    // the shown time jumped (a resume after a long pause or hidden tab, a stall): the viewer was not watching, so a
    // follow camera cuts to the fit at the new time instead of gliding there (review I1); cinema keeps its own rules
    if (cutOnJump(out, ctl, refollow)) (applyPose(), (dirty = true));
    if (land.step(dt, { camY: cam3.position.y, target: controls.target, camDist: cam3.position.distanceTo(controls.target) })) dirty = true;
    // "bigger when far" (farscale.ts): trees and flora drawn larger about their bases as the camera pulls back
    const viewH = Math.max(1, canvas.clientHeight);
    farView = effectiveDistance(cam3.position.distanceTo(controls.target), viewH) > FAR_D0;
    if (layer.setView(cam3.position, viewH, dt, t0)) camInputsDirty = shadowDirty = dirty = true;
    if (ground.setView(cam3.position, viewH, rel)) shadowDirty = dirty = true;
    if (layer.animate(t0, dt, rel)) shadowDirty = dirty = true;
    const floraDue = t0 - lastFlora > FLORA_SHADOW_MS;
    if (ground.grow(rel, floraDue)) {
      dirty = true;
      if (floraDue) (lastFlora = t0), (shadowDirty = true);
    }
    if (trail.step(t0)) dirty = true;
    if (sky.follow(controls.target)) shadowDirty = true;
    if (!dirty) return;
    render();
    renderedPrev = true;
  };

  /** Far plane and fog for the camera's distance (a wide range's fit stands far off; `viewDepth`). */
  let depthAt = -1;
  const fitDepth = () => {
    const d = cam3.position.distanceTo(controls.target);
    if (Math.abs(d - depthAt) < Math.max(1, depthAt * 0.02)) return;
    depthAt = d;
    const v = viewDepth(d);
    if (cam3.far !== v.far || cam3.near !== v.near) (cam3.far = v.far), (cam3.near = v.near), cam3.updateProjectionMatrix();
    if (scene.fog instanceof THREE.FogExp2) scene.fog.density = v.fog;
    dirty = true;
  };

  const render = () => {
    dirty = false;
    if (shadowDirty && shadows) renderer.shadowMap.needsUpdate = true;
    shadowDirty = false;
    sky.dome.position.copy(cam3.position);
    renderer.info.reset();
    composer.render();
    drawn = { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles };
  };

  /** The scene part of new data (after the frame core's intake): timeline, tree infos, layer, flora, first pose. */
  const applyState = (ss: ForestState, changed: boolean) => {
    // a range switch: shown trees keep their places (SceneLayout), the fit frames the new span and glides
    const switched = state !== null && ss.range !== state.range;
    state = ss;
    rel = core.clock?.rel ?? rel;
    if (changed || !timeline || switched) {
      const trees = [...ss.trees.values()];
      // X reaches back to the window's start (an hour more: flowers there), not only to its oldest turn
      const from = ss.now - (RANGE_S[ss.range] ?? 0) - 3600;
      // places frozen at first sight (a switch or new data never moves a shown tree), the front kept at a switch
      const tl = (timeline = layout.update(trees, ss.now, core.H, from));
      infos = new Map();
      for (const [id, t] of ss.trees) {
        const ps = tl.places(id);
        if (ps.length) infos.set(id, treeInfo(t, ps, core.inherited.has(id)));
      }
      layer.setData(ss, infos);
      ctl.setBaseOf((id, time) => {
        const ps = infos.get(id)?.places;
        return ps?.length ? { ...ps[Math.max(0, ps.findLastIndex((q) => q.from <= time))], s: TREE_SCALE } : undefined;
      });
    }
    ground.setData(ss, timeline.X, timeline.width, core.lateFor());
    { const tl = timeline; if (quiet.update(tl.gaps(), tl.X, (x, side) => tl.width(x, side))) shadowDirty = dirty = true; }
    if (!primed && ss.trees.size) {
      primed = true;
      placeAtFront(ss, timeline); // the controller was reset (never replaced): give it the inputs below
      camInputsDirty = true;
      lastLand = -Infinity; // later the shore is re-measured every LAND_MS
    }
    if (switched) lastLand = -Infinity; // the land reaches back to the new range's start at once
    tickTrees(performance.now());
    if (switched && primed) {
      refollow();
      ctl.glideToFit(); // follow: a ≤ RETURN_S glide to the new span (re-fits while it runs keep its deadline)
    }
    dirty = true;
  };

  const resize = () => {
    const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
    renderer.setSize(w, h, false);
    composer.setPixelRatio(renderer.getPixelRatio());
    composer.setSize(w, h);
    cam3.aspect = w / h;
    cam3.updateProjectionMatrix();
    ctl.setViewport(FOV, cam3.aspect);
    safe = safeRectFromPanels(w, h, panels);
    refollow();
    dirty = true;
  };
  resize();
  raf = requestAnimationFrame(frame);

  const scene_: ForestScene = {
    camera: ctl,
    setState(s) {
      const prev = state;
      const { state: ss, changed } = core.intake(s, performance.now());
      if (ss === prev) return; // unchanged, or queued while paused
      applyState(ss, changed);
    },
    shownAbs: () => core.shownAbs(),
    jumpAbs(abs) {
      core.jumpAbs(abs);
      dirty = true;
    },
    pick(x, y) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1), cam3);
      const hits = [layer.pick(ray), ground.pick(ray), quiet.pick(ray)].filter((h) => h !== null);
      const r = hits.sort((p, q) => p.distance - q.distance)[0]?.result ?? null;
      const o = core.origin ?? 0;
      if (!r) return null;
      if (r.kind === "ghost") return { ...r, plantedAt: r.plantedAt + o, movedAt: r.movedAt + o };
      if (r.kind === "quiet") return { ...r, time: r.time + o, to: r.to + o };
      return { ...r, time: r.time + o };
    },
    highlight(id) {
      layer.highlight(id);
      dirty = true;
    },
    resize,
    stats() {
      return { ...layer.stats(), flora: ground.count(), frameMs, pixelRatio: renderer.getPixelRatio(), paused, drawCalls: drawn.calls, triangles: drawn.triangles };
    },
    pause() {
      if (paused || !alive) return;
      paused = true;
      gate.release(); // the card scrolled away: the page's wheel again
      pausedAt = performance.now();
      core.pause();
      cancelAnimationFrame(raf);
      raf = 0;
    },
    resume() {
      if (!paused || !alive) return;
      paused = false;
      const r = core.resume(performance.now());
      if (r && r.state !== state) applyState(r.state, r.changed);
      lastT = null; // the paused stretch is not a frame interval
      renderedPrev = false;
      dirty = true;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(frame);
    },
    async snapshot() {
      if (!alive || lost) throw new Error("forest scene unavailable");
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([layer.settle(), new Promise<void>((res) => (timer = setTimeout(res, SNAPSHOT_WAIT_MS)))]);
      clearTimeout(timer);
      if (!alive || lost) throw new Error("forest scene unavailable");
      // one task from here to the read: with preserveDrawingBuffer off the buffer is only valid until it yields
      const t = paused ? pausedAt : performance.now();
      layer.animate(t, 0, rel); // swaps in results whose cut is reached (the loop may be stopped); dt 0
      if (camInputsDirty) (camInputsDirty = false), syncCamera(t);
      shadowDirty = true;
      render();
      renderedPrev = false; // an extra render: not a frame interval
      return renderer.domElement.toDataURL("image/png");
    },
    exportView() {
      if (!timeline || !primed) return null;
      const p = cam3.position, q = controls.target, front = timeline.X(rel);
      return { pos: { x: p.x - front, y: p.y, z: p.z }, target: { x: q.x - front, y: q.y, z: q.z }, mode: ctl.cameraMode, frontRel: true };
    },
    importView(v) {
      if (v.frontRel !== true) return; // the earlier world-space format: meaningless in another layout
      if (!timeline || !primed) return void (pendingView = v); // the front is known with the first data
      const w = fromFrontRel(v, timeline.X(rel));
      if (!ctl.importView(w.pos, w.target, v.mode)) return; // ignored during a live drag
      if (ctl.cameraMode === "follow") (refollow(), ctl.settleToFit());
      applyPose();
      dirty = true;
    },
    setCameraMode(mode) {
      ctl.setMode(mode); // releases a focused session (final review I3)
      if (ctl.paused) ctl.resume();
    },
    resetView() {
      dragging = false;
      controls.maxPolarAngle = HORIZON;
      ctl.resetView();
      dirty = true;
    },
    setPanels(rects) {
      panels = rects;
      safe = safeRectFromPanels(Math.max(1, canvas.clientWidth), Math.max(1, canvas.clientHeight), panels);
      refollow();
    },
    debug: {
      girthAt: (id) => layer.girthAt(id, rel),
      heightAt: (id) => layer.heightAt(id, rel),
      cellsAt: (id) => layer.cellsAt(id),
      levelOf: (id) => layer.levelOf(id),
      levelCubes: () => layer.levelCubes(),
      holdPose: (pos, target) => {
        const now = performance.now();
        ctl.setDragging(true, now);
        ctl.setPose(new THREE.Vector3(pos.x, pos.y, pos.z), new THREE.Vector3(target.x, target.y, target.z));
        applyPose();
        ctl.setDragging(false, now);
        dirty = true;
      },
      rel: () => rel,
      flights: () => layer.flights(),
      cameraPos: () => ({ x: cam3.position.x, y: cam3.position.y, z: cam3.position.z }),
      cameraTarget: () => ({ x: controls.target.x, y: controls.target.y, z: controls.target.z }),
      wheelEngaged: () => gate.engaged,
      groundAt,
      followGoal: () => ctl.followGoal(),
      treePx: () => {
        const out: number[] = [], h = canvas.clientHeight, a = new THREE.Vector3(), b = new THREE.Vector3();
        for (const [id, info] of infos) {
          const h0 = layer.heightAt(id, rel);
          if (h0 === null || !info.places.length) continue;
          const top = h0 * layer.scaleOf(id); // as drawn (bigger when far)
          const p = info.places[placeIdx(info.places, rel)];
          a.set(p.x, 0, p.z).project(cam3);
          b.set(p.x, top, p.z).project(cam3);
          if (a.z > 1 || b.z > 1) continue;
          out.push(((b.y - a.y) / 2) * h);
        }
        return out;
      },
      pickWorld: (x: number, y: number, z: number) => {
        const v = new THREE.Vector3(x, y, z).project(cam3), rect = canvas.getBoundingClientRect();
        return scene_.pick(rect.left + ((v.x + 1) / 2) * rect.width, rect.top + ((1 - v.y) / 2) * rect.height);
      },
      treeDist: () => {
        const out: { id: string; d: number; top: number; scale: number; x: number; z: number; r: number }[] = [];
        for (const [id, info] of infos) {
          const top = layer.heightAt(id, rel);
          if (top === null || !info.places.length) continue;
          const p = info.places[placeIdx(info.places, rel)];
          out.push({ id, d: cam3.position.distanceTo(new THREE.Vector3(p.x, 0, p.z)), top, scale: layer.scaleOf(id), x: p.x, z: p.z, r: layer.crownR(id) ?? 0 });
        }
        return out;
      },
      spanLength: () => (timeline && state ? timeline.X(state.now) - spanX0() : 0),
    },
    dispose() {
      alive = false;
      cancelAnimationFrame(raf);
      controls.removeEventListener("start", onStart);
      controls.removeEventListener("change", onChange);
      controls.removeEventListener("end", onEnd);
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("gesturestart", onGestureStart);
      canvas.removeEventListener("gesturechange", onGestureChange);
      canvas.removeEventListener("gestureend", onGestureEnd);
      document.removeEventListener("gesturestart", onDocGesture);
      document.removeEventListener("wheel", onDocPinch);
      document.removeEventListener("pointerdown", onDocPointerDown, true);
      document.removeEventListener("keydown", onDocKey);
      window.removeEventListener("blur", onWinBlur);
      document.removeEventListener("visibilitychange", onVisible);
      controls.dispose();
      layer.dispose();
      ground.dispose();
      trail.dispose();
      quiet.dispose();
      land.dispose();
      sky.dispose();
      mats.dispose();
      output.dispose();
      saturate.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
  return scene_;
}
