import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { type Browser, type Page, type TestInfo, expect, test } from '@playwright/test';
import { type SeedResult, farmSeeded, historyRows, resumeSession, seedFarmWeek, startGrowing } from './fixtures/forest-seed';

// The session forest (/ui/forest), end to end, against the local stack of playwright.forest.config.ts
// (fixtures/forest-stack.ts: uvicorn + the UI + a one-origin proxy, request_history seeded by fixtures/forest-seed.ts).
//
//   cd frontend && npx playwright test -c playwright.forest.config.ts
//
// Live view only (spec §10.6): the last 48 h, 30 s behind now, cinematic camera by default. The motion tests sample
// the default cinematic live view.
//
// Frame time (ruling R27): the p99 ≤ 25 ms bound is a manual release check. It runs only with FOREST_PERF=1, on an
// unloaded target machine (RTX A4000 class, the plan's Global Constraints); otherwise it is skipped. A shared, loaded
// dev box gives no valid verdict. Even with FOREST_PERF=1 it is test.fixme wherever WebGL is a software rasterizer
// (SwiftShader, llvmpipe), whose frame times are meaningless. The "gpu" project runs the motion tests headed in the
// system Chrome. Every motion run records p99, the camera and girth steps, the pixel ratio and the load average in a
// test annotation ("motion"), whether or not the bound runs.
// The camera, girth and flight bounds do not depend on the GPU or the load and always hold, headless included.

const BASE = process.env.FOREST_E2E_URL ?? '';
const ADMIN_PW = process.env.FOREST_E2E_ADMIN_PW ?? '';
const DB = process.env.FOREST_E2E_DB ?? '';
const SEED: SeedResult = JSON.parse(process.env.FOREST_E2E_SEED ?? '{}');

test.use({ baseURL: BASE });

type ForestStats = { trees: number; cubes: number; flora: number; frameMs: number; pixelRatio?: number; paused?: boolean; levels?: number[]; drawCalls?: number; triangles?: number };
type ForestHooks = {
  stats(): ForestStats;
  cameraPos(): { x: number; y: number; z: number };
  groundAt(x: number, z: number): number;
  cameraLook(): { x: number; y: number; z: number } | null;
  girthAt(id: string): number | null;
  heightAt(id: string): number | null;
  cellsAt(id: string): number | null;
  levelOf(id: string): number | null;
  levelCubes(): number[];
  holdPose(pos: V, target: V): void;
  treePx(): number[];
  spanLength(): number;
  pickWorld(x: number, y: number, z: number): { kind: string; treeId?: string } | null;
  treeDist(): { id: string; d: number; top: number; scale: number; x: number; z: number; r: number }[];
  flights(): { treeId: string; fromX: number; toX: number }[];
  treeIds(): string[];
  shownAbs(): number;
  snapshot(): Promise<string | null>;
  exportView(): View | null;
  followGoal(): { x: number; y: number; z: number } | null;
  cameraTarget(): { x: number; y: number; z: number } | null;
  userHold(): boolean;
  wheelEngaged(): boolean;
};
/** The hand-off view: x relative to the scene's front (X of the shown time), y and z as they are. */
type View = { pos: V; target: V; mode: 'cinema' | 'follow'; frontRel: true };
/** Worker traffic: job posts, replies, reply times, and per reply the tree (job key) it built. */
type WorkerLog = { posts: number; replies: number; replyAt: number[]; built: { key: string; at: number }[] };
type W = Window & { __forest?: ForestHooks; __e2eWorkers?: WorkerLog };

/** Counts the geometry jobs sent to (and answered by) the scene's workers. Test-side only: wraps window.Worker. */
const COUNT_WORKERS = () => {
  const w = window as unknown as W;
  w.__e2eWorkers = { posts: 0, replies: 0, replyAt: [], built: [] };
  const keys = new Map<number, string>(); // job id → job key (tree id)
  const Base = window.Worker;
  if (!Base) return;
  const post = Base.prototype.postMessage;
  Base.prototype.postMessage = function (this: Worker, ...args: unknown[]) {
    w.__e2eWorkers!.posts++;
    const job = args[0] as { id?: unknown; key?: unknown } | null;
    if (job && typeof job.id === 'number' && typeof job.key === 'string') keys.set(job.id, job.key);
    return (post as (...a: unknown[]) => void).apply(this, args);
  } as Worker['postMessage'];
  class Counted extends Base {
    constructor(url: string | URL, opts?: WorkerOptions) {
      super(url, opts);
      this.addEventListener('message', (e: MessageEvent<{ id?: unknown }>) => {
        const log = w.__e2eWorkers!, at = performance.now();
        log.replies++;
        log.replyAt.push(at);
        const key = typeof e.data?.id === 'number' ? keys.get(e.data.id) : undefined;
        if (key !== undefined) log.built.push({ key, at });
      });
    }
  }
  window.Worker = Counted;
};

async function loginAdmin(page: Page) {
  await page.goto('/ui/login');
  await page.fill('input[name=username]', 'admin');
  await page.fill('input[name=password]', ADMIN_PW);
  await page.click('button:has-text("Log in")');
  await expect(page).toHaveURL(/\/ui\/stats/);
}

/** Opens the forest and waits until the scene shows trees. */
async function openForest(page: Page, query = '?debug=1') {
  await page.goto(`/ui/forest${query}`);
  await expect(page.getByTestId('forest-view')).toBeVisible();
  await expect(page.locator('[data-testid=forest-canvas-host] canvas')).toHaveCount(1);
  await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
}

/** The WebGL renderer string; software rasterizers make frame times meaningless. */
async function webglRenderer(page: Page): Promise<{ renderer: string; software: boolean }> {
  const renderer = await page.evaluate(() => {
    const gl = document.createElement('canvas').getContext('webgl2') ?? document.createElement('canvas').getContext('webgl');
    if (!gl) return 'none';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  });
  return { renderer, software: /swiftshader|llvmpipe|software|none/i.test(renderer) };
}

/** Raw GET through the proxy: the bytes on the wire with and without gzip. */
function rawGet(url: string, headers: Record<string, string>): Promise<{ status: number; bytes: number; encoding: string }> {
  return new Promise((resolve, reject) => {
    http
      .get(url, { headers }, (res) => {
        let bytes = 0;
        res.on('data', (c: Buffer) => (bytes += c.length));
        res.on('end', () => resolve({ status: res.statusCode ?? 0, bytes, encoding: String(res.headers['content-encoding'] ?? 'identity') }));
      })
      .on('error', reject);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Camera ground clamp and Reset view (follow-ups A and B)
// ---------------------------------------------------------------------------------------------------------------

type GroundProbe = { frames: number; minClear: number; at: V | null };
type WG = W & { __ground?: GroundProbe & { stop: boolean } };
/** Every animation frame from now on: the camera's height over the ground under it (minimum, and where). */
async function startGroundProbe(page: Page) {
  await page.evaluate(() => {
    const w = window as WG;
    const probe = { frames: 0, minClear: Infinity, at: null as V | null, stop: false };
    w.__ground = probe;
    const tick = () => {
      if (probe.stop) return;
      const f = w.__forest;
      if (f) {
        const p = f.cameraPos(), c = p.y - f.groundAt(p.x, p.z);
        probe.frames++;
        if (c < probe.minClear) (probe.minClear = c), (probe.at = p);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}
async function stopGroundProbe(page: Page): Promise<GroundProbe> {
  return page.evaluate(() => {
    const g = (window as WG).__ground!;
    g.stop = true;
    return { frames: g.frames, minClear: g.minClear, at: g.at };
  });
}

/** Scripted user input on a canvas: orbits toward and past the horizon, wheel zooms in and out, a right-button pan. */
async function tortureCamera(page: Page, canvas: { x: number; y: number; width: number; height: number }) {
  const cx = canvas.x + canvas.width / 2, cy = canvas.y + canvas.height / 2;
  const drag = async (dx: number, dy: number, button: 'left' | 'right' = 'left') => {
    await page.mouse.move(cx, cy);
    await page.mouse.down({ button });
    for (let i = 1; i <= 12; i++) await page.mouse.move(cx + (dx * i) / 12, cy + (dy * i) / 12);
    await page.mouse.up({ button });
  };
  await drag(0, -Math.min(400, canvas.height / 2 - 4)); // mouse up: the camera sinks toward the horizon
  await drag(-Math.min(300, canvas.width / 2 - 4), -60); // orbit round, still low
  await page.mouse.move(cx, cy);
  for (let i = 0; i < 8; i++) await page.mouse.wheel(0, -400); // zoom in hard
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, 600); // and out, low
  await drag(0, Math.min(200, canvas.height / 2 - 4), 'right'); // pan
  await drag(0, -Math.min(400, canvas.height / 2 - 4)); // low again
  for (let i = 0; i < 6; i++) await page.mouse.wheel(0, -500);
}

/** Clicks Reset view and samples, every frame, the camera's step and its distance to the follow goal, until it has
 * been within 0.5 of the goal or 6 s pass. */
async function resetAndTime(page: Page, button: ReturnType<Page['getByRole']>) {
  await page.evaluate(() => {
    const w = window as W & { __reset?: Promise<unknown>; __resetArm?: () => void };
    w.__reset = new Promise((resolve) => {
      w.__resetArm = () => {
        const f = w.__forest!, t0 = performance.now();
        const hyp = (a: V, b: V) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        const from = f.cameraPos(), g0 = f.followGoal();
        let prev = from, maxStep = 0, frames = 0;
        const tick = () => {
          const p = f.cameraPos(), g = f.followGoal();
          frames++;
          maxStep = Math.max(maxStep, hyp(p, prev));
          prev = p;
          const d = g ? hyp(p, g) : Infinity;
          if (d < 0.5) return resolve({ backMs: performance.now() - t0, maxStep, frames, d0: g0 ? hyp(from, g0) : null });
          if (performance.now() - t0 > 6000) return resolve({ backMs: null, maxStep, frames, d, d0: g0 ? hyp(from, g0) : null });
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      };
    });
    window.addEventListener('click', () => w.__resetArm!(), { capture: true, once: true });
  });
  await button.click();
  return page.evaluate(() => (window as W & { __reset?: Promise<unknown> }).__reset) as Promise<{
    backMs: number | null; maxStep: number; frames: number; d0: number | null; d?: number;
  }>;
}

// ---------------------------------------------------------------------------------------------------------------
// Range switch (follow-up C)
// ---------------------------------------------------------------------------------------------------------------

type SwitchSample = {
  /** ms from the click to the follow goal's first move (the new range's data reframed the fit). */
  goalMovedMs: number | null;
  /** ms from that move until the camera is first within 0.5 of the goal (the goal as it is at that frame). */
  arrivedMs: number | null;
  /** How far the goal moved after the glide's deadline (goal move + 3 s): trees still landing, followed in steady motion. */
  lateGoalMove: number;
  /** Camera distance to the new goal at the move; the largest per-frame camera step; frames sampled. */
  d0: number | null; maxStep: number; frames: number;
  /** Camera distance to its look target before the click and at the end: a wider range frames from further out. */
  zoomBefore: number; zoomAfter: number;
  /** Trees built before the click: the lowest height each later showed, over its height at the click. */
  heightRatioMin: number; heightTrees: number;
};

/** Arms a sampler on the next click (capture phase): follow goal, camera steps and the built trees' heights per frame,
 * for `ms`. */
async function armSwitchSampler(page: Page, ms = 8000) {
  await page.evaluate((ms) => {
    const w = window as W & { __switch?: Promise<SwitchSample> };
    w.__switch = new Promise((resolve) => {
      window.addEventListener('click', () => {
        const f = w.__forest!, t0 = performance.now();
        const hyp = (a: V, b: V) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        const zoom = () => { const p = f.cameraPos(), l = f.cameraLook(); return l ? Math.hypot(l.x, l.y, l.z) : hyp(p, p); };
        const zoomBefore = zoom();
        const h0 = new Map<string, number>();
        for (const id of f.treeIds()) { const h = f.heightAt(id); if (h !== null && h > 0.5) h0.set(id, h); }
        const minRatio = new Map<string, number>();
        const g0 = f.followGoal();
        let prev = f.cameraPos(), maxStep = 0, frames = 0, movedAt: number | null = null, d0: number | null = null;
        let arrivedAt: number | null = null, atDeadline: V | null = null, lateGoalMove = 0;
        const tick = () => {
          const now = performance.now(), p = f.cameraPos(), g = f.followGoal();
          frames++;
          maxStep = Math.max(maxStep, hyp(p, prev));
          prev = p;
          for (const [id, h] of h0) { const x = f.heightAt(id); if (x !== null) minRatio.set(id, Math.min(minRatio.get(id) ?? Infinity, x / h)); }
          if (movedAt === null && g && g0 && hyp(g, g0) > 2) (movedAt = now - t0), (d0 = hyp(p, g));
          if (movedAt !== null && g) {
            if (hyp(p, g) < 0.5) arrivedAt ??= now - t0 - movedAt;
            if (now - t0 - movedAt >= 3000) {
              atDeadline ??= g;
              lateGoalMove = Math.max(lateGoalMove, hyp(g, atDeadline));
            }
          }
          if (now - t0 < ms) return void requestAnimationFrame(tick);
          resolve({
            goalMovedMs: movedAt, arrivedMs: arrivedAt, lateGoalMove, d0, maxStep, frames, zoomBefore, zoomAfter: zoom(),
            heightRatioMin: Math.min(1, ...minRatio.values()), heightTrees: minRatio.size,
          });
        };
        requestAnimationFrame(tick);
      }, { capture: true, once: true });
    });
  }, ms);
}
const switchResult = (page: Page) => page.evaluate(() => (window as W & { __switch?: Promise<SwitchSample> }).__switch!) as Promise<SwitchSample>;

/** The forest polls of `range` a page makes from now on (full fetches and deltas apart). */
function recordRangePolls(page: Page, range: string) {
  const full: string[] = [], delta: string[] = [], other: string[] = [];
  page.on('request', (r) => {
    const u = r.url();
    if (!u.includes('/api/stats/forest?')) return;
    if (!u.includes(`range=${range}`)) other.push(u);
    else (u.includes('since=') ? delta : full).push(u);
  });
  return { full, delta, other };
}

// ---------------------------------------------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------------------------------------------

test.describe('admin', () => {
  test('camera never underground after scripted drags and zooms; Reset view returns in ≤ 3 s', async ({ page }, testInfo) => {
    await loginAdmin(page);
    await openForest(page);
    const canvas = (await page.locator('[data-testid=forest-canvas-host] canvas').boundingBox())!;
    await startGroundProbe(page);
    await tortureCamera(page, canvas); // cinematic (the default)
    await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
    await page.waitForTimeout(3000);
    await tortureCamera(page, canvas); // wide
    await page.waitForTimeout(500);
    const g = await stopGroundProbe(page);
    const r = await resetAndTime(page, page.getByRole('button', { name: 'Reset view' }));
    testInfo.annotations.push({ type: 'ground', description: JSON.stringify({ ...g, reset: r }) });
    console.log(`full window: ground ${JSON.stringify(g)}; reset ${JSON.stringify(r)}`);
    expect(g.frames).toBeGreaterThan(30);
    expect(g.minClear).toBeGreaterThanOrEqual(0.5 - 1e-6);
    expect(r.backMs, 'Reset view returns to the fit').not.toBeNull();
    expect(r.backMs!).toBeLessThanOrEqual(3000 + 250); // the 2.4 s glide; slack for a slow headless frame
    // no jump: no frame step larger than a 30th of the way back (or the steady 0.1)
    expect(r.maxStep).toBeLessThanOrEqual(Math.max(0.1, (r.d0 ?? 0) / 30) + 1e-6);
    // the mode is kept
    await expect(page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' })).toHaveAttribute('aria-pressed', 'true');
  });

  test('range switch: 24h → 7d refetches once, reframes by a glide, regrows nothing, and is remembered', async ({ page }, testInfo) => {
    await loginAdmin(page);
    await page.evaluate(() => localStorage.removeItem('forest.range'));
    await openForest(page);
    const group = page.getByRole('group', { name: 'Range' });
    await expect(group.getByRole('button', { name: '24h' })).toHaveAttribute('aria-pressed', 'true');
    await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
    await page.waitForTimeout(4000); // the Wide fit of 24 h settles
    const polls = recordRangePolls(page, '7d');
    await armSwitchSampler(page);
    await group.getByRole('button', { name: '7d' }).click();
    const r = await switchResult(page);
    testInfo.annotations.push({ type: 'range', description: JSON.stringify({ ...r, polls: { full: polls.full.length, delta: polls.delta.length } }) });
    console.log(`full window 24h → 7d: ${JSON.stringify(r)}; polls ${polls.full.length} full, ${polls.delta.length} delta, ${polls.other.length} other`);
    expect(polls.full.length, 'one full fetch of 7d').toBe(1);
    expect(polls.delta.length, 'then deltas').toBeGreaterThan(0);
    expect(polls.other.length, 'no poll of another range after the switch').toBe(0);
    expect(r.goalMovedMs, 'the fit moved to the 7 d span').not.toBeNull();
    expect(r.arrivedMs, 'the camera reached the new fit').not.toBeNull();
    expect(r.arrivedMs!).toBeLessThanOrEqual(3000 + 250);
    expect(r.maxStep).toBeLessThanOrEqual(Math.max(0.1, (r.d0 ?? 0) / 30) + 1e-6);
    expect(r.zoomAfter, 'a wider range is framed from further out').toBeGreaterThan(r.zoomBefore);
    expect(r.heightTrees).toBeGreaterThan(0);
    expect(r.heightRatioMin, 'no tree shown before the switch regrew').toBeGreaterThanOrEqual(0.99);
    expect(await page.evaluate(() => localStorage.getItem('forest.range'))).toBe('7d');
    await openForest(page);
    await expect(page.getByRole('group', { name: 'Range' }).getByRole('button', { name: '7d' })).toHaveAttribute('aria-pressed', 'true');
    await page.evaluate(() => localStorage.removeItem('forest.range'));
  });

  test('forest page: canvas, no console errors, panels, camera toggle, Esc', async ({ page }) => {
    await loginAdmin(page);
    const errors: string[] = [];
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    page.on('pageerror', (e) => errors.push(String(e)));

    await openForest(page);
    // the admin sees both keys' trees
    const ids = await page.evaluate(() => (window as W).__forest!.treeIds());
    expect(ids.some((id) => id.startsWith(`${SEED.keyA.id}:`))).toBe(true);
    expect(ids.some((id) => id.startsWith(`${SEED.keyB.id}:`))).toBe(true);

    // description and legend: open by default on a wide window, collapse, and stay collapsed after a reload
    const desc = page.getByTestId('forest-desc');
    const legend = page.getByTestId('forest-legend');
    await expect(desc.locator('summary')).toContainText('Session forest');
    await expect(legend.locator('summary')).toContainText('Legend');
    await expect(desc).toHaveAttribute('open', '');
    await expect(legend).toHaveAttribute('open', '');
    await desc.locator('summary').click();
    await legend.locator('summary').click();
    await expect(desc).not.toHaveAttribute('open', /.*/);
    await expect(legend).not.toHaveAttribute('open', /.*/);
    await page.reload();
    await expect(page.getByTestId('forest-view')).toBeVisible();
    await expect(page.getByTestId('forest-desc')).not.toHaveAttribute('open', /.*/);
    await expect(page.getByTestId('forest-legend')).not.toHaveAttribute('open', /.*/);
    await page.getByTestId('forest-desc').locator('summary').click();
    await expect(page.getByTestId('forest-desc')).toHaveAttribute('open', '');
    await page.reload();
    await expect(page.getByTestId('forest-desc')).toHaveAttribute('open', '');
    await expect(page.getByTestId('forest-legend')).not.toHaveAttribute('open', /.*/);
    await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });

    // the token bars and in-flight panels
    await expect(page.getByTestId('forest-tokens')).toBeVisible();
    await expect(page.getByTestId('forest-tokens').locator('summary')).toContainText('Tokens per minute');
    await expect(page.getByTestId('forest-inflight')).toBeVisible();
    await expect(page.getByTestId('forest-inflight').locator('summary')).toContainText('Requests in flight');
    await expect(page.getByTestId('forest-inflight').locator('summary')).toContainText(/\d+ now/);

    // the camera toggle: cinematic by default, then wide, then back
    const cam = page.getByRole('group', { name: 'Camera' });
    const cinema = cam.getByRole('button', { name: 'Cinematic' });
    const wide = cam.getByRole('button', { name: 'Wide' });
    await expect(cinema).toHaveAttribute('aria-pressed', 'true');
    await expect(wide).toHaveAttribute('aria-pressed', 'false');
    await wide.click();
    await expect(wide).toHaveAttribute('aria-pressed', 'true');
    await expect(cinema).toHaveAttribute('aria-pressed', 'false');
    await cinema.click();
    await expect(cinema).toHaveAttribute('aria-pressed', 'true');

    // no admin chrome over the full window
    await expect(page.getByRole('button', { name: /open menu/i })).toHaveCount(0);

    expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);

    // Esc goes back to the stats page
    await page.keyboard.press('Escape');
    await expect(page).toHaveURL(/\/ui\/stats$/);
  });

  test('perf: /api/stats/forest payload for the seeded data', async ({ page }) => {
    const login = await page.request.post('/api/auth/login', {
      data: { username: 'admin', password: ADMIN_PW },
      headers: { Origin: BASE },
    });
    expect(login.ok()).toBe(true);
    const { access_token } = await login.json();
    const url = `${BASE}/api/stats/forest?range=48h`;
    const auth = { Authorization: `Bearer ${access_token}` };
    const plain = await rawGet(url, { ...auth, 'Accept-Encoding': 'identity' });
    const gz = await rawGet(url, { ...auth, 'Accept-Encoding': 'gzip' });
    expect(plain.status).toBe(200);
    expect(gz.encoding).toBe('gzip');
    const res = await page.request.get(url, { headers: auth });
    const body = await res.json();
    const payload = {
      trees: body.trees.length,
      flowers: body.flowers.length,
      bytes: plain.bytes,
      gzipBytes: gz.bytes,
    };
    console.log(`forest payload (admin, 48h): ${JSON.stringify(payload)}`);
    test.info().annotations.push({ type: 'payload', description: JSON.stringify(payload) });
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Key holder
// ---------------------------------------------------------------------------------------------------------------

test.describe('key holder', () => {
  test('signs in with a key, sees only that key, Esc keeps the session, Sign out', async ({ browser }) => {
    // a fresh context: no admin session anywhere
    const ctx = await browser.newContext({ baseURL: BASE });
    const page = await ctx.newPage();
    const errors: string[] = [];
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    try {
      await page.goto('/ui/forest/login');
      await expect(page.getByRole('heading', { name: 'Your session forest' })).toBeVisible();
      await page.fill('input[name=forest-api-key]', SEED.keyB.plaintext);
      await page.getByRole('button', { name: 'Open forest' }).click();
      await expect(page).toHaveURL(/\/ui\/forest\?key=1$/);
      await expect(page.getByTestId('forest-view')).toBeVisible();

      // no admin nav, and the key holder's own exit instead of ✕
      await expect(page.locator('header nav')).toHaveCount(0);
      await expect(page.getByRole('button', { name: /open menu/i })).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Close the full window' })).toHaveCount(0);

      // only key B's trees: the debug hooks need ?debug=1 (the token stays in this tab's sessionStorage)
      await page.goto('/ui/forest?key=1&debug=1');
      await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
      // treeIds: every tree in the data the view holds; stats().trees: the ones the scene has built so far.
      // The default range is 24 h: key B's tree from 30 h ago comes in with the key holder's own range switch (7d).
      expect(await page.evaluate(() => (window as W).__forest!.treeIds().length)).toBe(1);
      await page.getByRole('group', { name: 'Range' }).getByRole('button', { name: '7d' }).click();
      await page.waitForFunction(() => (window as W).__forest!.treeIds().length === 2, null, { timeout: 15_000 });
      const ids = await page.evaluate(() => (window as W).__forest!.treeIds());
      expect(ids.length).toBe(2);
      for (const id of ids) expect(id.startsWith(`${SEED.keyB.id}:`), id).toBe(true);
      const stats = await page.evaluate(() => (window as W).__forest!.stats());
      expect(stats.trees).toBeGreaterThan(0);
      expect(stats.trees).toBeLessThanOrEqual(2);

      // Esc keeps the key holder signed in (no admin page to go back to)
      await page.keyboard.press('Escape');
      await page.waitForTimeout(1000);
      await expect(page).toHaveURL(/\/ui\/forest\?/);
      await expect(page.getByTestId('forest-view')).toBeVisible();
      expect(await page.evaluate(() => sessionStorage.getItem('vw-forest-token'))).toBeTruthy();

      // Sign out: back to the key holder's login, token gone
      await page.getByRole('button', { name: 'Sign out' }).click();
      await expect(page).toHaveURL(/\/ui\/forest\/login$/);
      expect(await page.evaluate(() => sessionStorage.getItem('vw-forest-token'))).toBeNull();
      expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([]);
    } finally {
      await ctx.close();
    }
  });
});

// ---------------------------------------------------------------------------------------------------------------
// The Stats page card (Plan 3): bundle isolation, offscreen pause, hand-off, phone still, nav
// ---------------------------------------------------------------------------------------------------------------

type V = { x: number; y: number; z: number };
const dist = (a: V, b: V) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const isForestPoll = (url: string) => url.includes('/api/stats/forest?range=6h');

/** Records the forest polls (`/api/stats/forest?range=6h`) a page makes, with their wall time. */
function recordPolls(page: Page): number[] {
  const at: number[] = [];
  page.on('request', (r) => isForestPoll(r.url()) && at.push(Date.now()));
  return at;
}

/** The card's top edge is below the first screen: nothing has intersected it yet. */
async function expectCardBelowFold(page: Page) {
  const card = page.getByTestId('forest-card');
  await expect(card).toHaveCount(1);
  const box = await card.boundingBox();
  const vh = page.viewportSize()!.height;
  expect(box, 'the card has a layout box').not.toBeNull();
  expect(box!.y, `card top ${box!.y} px vs viewport ${vh} px`).toBeGreaterThan(vh);
  await expect(page.getByTestId('forest-card-placeholder')).toHaveCount(1);
}

/** Scrolls the card into view and waits until its scene has a canvas and shows trees. */
async function scrollCardIn(page: Page) {
  await page.getByTestId('forest-card').scrollIntoViewIfNeeded();
  await expect(page.locator('[data-testid=forest-card] [data-testid=forest-canvas-host] canvas')).toHaveCount(1, { timeout: 60_000 });
  await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
}

test.describe('stats card', () => {
  // the card follows the Stats page's range (vw.stats.range, default 1 h); these tests were written for 6 h: pin it
  // once per tab (a switch the test makes afterwards persists as usual)
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        if (!sessionStorage.getItem('e2e.range.pinned')) {
          localStorage.setItem('vw.stats.range', '6h');
          sessionStorage.setItem('e2e.range.pinned', '1');
        }
      } catch {
        /* no storage */
      }
    });
  });

  test("range switch on /stats: the card fetches the page's range, refetches once and reframes by a glide", async ({ page }, testInfo) => {
    await loginAdmin(page);
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    const card = page.getByTestId('forest-card');
    await expect(card.getByText('6 h · live · 30 s delay')).toBeVisible();
    await page.waitForTimeout(3000); // the 6 h fit settles
    const polls = recordRangePolls(page, '24h');
    await armSwitchSampler(page);
    // clicked in place (no scroll to the selector at the top: the card stays in view and live)
    await page.getByTestId('range-selector').getByRole('button', { name: '24h' }).evaluate((b: HTMLElement) => b.click());
    const r = await switchResult(page);
    testInfo.annotations.push({ type: 'range', description: JSON.stringify({ ...r, polls: { full: polls.full.length, delta: polls.delta.length } }) });
    console.log(`card 6h → 24h: ${JSON.stringify(r)}; polls ${polls.full.length} full, ${polls.delta.length} delta, ${polls.other.length} other`);
    await expect(card.getByText('24 h · live · 30 s delay')).toBeVisible();
    expect(polls.full.length, 'one full fetch of 24h').toBe(1);
    expect(polls.delta.length, 'then deltas').toBeGreaterThan(0);
    expect(polls.other.filter((u) => u.includes('range=6h')), 'no 6h poll after the switch').toEqual([]);
    expect(r.goalMovedMs, 'the fit moved to the 24 h span').not.toBeNull();
    expect(r.arrivedMs, 'the camera reached the new fit').not.toBeNull();
    expect(r.arrivedMs!).toBeLessThanOrEqual(3000 + 250);
    expect(r.maxStep).toBeLessThanOrEqual(Math.max(0.1, (r.d0 ?? 0) / 30) + 1e-6);
    expect(r.zoomAfter, 'a wider range is framed from further out').toBeGreaterThan(r.zoomBefore);
    expect(r.heightTrees).toBeGreaterThan(0);
    expect(r.heightRatioMin, 'no tree shown before the switch regrew').toBeGreaterThanOrEqual(0.99);
    // the card's budget for 24 h is 30 trees
    expect(await page.evaluate(() => (window as W).__forest!.stats().trees)).toBeLessThanOrEqual(30);

    // and on to 7 d (budget 30, the land reaching back): the same rules, and its frame time recorded (review)
    await page.waitForTimeout(2000);
    const polls7 = recordRangePolls(page, '7d');
    await armSwitchSampler(page);
    await page.getByTestId('range-selector').getByRole('button', { name: '7d' }).evaluate((b: HTMLElement) => b.click());
    const r7 = await switchResult(page);
    const st7 = await page.evaluate(() => (window as W).__forest!.stats());
    testInfo.annotations.push({ type: 'range7d', description: JSON.stringify({ ...r7, stats: st7 }) });
    console.log(`card 24h → 7d: ${JSON.stringify(r7)}; stats ${JSON.stringify(st7)}; polls ${polls7.full.length} full, ${polls7.delta.length} delta`);
    await expect(card.getByText('7 d · live · 30 s delay')).toBeVisible();
    expect(polls7.full.length, 'one full fetch of 7d').toBe(1);
    expect(r7.heightRatioMin, 'no tree regrew at 24h → 7d').toBeGreaterThanOrEqual(0.99);
    if (r7.goalMovedMs !== null) {
      expect(r7.arrivedMs, 'the camera reached the 7 d fit').not.toBeNull();
      expect(r7.arrivedMs!).toBeLessThanOrEqual(3000 + 250);
      expect(r7.maxStep).toBeLessThanOrEqual(Math.max(0.1, (r7.d0 ?? 0) / 30) + 1e-6);
    }
    // the card's budget at 7 d: 30, or FAR_BUILT (200) once the view is far (re-review 3, N3; final review I2)
    expect(st7.trees).toBeLessThanOrEqual(200);

    // and the Stats page's default, 1 h
    const polls1 = recordRangePolls(page, '1h');
    await page.getByTestId('range-selector').getByRole('button', { name: '1h' }).evaluate((b: HTMLElement) => b.click());
    await expect(card.getByText('1 h · live · 30 s delay')).toBeVisible();
    await expect.poll(() => polls1.full.length).toBe(1);
  });

  test('card: never underground after drags and zooms; Reset view returns in ≤ 3 s', async ({ page }, testInfo) => {
    await loginAdmin(page);
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    const canvas = (await page.locator('[data-testid=forest-card] [data-testid=forest-canvas-host] canvas').boundingBox())!;
    await startGroundProbe(page);
    await tortureCamera(page, canvas);
    await page.waitForTimeout(500);
    const g = await stopGroundProbe(page);
    const r = await resetAndTime(page, page.getByTestId('forest-card').getByRole('button', { name: 'Reset view' }));
    testInfo.annotations.push({ type: 'ground', description: JSON.stringify({ ...g, reset: r }) });
    console.log(`card: ground ${JSON.stringify(g)}; reset ${JSON.stringify(r)}`);
    expect(g.frames).toBeGreaterThan(30);
    expect(g.minClear).toBeGreaterThanOrEqual(0.5 - 1e-6);
    expect(r.backMs, 'Reset view returns to the fit').not.toBeNull();
    expect(r.backMs!).toBeLessThanOrEqual(3000 + 250);
    expect(r.maxStep).toBeLessThanOrEqual(Math.max(0.1, (r.d0 ?? 0) / 30) + 1e-6);
  });

  test('no three.js before the card scrolls in; it loads after', async ({ page }) => {
    // every script response from the first page on (login included), split at the scroll
    const before: Promise<string>[] = [], after: Promise<string>[] = [];
    const urls: { before: string[]; after: string[] } = { before: [], after: [] };
    let scrolled = false;
    page.on('response', (r) => {
      const url = r.url();
      if (r.request().resourceType() !== 'script' && !/\.m?js(\?|$)/.test(url)) return;
      const body = r
        .body()
        .then((b) => b.toString('utf8'))
        .catch(() => page.request.get(url).then((x) => x.text()).catch(() => ''));
      (scrolled ? after : before).push(body);
      (scrolled ? urls.after : urls.before).push(url);
    });
    await loginAdmin(page);
    await page.goto('/ui/stats?debug=1');
    await page.waitForLoadState('load');
    await expectCardBelowFold(page);
    await page.waitForTimeout(3000); // idle chunks (prefetch, lazy panels) have their chance to load
    await expect(page.getByTestId('forest-card-placeholder')).toHaveCount(1);
    const pre = await Promise.all(before);
    const preHits = urls.before.filter((_, i) => pre[i].includes('WebGLRenderer'));
    console.log(`stats card: ${pre.length} scripts before the scroll, ${preHits.length} with WebGLRenderer`);
    expect(pre.length, 'scripts recorded before the scroll').toBeGreaterThan(0);
    expect(preHits, 'scripts with WebGLRenderer before the card scrolled in').toEqual([]);

    scrolled = true;
    await scrollCardIn(page);
    const post = await Promise.all(after);
    const postHits = urls.after.filter((_, i) => post[i].includes('WebGLRenderer'));
    console.log(`stats card: ${post.length} scripts after the scroll, with WebGLRenderer: ${postHits.map((u) => u.split('/').pop()).join(', ')}`);
    expect(postHits.length, 'a script with WebGLRenderer after the scroll').toBeGreaterThan(0);
  });

  test('renders and polls in view; offscreen it pauses and stops polling', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await loginAdmin(page);
    const polls = recordPolls(page);
    await page.goto('/ui/stats?debug=1');
    await expectCardBelowFold(page);
    expect(polls.length, 'forest polls before the card scrolled in').toBe(0);

    await scrollCardIn(page);
    expect(await page.evaluate(() => (window as W).__forest!.stats().paused)).toBe(false);
    const inView0 = polls.length;
    await page.waitForTimeout(5000); // POLL_MS is 2 s
    const inView = polls.length - inView0;
    const gaps = polls.slice(1).map((t, i) => t - polls[i]);
    console.log(`stats card: ${polls.length} forest polls up to 5 s in view (${inView} in that 5 s), gaps ${gaps.join(', ')} ms`);
    test.info().annotations.push({ type: 'polls', description: JSON.stringify({ inView5s: inView, gapsMs: gaps }) });
    expect(polls.length, 'forest polls (range=6h) after the scroll').toBeGreaterThan(0);
    expect(inView, 'forest polls during 5 s in view').toBeGreaterThan(0);

    // scroll away: the card is off-screen, paused, and polls nothing
    await page.evaluate(() => window.scrollTo(0, 0));
    const box = await page.getByTestId('forest-card').boundingBox();
    expect(box!.y).toBeGreaterThan(page.viewportSize()!.height);
    await page.waitForFunction(() => (window as W).__forest?.stats().paused === true, null, { timeout: 5_000 });
    await page.waitForTimeout(500); // a poll already in flight at the scroll is not a new one
    const off0 = polls.length;
    await page.waitForTimeout(5000);
    const offPolls = polls.length - off0;
    const pausedStats = await page.evaluate(() => (window as W).__forest!.stats());
    console.log(`stats card: ${offPolls} forest polls in 5 s offscreen; stats ${JSON.stringify(pausedStats)}`);
    expect(offPolls, 'forest polls during 5 s offscreen').toBe(0);
    expect(pausedStats.paused).toBe(true);

    // a snapshot while paused off-screen: a real PNG (the render and the read in one task), and still paused after
    const png = await page.evaluate(() => (window as W).__forest!.snapshot());
    expect(png, 'paused snapshot').toMatch(/^data:image\/png;base64,/);
    const bytes = Buffer.from(png!.slice(png!.indexOf(',') + 1), 'base64');
    console.log(`stats card: paused snapshot ${bytes.length} bytes`);
    test.info().annotations.push({ type: 'pausedSnapshot', description: `${bytes.length} bytes` });
    expect(bytes.subarray(1, 4).toString('latin1')).toBe('PNG');
    expect(bytes.length, 'paused snapshot PNG size').toBeGreaterThan(5 * 1024);
    expect((await page.evaluate(() => (window as W).__forest!.stats())).paused).toBe(true);
    expect(errors, `page errors:\n${errors.join('\n')}`).toEqual([]);
  });

  test('Open full window: /forest continues from the card camera, with no jump', async ({ page }) => {
    await loginAdmin(page);
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    await page.waitForTimeout(1500); // some follow camera time: not the initial placement

    // The hand-off is front-relative (x − X(shown time)): the card's 6 h layout and the full window's 48 h one put the
    // same forest at different world x. Two probes that outlive the client-side route change:
    //   - a capture-phase click listener reads the card's front-relative view from its debug hooks just before the
    //     card's own click handler exports it (the card's follow camera keeps moving between a read here and the click);
    //   - a sampler waits for the full window's scene to have its first data (its front-relative view exists: the
    //     imported view is placed then), records that view, the per-frame camera steps over the next 30 frames, and
    //     the camera's distance to the follow goal until 10 s after the placement.
    await page.evaluate(() => {
      type Probe = { atClick?: View | null; sample?: Promise<unknown> };
      const w = window as W & { __handoff?: Probe };
      const card = w.__forest;
      const probe: Probe = {};
      w.__handoff = probe;
      window.addEventListener('click', () => (probe.atClick ??= card?.exportView() ?? null), { capture: true, once: true });
      probe.sample = new Promise((resolve) => {
        const t0 = performance.now();
        const hyp = (a: V, b: V) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        let first: View | null = null, prev: V | null = null, maxStep = 0, maxStepAll = 0, waited = 0, lastT = 0, placedAt = 0;
        let settledAt: number | null = null, goalDist10: number | null = null, lastTrace = -Infinity;
        const trace: { t: number; d: number | null; goalMoved: number | null }[] = [];
        let prevGoal: V | null = null;
        const steps: number[] = [], dts: number[] = [], trees: number[] = [];
        const tick = (t: number) => {
          const f = w.__forest;
          const v = f && f !== card ? f.exportView() : null;
          if (!v) {
            waited++; // the full window's scene (or its first data) does not exist yet
            if (performance.now() - t0 > 30_000) return resolve({ error: 'no full-window scene with data within 30 s' });
            return void requestAnimationFrame(tick);
          }
          const p = f!.cameraPos();
          if (!first) (first = v), (placedAt = t);
          if (prev) {
            const s = hyp(p, prev);
            maxStepAll = Math.max(maxStepAll, s);
            if (steps.length < 30) {
              steps.push(s);
              dts.push(t - lastT);
              trees.push(f!.stats().trees);
              maxStep = Math.max(maxStep, s);
            }
          }
          const g = f!.followGoal();
          const gd = g ? hyp(p, g) : null;
          if (t - placedAt - lastTrace >= 1000) {
            lastTrace = t - placedAt;
            trace.push({ t: Math.round(lastTrace), d: gd === null ? null : +gd.toFixed(2), goalMoved: g && prevGoal ? +hyp(g, prevGoal).toFixed(2) : null });
            prevGoal = g;
          }
          if (gd !== null && gd < 2) settledAt ??= t - placedAt;
          else if (gd !== null) settledAt = null; // settled means: and stays there
          prev = p;
          lastT = t;
          if (t - placedAt < 10_000) return void requestAnimationFrame(tick);
          goalDist10 = gd;
          resolve({ first, maxStep, maxStepAll, steps, dts, trees, waited, settledAt, goalDist10, trace });
        };
        requestAnimationFrame(tick);
      });
    });

    const beforeClick = await page.evaluate(() => (window as W).__forest!.exportView());
    await page.getByRole('button', { name: 'Open full window' }).click();
    await expect(page).toHaveURL(/\/ui\/forest\?range=6h&debug=1$/);
    await expect(page.getByTestId('forest-view')).toBeVisible();
    type HandoffSample = {
      error?: string; first: View; maxStep: number; maxStepAll: number; steps: number[]; dts: number[]; trees: number[];
      waited: number; settledAt: number | null; goalDist10: number | null;
      trace: { t: number; d: number | null; goalMoved: number | null }[];
    };
    const { atClick, r } = await page.evaluate(async () => {
      const h = (window as W & { __handoff?: { atClick?: View | null; sample?: Promise<unknown> } }).__handoff!;
      return { atClick: h.atClick, r: (await h.sample) as HandoffSample };
    });
    expect(r.error, r.error).toBeUndefined();
    expect(atClick, 'the card view (front-relative) read at the click').toBeTruthy();
    expect(atClick!.frontRel).toBe(true);
    // the hand-off is one-shot: read and deleted on arrival
    expect(await page.evaluate(() => sessionStorage.getItem('forest.handoff'))).toBeNull();
    const d = dist(r.first.pos, atClick!.pos);
    const m = {
      cardAtClick: atClick,
      cardBeforeClick: beforeClick,
      fullFirst: r.first,
      frontRelDistance: +d.toFixed(4),
      targetFrontRelDistance: +dist(r.first.target, atClick!.target).toFixed(4),
      maxStep: +r.maxStep.toFixed(4),
      maxStep10s: +r.maxStepAll.toFixed(4),
      settledAtMs: r.settledAt === null ? null : Math.round(r.settledAt),
      goalDistanceAt10s: r.goalDist10 === null ? null : +r.goalDist10.toFixed(3),
      steps: r.steps.map((x) => +x.toFixed(3)),
      dtMs: r.dts.map((x) => Math.round(x)),
      trees: r.trees,
      framesWaitedForData: r.waited,
      goalTrace: r.trace,
    };
    console.log(`hand-off: ${JSON.stringify(m)}`);
    test.info().annotations.push({ type: 'handoff', description: JSON.stringify(m) });
    expect(d, 'full window camera vs the card camera at the click, front-relative').toBeLessThanOrEqual(0.5);
    expect(r.first.mode).toBe(atClick!.mode);
    expect(r.steps.length).toBe(30);
    // the existing motion bound (camera moves ≤ 0.1 per frame) over the first 30 frames after the hand-off
    expect(r.maxStep, `per-frame camera steps ${JSON.stringify(m.steps)}`).toBeLessThanOrEqual(0.1);
    // and it settles on the full window's own fit within 10 s
    expect(r.goalDist10, 'distance to the follow goal 10 s after the hand-off').not.toBeNull();
    expect(r.goalDist10!).toBeLessThan(2);
    // the full window opened with the card's range
    await expect(page.getByRole('group', { name: 'Range' }).getByRole('button', { name: '6h' })).toHaveAttribute('aria-pressed', 'true');
    // only one live scene on /forest: the card's canvas is gone
    await expect(page.locator('[data-testid=forest-card]')).toHaveCount(0);
    await expect(page.locator('canvas')).toHaveCount(1);
  });

  test('phone: a still PNG image and no live scene', async ({ browser }) => {
    const ctx = await browser.newContext({ baseURL: BASE, viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
    const page = await ctx.newPage();
    try {
      await loginAdmin(page);
      const polls = recordPolls(page);
      await page.goto('/ui/stats?debug=1');
      await page.getByTestId('forest-card').scrollIntoViewIfNeeded();
      const img = page.locator('[data-testid=forest-card] img[alt="Session forest"]');
      await expect(img).toBeVisible({ timeout: 60_000 });
      const src = (await img.getAttribute('src')) ?? '';
      expect(src.startsWith('data:image/png;base64,'), src.slice(0, 40)).toBe(true);
      const bytes = Math.floor(((src.length - 'data:image/png;base64,'.length) * 3) / 4);
      // the image decodes and is not blank (a blank 390×320 PNG is a few hundred bytes)
      const natural = await img.evaluate((el: HTMLImageElement) => ({ w: el.naturalWidth, h: el.naturalHeight }));
      console.log(`phone still: ${bytes} bytes, ${natural.w}×${natural.h}`);
      expect(bytes, 'still PNG size (bytes)').toBeGreaterThan(5 * 1024);
      expect(natural.w).toBeGreaterThan(0);
      // the scene is disposed: no canvas, no scene behind the hooks, and no more polling
      await expect(page.locator('[data-testid=forest-card] canvas')).toHaveCount(0);
      const s = await page.evaluate(() => (window as W).__forest!.stats());
      expect(s, 'no live scene behind the debug hooks').toEqual({ trees: 0, cubes: 0, flora: 0, frameMs: 0 });
      await page.waitForTimeout(500);
      const p0 = polls.length;
      await page.waitForTimeout(4500);
      expect(polls.length - p0, 'forest polls after the still').toBe(0);
    } finally {
      await ctx.close();
    }
  });

  test('nav: "Forest" opens /ui/forest', async ({ page }) => {
    await loginAdmin(page);
    await page.getByRole('button', { name: 'Open menu' }).click();
    const link = page.getByRole('menu', { name: 'Main menu' }).getByRole('menuitem', { name: 'Forest' });
    await expect(link).toHaveAttribute('href', '/ui/forest');
    await link.click();
    await expect(page).toHaveURL(/\/ui\/forest$/);
    await expect(page.getByTestId('forest-view')).toBeVisible();
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Trackpad gestures (Plan 4 Task 4)
// ---------------------------------------------------------------------------------------------------------------

type CamSnap = { p: V; t: V; d: number; az: number; polar: number; hold: boolean; clear: number; engaged: boolean };
const camSnap = (page: Page) =>
  page.evaluate(() => {
    const f = (window as W).__forest!, p = f.cameraPos(), t = f.cameraTarget()!;
    const o = { x: p.x - t.x, y: p.y - t.y, z: p.z - t.z }, d = Math.hypot(o.x, o.y, o.z);
    return {
      p, t, d, az: Math.atan2(o.x, o.z), polar: Math.acos(o.y / d), hold: f.userHold(),
      clear: p.y - f.groundAt(p.x, p.z), engaged: f.wheelEngaged(),
    } as CamSnap;
  });
/** A synthetic WheelEvent on the canvas (its centre); whether the scene prevented its default. */
const synthWheel = (page: Page, sel: string, init: Record<string, number | boolean>) =>
  page.evaluate(({ sel, init }) => {
    const c = document.querySelector(sel)!, r = c.getBoundingClientRect();
    const e = new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, ...init });
    c.dispatchEvent(e);
    return e.defaultPrevented;
  }, { sel, init });
const hyp = (a: V, b: V) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const angle = (a: number) => Math.abs(Math.atan2(Math.sin(a), Math.cos(a)));
const pageZoom = (page: Page) => page.evaluate(() => ({ scale: window.visualViewport?.scale ?? 1, dpr: window.devicePixelRatio }));

test.describe('trackpad gestures', () => {
  test('full window: slide pans, pinch zooms (never the page), Shift+slide orbits above the horizon, a wheel zooms; held, above ground, Reset view', async ({ page }, testInfo) => {
    await loginAdmin(page);
    await openForest(page);
    const sel = '[data-testid=forest-canvas-host] canvas';
    const zoom0 = await pageZoom(page);
    await startGroundProbe(page);
    // a first small slide takes the camera from the automatic one (the hold)
    expect(await synthWheel(page, sel, { deltaX: 0.5, deltaY: 0.25 })).toBe(true);
    const s0 = await camSnap(page);
    expect(s0.hold, 'a gesture holds the camera like a drag').toBe(true);

    // two-finger slide: camera and target move together, by distance × 0.0016 per px
    expect(await synthWheel(page, sel, { deltaX: 60.5, deltaY: 0.5 })).toBe(true);
    const s1 = await camSnap(page);
    const moved = hyp(s1.t, s0.t);
    expect(moved, 'the slide pans').toBeGreaterThan(0.05 * s0.d);
    expect(Math.abs(s1.d - s0.d), 'a pan keeps the distance').toBeLessThan(0.05 * s0.d);

    // pinch (ctrlKey wheel): zooms in, the default (page zoom) prevented
    expect(await synthWheel(page, sel, { ctrlKey: true, deltaY: -15 })).toBe(true);
    const s2 = await camSnap(page);
    expect(s2.d, 'the pinch zooms in').toBeLessThan(s1.d * 0.95);
    // the same through the browser's input pipeline (CDP): Ctrl + wheel, as Chrome reports a trackpad pinch, at a
    // point where the canvas is not under a panel
    const free = await page.evaluate((sel) => {
      const c = document.querySelector(sel)!, r = c.getBoundingClientRect();
      for (const fy of [0.5, 0.65, 0.35, 0.8])
        for (const fx of [0.5, 0.4, 0.6, 0.3, 0.7]) {
          const x = r.left + r.width * fx, y = r.top + r.height * fy;
          if (document.elementFromPoint(x, y) === c) return { x, y };
        }
      return null;
    }, sel);
    expect(free, 'a free spot of canvas').not.toBeNull();
    await page.mouse.move(free!.x, free!.y);
    await page.keyboard.down('Control');
    await page.mouse.wheel(0, -20);
    await page.keyboard.up('Control');
    await page.waitForTimeout(300);
    const s2b = await camSnap(page);
    expect(s2b.d, 'a real Ctrl+wheel zooms the forest').toBeLessThan(s2.d * 0.95);
    expect(await pageZoom(page), 'the page itself never zooms').toEqual(zoom0);
    // a pinch over a panel of the full window does not zoom the page either
    expect(await page.evaluate(() => {
      const panel = document.querySelector('[data-forest-panel]')!, r = panel.getBoundingClientRect();
      const e = new WheelEvent('wheel', { bubbles: true, cancelable: true, ctrlKey: true, deltaY: -10, clientX: r.left + 5, clientY: r.top + 5 });
      panel.dispatchEvent(e);
      return e.defaultPrevented;
    }), 'a pinch over a panel is not the page\'s').toBe(true);

    // Shift + two fingers: orbit about the target, same distance
    expect(await synthWheel(page, sel, { shiftKey: true, deltaX: 40.5 })).toBe(true);
    const s3 = await camSnap(page);
    expect(angle(s3.az - s2b.az), 'Shift+slide orbits').toBeGreaterThan(0.1);
    expect(hyp(s3.t, s2b.t), 'the orbit keeps the target').toBeLessThan(0.05 * s2b.d + 0.5);
    // pushed down hard: never below the horizon (0.49π, or the polar it already had)
    for (let i = 0; i < 10; i++) await synthWheel(page, sel, { shiftKey: true, deltaY: 300.5 });
    const s4 = await camSnap(page);
    expect(s4.polar).toBeLessThanOrEqual(Math.max(Math.PI * 0.49, s3.polar) + 1e-3);
    expect(s4.p.y).toBeGreaterThan(s4.t.y);

    // a real mouse wheel (line deltas): zoom, as before
    expect(await synthWheel(page, sel, { deltaMode: 1, deltaY: 3 })).toBe(true);
    const s5 = await camSnap(page);
    expect(s5.d, 'a mouse wheel notch zooms out').toBeGreaterThan(s4.d * 1.02);

    // slides that try to drive the camera into the ground: clamped
    for (let i = 0; i < 20; i++) await synthWheel(page, sel, { deltaY: -400.5 });
    await page.waitForTimeout(300);
    const g = await stopGroundProbe(page);
    const s6 = await camSnap(page);
    testInfo.annotations.push({ type: 'gestures', description: JSON.stringify({ s0, s1, s2, s2b, s3, s4, s5, s6, g }) });
    expect(g.minClear).toBeGreaterThanOrEqual(0.5 - 1e-6);
    expect(s6.clear).toBeGreaterThanOrEqual(0.5 - 1e-6);
    expect(s6.hold).toBe(true);

    // Reset view ends the hold
    await page.getByRole('button', { name: 'Reset view' }).click();
    await expect.poll(async () => (await camSnap(page)).hold).toBe(false);
  });

  test('card: the Stats page scrolls over it until it is clicked into; then gestures drive it; a pinch never zooms the page', async ({ page }) => {
    await page.addInitScript(() => {
      try {
        localStorage.setItem('vw.stats.range', '6h');
      } catch {
        /* no storage */
      }
    });
    await loginAdmin(page);
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    const sel = '[data-testid=forest-card] [data-testid=forest-canvas-host] canvas';
    const canvas = page.locator(sel);
    const zoom0 = await pageZoom(page);

    // not engaged: a trackpad slide is the page's (default not prevented), and a real wheel scrolls the page
    const c0 = await camSnap(page);
    expect(c0.engaged).toBe(false);
    const ring = page.getByTestId('forest-card-focus');
    await expect(ring, 'no focus cue before a click').toHaveCount(0);
    expect(await synthWheel(page, sel, { deltaY: -40.5 }), 'an unengaged card leaves the slide to the page').toBe(false);
    const y0 = await page.evaluate(() => window.scrollY);
    expect(y0, 'the card is below the first screen: the page is scrolled').toBeGreaterThan(50);
    let box = (await canvas.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -120);
    await expect.poll(() => page.evaluate(() => window.scrollY), { message: 'the Stats page scrolls over the card' }).toBeLessThan(y0 - 20);
    expect((await camSnap(page)).hold, 'the card camera was not touched').toBe(false);

    // a pinch over the unengaged card zooms the card, never the page
    await scrollCardIn(page);
    expect(await synthWheel(page, sel, { ctrlKey: true, deltaY: -10 })).toBe(true);
    expect((await camSnap(page)).hold).toBe(true);
    expect(await pageZoom(page)).toEqual(zoom0);

    // clicked into: the wheel and the slide are the card's; the page stays put
    box = (await canvas.boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    expect((await camSnap(page)).engaged).toBe(true);
    await expect(ring, 'the focus cue shows while the card captures the trackpad').toBeVisible();
    // Esc lets it go (and the cue with it); a click engages it again
    await page.keyboard.press('Escape');
    expect((await camSnap(page)).engaged).toBe(false);
    await expect(ring).toHaveCount(0);
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await expect(ring).toBeVisible();
    const y1 = await page.evaluate(() => window.scrollY);
    const c1 = await camSnap(page);
    expect(await synthWheel(page, sel, { deltaX: 50.5 }), 'an engaged card takes the slide').toBe(true);
    const c2 = await camSnap(page);
    expect(hyp(c2.t, c1.t), 'the slide pans the card').toBeGreaterThan(0.03 * c1.d);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.wheel(0, -120);
    await page.waitForTimeout(400);
    expect(await page.evaluate(() => window.scrollY), 'the page does not scroll under an engaged card').toBe(y1);
    expect((await camSnap(page)).d, 'the wheel zooms the card').toBeLessThan(c2.d);

    // a pointer-down elsewhere releases it: the page scrolls again
    await page.evaluate(() => document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));
    expect((await camSnap(page)).engaged).toBe(false);
    await expect(ring).toHaveCount(0);
    // a window blur lets it go too
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await expect(ring).toBeVisible();
    await page.evaluate(() => window.dispatchEvent(new Event('blur')));
    await expect(ring).toHaveCount(0);
    expect((await camSnap(page)).engaged).toBe(false);
    expect(await synthWheel(page, sel, { deltaY: -40.5 })).toBe(false);
    await page.getByTestId('forest-card').getByRole('button', { name: 'Reset view' }).click();
    await expect.poll(async () => (await camSnap(page)).hold).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Motion (default cinematic live view)
// ---------------------------------------------------------------------------------------------------------------

const WARMUP_MS = 36_000; // the view is 30 s behind now: let the growing tree's first turns come into view
const SAMPLE_MS = 25_000;
// a second session joins the growing tree here, so the new limb shows ~30 s later, mid-sample
const NEW_SESSION_AFTER_MS = 28_000;
const RESUME_AFTER_MS = 12_000; // a session R comes back here, so its transplant shows ~30 s later, mid-sample

interface Sample {
  frames: number;
  p50: number;
  p99: number;
  maxFrame: number;
  maxCamStep: number;
  maxGirthStep: number;
  maxGirthTree: string;
  /** The largest steps with the frame time they happened in (ms). */
  topGirth: { step: number; from: number; dt: number; id: string }[];
  topCam: { step: number; dt: number }[];
  /** Largest look-direction turn rate over one frame (deg/s), and the worst frames. */
  maxTurnDegS: number;
  topTurn: { degS: number; deg: number; dt: number }[];
  /** Geometry builds of the growing tree that arrived during the sample (worker replies keyed by tree id). */
  growingBuilds: number;
  girthOver: number;
  framesOver25: number;
  /**
   * Frames over 25 ms, and how many of them came within 100 ms after a geometry build arrived (a worker reply) or a
   * forest poll response landed: tells main-thread data work apart from steady render cost.
   */
  slow: { n: number; afterBuild: number; afterPoll: number; other: number };
  /** Main-thread tasks over 50 ms during the sample (PerformanceObserver "longtask"). */
  longTasks: { n: number; totalMs: number; maxMs: number };
  girthTrees: number;
  growingGirth: { first: number | null; last: number | null };
  /** The growing tree's fine voxel cells at the first and last frame it was sampled (new cubes landed in between). */
  growingCells: { first: number | null; last: number | null };
  /** Frames where some tree's girth went down (cells never go: none). */
  girthBack: number;
  flights: { treeId: string; fromX: number; toX: number }[];
  flightTrees: string[];
  minFlightDx: number | null;
  stats: ForestStats;
  workerPosts: number;
  workerReplies: number;
  seconds: number;
}

/** Samples every animation frame for `ms`: frame time, camera step, every tree's girth step and the flights. */
/** `trunkId`: a tree that first appears during the sample (its trunk base must thicken: S1). */
async function sampleMotion(page: Page, ms: number, growingId: string, trunkId: string): Promise<Sample> {
  return page.evaluate(
    async ({ ms, growingId, trunkId }) => {
      const f = (window as W).__forest!;
      const wk: WorkerLog = (window as W).__e2eWorkers ?? { posts: 0, replies: 0, replyAt: [], built: [] };
      const posts0 = wk.posts, replies0 = wk.replies;
      const dts: number[] = [];
      const frameEnds: { at: number; dt: number }[] = [];
      let maxCam = 0, maxGirth = 0, maxGirthTree = '', girthOver = 0, girthBack = 0;
      let cellsFirst: number | null = null, cellsLast: number | null = null;
      const topGirth: { step: number; from: number; dt: number; id: string }[] = [];
      const topCam: { step: number; dt: number }[] = [];
      const topTurn: { degS: number; deg: number; dt: number; step: number }[] = [];
      let maxTurn = 0;
      const unit = (v: { x: number; y: number; z: number } | null) => {
        const l = v ? Math.hypot(v.x, v.y, v.z) : 0;
        return v && l > 1e-9 ? { x: v.x / l, y: v.y / l, z: v.z / l } : null;
      };
      let prevLook = unit(f.cameraLook());
      const top = <T extends { step: number }>(arr: T[], x: T) => {
        arr.push(x);
        arr.sort((a, b) => b.step - a.step);
        arr.length = Math.min(arr.length, 5);
      };
      const girth = new Map<string, number>();
      const flights: { treeId: string; fromX: number; toX: number }[] = [];
      let minDx: number | null = null;
      let prevCam = f.cameraPos(), prevShown = f.shownAbs();
      let growFirst: number | null = null, growLast: number | null = null;
      const long = { n: 0, totalMs: 0, maxMs: 0 };
      const po = new PerformanceObserver((l) => {
        for (const e of l.getEntries()) (long.n++, (long.totalMs += e.duration), (long.maxMs = Math.max(long.maxMs, e.duration)));
      });
      try {
        po.observe({ type: 'longtask' });
      } catch {
        /* no longtask support */
      }
      const t0 = performance.now();
      let last: number | null = null;
      await new Promise<void>((resolve) => {
        const tick = (t: number) => {
          const dt = last === null ? 0 : t - last;
          if (last !== null) dts.push(dt), frameEnds.push({ at: t, dt });
          last = t;
          const c = f.cameraPos();
          const cs = Math.hypot(c.x - prevCam.x, c.y - prevCam.y, c.z - prevCam.z);
          maxCam = Math.max(maxCam, cs);
          top(topCam, { step: cs, dt });
          prevCam = c;
          const look = unit(f.cameraLook());
          if (look && prevLook && dt > 0) {
            const dot = Math.min(1, Math.max(-1, look.x * prevLook.x + look.y * prevLook.y + look.z * prevLook.z));
            const deg = (Math.acos(dot) * 180) / Math.PI, degS = deg / (dt / 1000);
            maxTurn = Math.max(maxTurn, degS);
            top(topTurn, { degS, deg, dt, step: degS });
          }
          prevLook = look;
          const shown = f.shownAbs(), dShown = Math.max(0, shown - prevShown);
          prevShown = shown;
          for (const id of f.treeIds()) {
            const g = f.girthAt(id);
            if (g === null) {
              girth.delete(id); // not built (or dropped): a fresh start, not a step
              continue;
            }
            const p = girth.get(id);
            if (p !== undefined) {
              const st = Math.abs(g - p);
              if (st > maxGirth) (maxGirth = st), (maxGirthTree = id);
              // S1, the cube form of the bark's bound: ≤ one cell (0.25 u) per scale-in (0.8 s) of the history shown in
              // this frame (≈ the frame's dt at 1×; ≈ 0.0063 u at 60 fps), + 20 %. Growth runs on the shown time, which a
              // frame can advance a little more than its wall dt (the clock's catch-up, rAF order), so it is read here
              if (st > (1.2 * 0.25 * Math.max(dShown, 1e-3)) / 0.8 + 1e-9) girthOver++;
              if (g < p - 1e-9) girthBack++;
              top(topGirth, { step: st, from: p, dt, id });
            }
            girth.set(id, g);
            if (id === trunkId) (growFirst ??= g), (growLast = g);
            if (id === growingId) {
              const c = f.cellsAt(id);
              if (c !== null) (cellsFirst ??= c), (cellsLast = c);
            }
          }
          for (const fl of f.flights()) {
            flights.push(fl);
            const d = fl.toX - fl.fromX;
            minDx = minDx === null ? d : Math.min(minDx, d);
          }
          if (t - t0 < ms) requestAnimationFrame(tick);
          else resolve();
        };
        requestAnimationFrame(tick);
      });
      const s = [...dts].sort((a, b) => a - b);
      const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] ?? 0;
      po.disconnect();
      const polls = performance
        .getEntriesByType('resource')
        .filter((e) => e.name.includes('/api/stats/forest') && e.startTime >= t0)
        .map((e) => (e as PerformanceResourceTiming).responseEnd);
      const near = (at: number, evs: number[]) => evs.some((x) => x <= at && at - x <= 100);
      const slow = { n: 0, afterBuild: 0, afterPoll: 0, other: 0 };
      for (const fr of frameEnds) {
        if (fr.dt <= 25) continue;
        slow.n++;
        const b = near(fr.at, wk.replyAt), p = near(fr.at, polls);
        if (b) slow.afterBuild++;
        if (p) slow.afterPoll++;
        if (!b && !p) slow.other++;
      }
      const seconds = (performance.now() - t0) / 1000;
      return {
        frames: dts.length,
        p50: q(0.5),
        p99: q(0.99),
        maxFrame: s.at(-1) ?? 0,
        maxCamStep: maxCam,
        maxGirthStep: maxGirth,
        maxGirthTree,
        topGirth,
        topCam,
        maxTurnDegS: maxTurn,
        topTurn: topTurn.map(({ degS, deg, dt }) => ({ degS, deg, dt })),
        growingBuilds: wk.built.filter((b) => b.key === growingId && b.at >= t0).length,
        girthOver,
        girthBack,
        growingCells: { first: cellsFirst, last: cellsLast },
        framesOver25: dts.filter((d) => d > 25).length,
        longTasks: long,
        slow,
        girthTrees: girth.size,
        growingGirth: { first: growFirst, last: growLast },
        flights,
        flightTrees: [...new Set(flights.map((x) => x.treeId))],
        minFlightDx: minDx,
        stats: f.stats(),
        workerPosts: wk.posts - posts0,
        workerReplies: wk.replies - replies0,
        seconds,
      };
    },
    { ms, growingId, trunkId },
  );
}

/** Self time by function (top 25) of a CDP CPU profile. */
function summarizeProfile(p: { nodes: { id: number; callFrame: { functionName: string; url: string; lineNumber: number; columnNumber: number } }[]; samples?: number[]; timeDeltas?: number[] }): string {
  const byId = new Map(p.nodes.map((n) => [n.id, n]));
  const self = new Map<string, number>();
  let total = 0;
  (p.samples ?? []).forEach((id, i) => {
    const dt = (p.timeDeltas?.[i] ?? 0) / 1000;
    const n = byId.get(id);
    if (!n) return;
    const key = `${n.callFrame.functionName || '(anon)'} ${n.callFrame.url.split('/').pop()}:${n.callFrame.lineNumber}:${n.callFrame.columnNumber}`;
    self.set(key, (self.get(key) ?? 0) + dt);
    total += dt;
  });
  return [...self.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 25)
    .map(([k, ms]) => `  ${ms.toFixed(0).padStart(7)} ms ${((100 * ms) / total).toFixed(1).padStart(5)}%  ${k}`)
    .join('\n');
}

test.describe('motion @motion', () => {
  // One sample per project, shared by the four bound tests. Not serial (a failed bound must not skip the others); a
  // failure restarts the worker and re-runs beforeAll, which then reads the sample back instead of sampling again (a
  // second sample would need another resumable session and another minute).
  let sample: Sample;
  let gpu: { renderer: string; software: boolean };
  let resumed = '';
  /** os.loadavg() over the sample: [1, 5, 15 min]. */
  let loadAvg: number[] = [];

  test.beforeAll(async ({ browser }: { browser: Browser }) => {
    test.setTimeout(240_000);
    // the sample is reused only within the run that took it (a failed bound restarts the worker): keyed by the run id,
    // inside the run's own data dir, never a shared temp dir
    const dataDir = process.env.FOREST_E2E_DATA_DIR, runId = process.env.FOREST_E2E_RUN_ID;
    if (!dataDir || !runId) throw new Error('forest.spec: no forest stack (run with -c playwright.forest.config.ts)');
    const cache = path.join(dataDir, `motion-${runId}-${test.info().project.name}.json`);
    if (existsSync(cache)) {
      ({ sample, gpu, resumed, loadAvg } = JSON.parse(readFileSync(cache, 'utf8')));
      return;
    }
    const dpr = process.env.FOREST_E2E_DPR ? { deviceScaleFactor: Number(process.env.FOREST_E2E_DPR) } : {};
    const ctx = await browser.newContext({ baseURL: BASE, viewport: { width: 1440, height: 900 }, ...dpr });
    const browserPage = await ctx.newPage();
    await browserPage.addInitScript(COUNT_WORKERS);
    gpu = { renderer: '', software: true };
    await loginAdmin(browserPage);
    const grow = startGrowing(DB, SEED.keyA, { newSessionAfterMs: NEW_SESSION_AFTER_MS });
    let sprout: ReturnType<typeof startGrowing> | null = null;
    try {
      await openForest(browserPage);
      gpu = await webglRenderer(browserPage);
      const dprSeen = await browserPage.evaluate(() => window.devicePixelRatio);
      console.log(`devicePixelRatio ${dprSeen}: the scene renders at ${Math.min(1.5, dprSeen)}x`);
      // the default view: cinematic, live
      await expect(browserPage.getByRole('button', { name: 'Cinematic' })).toHaveAttribute('aria-pressed', 'true');
      await browserPage.waitForTimeout(RESUME_AFTER_MS);
      resumed = resumeSession(DB, SEED.keyA, SEED.resumeSessionKeys);
      // a new tree (key B's last spell ended hours ago): its first turn is shown ~6 s into the sample, so its trunk base
      // gains its cells (2 × 2 blocks, from the inside out) while the sample runs (S1: the girth must thicken)
      sprout = startGrowing(DB, SEED.keyB, { prior: 0 });
      await browserPage.waitForTimeout(WARMUP_MS - RESUME_AFTER_MS);
      // FOREST_E2E_PROFILE=1: a CPU profile of the sample (CDP), written next to the stack's data and summarized
      const cdp = process.env.FOREST_E2E_PROFILE ? await ctx.newCDPSession(browserPage) : null;
      if (cdp) {
        await cdp.send('Profiler.enable');
        await cdp.send('Profiler.setSamplingInterval', { interval: 200 });
        await cdp.send('Profiler.start');
      }
      sample = await sampleMotion(browserPage, SAMPLE_MS, grow.treeId, sprout.treeId);
      loadAvg = os.loadavg().map((x) => +x.toFixed(2));
      if (cdp) {
        const { profile } = await cdp.send('Profiler.stop');
        const out = path.join(dataDir, `motion-${runId}-${test.info().project.name}.cpuprofile`);
        writeFileSync(out, JSON.stringify(profile));
        console.log(`cpu profile: ${out}\n${summarizeProfile(profile)}`);
      }
    } finally {
      const added = grow.stop();
      sprout?.stop();
      console.log(`growing tree: ${added} turns added`);
      await ctx.close();
    }
    const report = {
      project: test.info().project.name,
      renderer: gpu.renderer,
      software: gpu.software,
      resumed,
      loadAvg,
      ...sample,
      flights: sample.flights.length,
      jobsPerS: +(sample.workerPosts / sample.seconds).toFixed(2),
      geometryBuildsPerS: +((2 * sample.workerPosts) / sample.seconds).toFixed(2),
    };
    console.log(`motion sample: ${JSON.stringify(report)}`);
    writeFileSync(cache, JSON.stringify({ sample, gpu, resumed, loadAvg }));
  });

  /** Every motion test leaves the numbers behind (R27), so each run records them whatever passes or skips. */
  test.beforeEach(async ({}, testInfo: TestInfo) => {
    testInfo.annotations.push({
      type: 'motion',
      description: JSON.stringify({
        p99Ms: +sample.p99.toFixed(1),
        maxCamStep: +sample.maxCamStep.toPrecision(3),
        maxTurnDegS: +sample.maxTurnDegS.toFixed(1),
        maxGirthStep: +sample.maxGirthStep.toPrecision(3),
        pixelRatio: sample.stats.pixelRatio ?? null,
        loadAvg: loadAvg,
        renderer: gpu.renderer,
      }),
    });
  });

  test('p99 frame time ≤ 25 ms (FOREST_PERF=1, unloaded target machine)', async () => {
    test.skip(
      process.env.FOREST_PERF !== '1',
      'frame time is a manual release check: set FOREST_PERF=1 and measure on an unloaded target machine (RTX A4000 class)',
    );
    test.fixme(gpu.software, `no hardware GPU (WebGL renderer: ${gpu.renderer}); frame times are meaningless`);
    expect(sample.frames).toBeGreaterThan(100);
    expect(sample.p99, `p99 ${sample.p99.toFixed(1)} ms over ${sample.frames} frames`).toBeLessThanOrEqual(25);
  });

  test('camera moves ≤ 0.1 per frame', async () => {
    expect(sample.frames).toBeGreaterThan(20);
    expect(sample.maxCamStep, `worst steps ${JSON.stringify(sample.topCam)}`).toBeLessThanOrEqual(0.1);
  });

  test('look direction turns ≤ 90°/s', async () => {
    // the camera planner's own limit is ≈ 69°/s (camera.ts); 90°/s leaves headroom for frame-time jitter
    expect(sample.frames).toBeGreaterThan(20);
    expect(sample.maxTurnDegS, `worst frames ${JSON.stringify(sample.topTurn)}`).toBeLessThanOrEqual(90);
  });

  test('trunk girth (the wood extent at the base) never shrinks, steps ≤ a cell per scale-in, and thickens', async () => {
    expect(sample.girthTrees).toBeGreaterThan(0);
    // the new tree's trunk was on screen and sampled, and its base thickened during the sample
    expect(sample.growingGirth.first).not.toBeNull();
    expect(sample.growingGirth.last!, 'the trunk base thickened during the sample').toBeGreaterThan(sample.growingGirth.first!);
    // not a static forest: the growing tree got new builds and new cubes during the sample
    expect(sample.growingBuilds, 'geometry builds of the growing tree during the sample').toBeGreaterThan(0);
    expect(sample.growingCells.last!, 'cells of the growing tree').toBeGreaterThan(sample.growingCells.first!);
    // non-decreasing, and smooth: every step within the scale-in rate
    expect(sample.girthBack, `worst steps ${JSON.stringify(sample.topGirth)}`).toBe(0);
    expect(sample.girthOver, `worst steps ${JSON.stringify(sample.topGirth)}`).toBe(0);
  });

  test('transplants only move forward', async () => {
    // a session R came back mid-test: its tree flew
    expect(sample.flights.length).toBeGreaterThan(0);
    for (const f of sample.flights) expect(f.toX - f.fromX, f.treeId).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// A farm-like week (quiet hours compressed): last in the file, it adds two keys' week to the stack's data
// ---------------------------------------------------------------------------------------------------------------

const median = (xs: number[]) => {
  const v = [...xs].sort((a, b) => a - b);
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : 0;
};

test.describe('week', () => {
  test('7d Wide over a farm-like week: the trees are clearly visible (median height on screen)', async ({ page }, testInfo) => {
    const rows = seedFarmWeek(DB);
    const grow = startGrowing(DB, SEED.keyA, {}); // a tree in the last hour (the 1 h view)
    await loginAdmin(page);
    // the short ranges, for the record (they should read as before the compression)
    const short: Record<string, { median: number; n: number; span: number }> = {};
    for (const r of ['1h', '6h', '24h']) {
      await page.goto(`/ui/forest?range=${r}&debug=1`);
      await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
      await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
      await page.waitForTimeout(6000);
      const v = await page.evaluate(() => ({ px: (window as W).__forest!.treePx(), span: (window as W).__forest!.spanLength(), td: (window as W).__forest!.treeDist(), h: innerHeight }));
      short[r] = { median: median(v.px), n: v.px.length, span: v.span };
      // "bigger when far": true size in every 1h / 6h / 24h view
      expect(Math.max(...v.td.map((x) => x.scale)), `${r}: largest tree scale`).toBe(1);
      console.log(`DIST full ${r}: d ${median(v.td.map((x) => x.d)).toFixed(0)} [${Math.min(...v.td.map((x) => x.d)).toFixed(0)}..${Math.max(...v.td.map((x) => x.d)).toFixed(0)}] top ${median(v.td.map((x) => x.top)).toFixed(2)}`);
    }
    await page.goto('/ui/forest?range=7d&debug=1');
    await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
    await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
    await page.waitForTimeout(8000);
    const full = await page.evaluate(() => ({ px: (window as W).__forest!.treePx(), span: (window as W).__forest!.spanLength(), stats: (window as W).__forest!.stats(), td: (window as W).__forest!.treeDist(), ids: (window as W).__forest!.treeIds() }));
    // every tree of the week is drawn, the live front included (re-review 3, N3)
    console.log(`7d full: ${full.stats.trees} trees built of ${full.ids.length}; cubes ${full.stats.cubes} + flora ${full.stats.flora}, trees per level ${JSON.stringify(full.stats.levels)}, cubes if all at level 0/1/2 ${JSON.stringify(await page.evaluate(() => (window as W).__forest!.levelCubes()))}`);
    expect(full.stats.trees).toBe(full.ids.length);
    // the cube budget (trees, ghosts and the flora's voxels): ≤ 120k in the full window
    expect(full.stats.cubes + full.stats.flora).toBeLessThanOrEqual(120_000);
    console.log(`DIST full 7d: d ${median(full.td.map((x) => x.d)).toFixed(0)} [${Math.min(...full.td.map((x) => x.d)).toFixed(0)}..${Math.max(...full.td.map((x) => x.d)).toFixed(0)}] top ${median(full.td.map((x) => x.top)).toFixed(2)} scale ${median(full.td.map((x) => x.scale)).toFixed(2)} max ${Math.max(...full.td.map((x) => x.scale)).toFixed(2)}`);
    // no scaled crown overlaps a neighbour's (crowns already touching at true size stay at 1)
    let overlaps = 0;
    for (const p of full.td)
      for (const q of full.td) {
        if (p === q) continue;
        const d = Math.hypot(p.x - q.x, p.z - q.z);
        if (d >= p.r + q.r && p.scale * p.r + q.scale * q.r > d + 1e-6) overlaps++;
      }
    expect(overlaps, 'scaled crowns overlapping').toBe(0);
    // hover picking works on the scaled trees: scanning up the scaled trunk of the most scaled tree finds that tree,
    // above its true-size top too
    const big = full.td.reduce((a, b) => (a.scale > b.scale ? a : b));
    const picked = await page.evaluate((t) => {
      const f = (window as W).__forest!;
      const hits: string[] = [];
      for (let k = 1; k <= 30; k++) {
        const y = (k / 30) * t.top * t.scale;
        for (const dx of [0, 0.4, -0.4]) {
          const r = f.pickWorld(t.x + dx, y, t.z);
          if (r && r.kind === 'turn' && r.treeId === t.id) hits.push(y > t.top ? 'above' : 'below');
        }
      }
      return hits;
    }, big);
    console.log(`pick on the most scaled tree (scale ${big.scale.toFixed(2)}): ${picked.length} hits, ${picked.filter((h) => h === 'above').length} above its true top`);
    expect(picked.length).toBeGreaterThan(0);
    expect(picked.filter((h) => h === 'above').length).toBeGreaterThan(0);
    // zooming in returns the trees to true size, and no tree moved
    const canvasBox = (await page.locator('[data-testid=forest-canvas-host] canvas').boundingBox())!;
    await page.mouse.move(canvasBox.x + canvasBox.width * 0.55, canvasBox.y + canvasBox.height * 0.6);
    for (let i = 0; i < 25; i++) await page.mouse.wheel(0, -600);
    // the scales ease back (SCALE_TAU 0.25 s, no pop): a zoom that lands at once settles within a few seconds
    await page
      .waitForFunction(() => {
        const t = (window as W).__forest!.treeDist();
        return t.length > 0 && t.reduce((a, b) => (a.d < b.d ? a : b)).scale === 1;
      }, null, { timeout: 5000 })
      .catch(() => {});
    const near = await page.evaluate(() => (window as W).__forest!.treeDist());
    const byPos = new Map(full.td.map((x) => [`${x.x.toFixed(4)},${x.z.toFixed(4)}`, x]));
    const closest = near.reduce((a, b) => (a.d < b.d ? a : b));
    console.log(`zoomed in: closest tree ${closest.d.toFixed(0)} u, scale ${closest.scale.toFixed(3)}; ${near.filter((x) => byPos.has(`${x.x.toFixed(4)},${x.z.toFixed(4)}`)).length}/${near.length} places unchanged`);
    expect(closest.scale).toBe(1);
    expect(near.every((x) => byPos.has(`${x.x.toFixed(4)},${x.z.toFixed(4)}`))).toBe(true);
    await page.getByRole('button', { name: 'Reset view' }).click();
    // the card at 24 h: true size too
    await page.evaluate(() => localStorage.setItem('vw.stats.range', '24h'));
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    await page.waitForTimeout(6000);
    const card24 = await page.evaluate(() => (window as W).__forest!.treeDist());
    console.log(`DIST card 24h: d ${median(card24.map((x) => x.d)).toFixed(0)} [${Math.min(...card24.map((x) => x.d)).toFixed(0)}..${Math.max(...card24.map((x) => x.d)).toFixed(0)}] max scale ${Math.max(...card24.map((x) => x.scale)).toFixed(3)}`);
    expect(Math.max(...card24.map((x) => x.scale)), 'card 24h: largest tree scale').toBe(1);
    await page.evaluate(() => localStorage.setItem('vw.stats.range', '7d'));
    await page.goto('/ui/stats?debug=1');
    await scrollCardIn(page);
    await page.waitForTimeout(8000);
    const card = await page.evaluate(() => ({ px: (window as W).__forest!.treePx(), span: (window as W).__forest!.spanLength(), stats: (window as W).__forest!.stats(), td: (window as W).__forest!.treeDist(), ids: (window as W).__forest!.treeIds() }));
    console.log(`7d card: ${card.stats.trees} trees built of ${card.ids.length}; cubes ${card.stats.cubes} + flora ${card.stats.flora}, trees per level ${JSON.stringify(card.stats.levels)}, cubes if all at level 0/1/2 ${JSON.stringify(await page.evaluate(() => (window as W).__forest!.levelCubes()))}`);
    expect(card.stats.trees).toBe(card.ids.length);
    expect(card.stats.cubes + card.stats.flora, 'the card cube budget').toBeLessThanOrEqual(15_000);
    console.log(`DIST card 7d: d ${median(card.td.map((x) => x.d)).toFixed(0)} [${Math.min(...card.td.map((x) => x.d)).toFixed(0)}..${Math.max(...card.td.map((x) => x.d)).toFixed(0)}] top ${median(card.td.map((x) => x.top)).toFixed(2)} scale ${median(card.td.map((x) => x.scale)).toFixed(2)} max ${Math.max(...card.td.map((x) => x.scale)).toFixed(2)}`);
    grow.stop();
    const m = { rows, short, full: { median: median(full.px), n: full.px.length, span: full.span, trees: full.stats.trees }, card: { median: median(card.px), n: card.px.length, span: card.span, trees: card.stats.trees } };
    testInfo.annotations.push({ type: 'week', description: JSON.stringify(m) });
    console.log(`week 7d: ${JSON.stringify(m)}`);
    expect(full.px.length).toBeGreaterThan(10);
    // Quiet hours only (re-review N1): ~8 px full window / ~9 px card at true size on this week. "Bigger when far"
    // "bigger when far" (user decision): the 7 d median tree as drawn
    // the user's target was 20 px; the no-overlap cap holds the farm week's median scale to ~2.2. Since every tree of the
    // week is drawn (re-review 3, N3: the live front and both ends too, smaller on screen), the median is ~13 px.
    // Plan 4 R5 (one round crown per tree, the mockup's fit): the crowns are wider (1.7σ + leaf size), so the no-overlap
    // cap holds the median far scale to ~1.36 (was ~1.72 with the twig-following crowns) at the same 5.25 u median top
    expect(m.full.median, 'median tree height on screen, full window (px)').toBeGreaterThanOrEqual(9); // measured 9.8 px (was 11.8–12.8)
    expect(m.card.median, 'median tree height on screen, card (px)').toBeGreaterThanOrEqual(11); // measured 11.6 px (was 13.7–15.4)
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Perf matrix (Plan 4, Task 5): cubes, draw calls and frame times per range, full window and card, over a farm week
// ---------------------------------------------------------------------------------------------------------------

/** rAF intervals over `ms` on the open page, with the scene's stats at the end. */
async function sampleFrames(page: Page, ms: number) {
  return page.evaluate(async (ms) => {
    const dts: number[] = [];
    const long = { n: 0, maxMs: 0 };
    const po = new PerformanceObserver((l) => {
      for (const e of l.getEntries()) (long.n++, (long.maxMs = Math.max(long.maxMs, e.duration)));
    });
    try {
      po.observe({ type: 'longtask' });
    } catch {
      /* no longtask support */
    }
    await new Promise<void>((resolve) => {
      let last: number | null = null;
      const t0 = performance.now();
      const tick = (t: number) => {
        if (last !== null) dts.push(t - last);
        last = t;
        if (t - t0 < ms) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
    po.disconnect();
    const s = [...dts].sort((a, b) => a - b);
    const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] ?? 0;
    const f = (window as W).__forest!;
    return { frames: dts.length, p50: q(0.5), p99: q(0.99), max: s.at(-1) ?? 0, over25: dts.filter((d) => d > 25).length, long, stats: f.stats(), ids: f.treeIds().length };
  }, ms);
}

test.describe('perf matrix', () => {
  // FOREST_PERF=1: 15 s per view and the p99 bounds (spec §6.5: full ≤ 25 ms, card ≤ 20 ms), a manual check on an
  // unloaded target machine (R27). Without it: 5 s per view, numbers recorded, budgets asserted.
  // FOREST_PERF_FARM=<keys>x<turns>: the extra farm on top of the week test's (default 2x300: with the default week,
  // ~33k turns over ~150 trees at 7 d).
  test('cubes, draw calls and p99 per range (1h, 6h, 24h, 7d), full window and card, over a realistic farm week', async ({ page }, testInfo) => {
    test.setTimeout(900_000);
    if (!farmSeeded(DB)) seedFarmWeek(DB);
    if (!farmSeeded(DB, 'e2e-bigfarm')) {
      const [keys, turns] = (process.env.FOREST_PERF_FARM ?? '2x300').split('x').map(Number);
      seedFarmWeek(DB, undefined, { keys, turns, name: 'e2e-bigfarm' });
    }
    const rows = historyRows(DB);
    const perf = process.env.FOREST_PERF === '1';
    const ms = perf ? 15_000 : 5_000;
    const grow = startGrowing(DB, SEED.keyA, {}); // a tree in the last hour (the 1 h view)
    const out: Record<string, unknown>[] = [];
    try {
      await loginAdmin(page);
      await page.goto('/ui/forest?range=1h&debug=1');
      const gpu = await webglRenderer(page);
      for (const r of ['1h', '6h', '24h', '7d']) {
        await page.goto(`/ui/forest?range=${r}&debug=1`);
        await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
        await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
        await page.waitForTimeout(8000); // the glide and the builds settle
        const x = await sampleFrames(page, ms);
        out.push({ view: 'full', range: r, ...x });
        expect(x.stats.cubes + x.stats.flora, `full ${r}: cube budget`).toBeLessThanOrEqual(120_000);
        // every tree in range is built up to the build budget (60; FAR_BUILT 200 in a far view: a realistic week's ~155
        // trees all stand at 7 d, final review I2; recorded: built / in range)
        expect(x.stats.trees, `full ${r}: trees built`).toBeGreaterThanOrEqual(Math.min(x.ids, r === '7d' ? 200 : 60));
        expect(x.stats.trees, `full ${r}: trees built`).toBeLessThanOrEqual(Math.min(x.ids, 200));
      }
      for (const r of ['1h', '6h', '24h', '7d']) {
        await page.evaluate((r) => localStorage.setItem('vw.stats.range', r), r);
        await page.goto('/ui/stats?debug=1');
        await scrollCardIn(page);
        await page.waitForTimeout(8000);
        const x = await sampleFrames(page, ms);
        out.push({ view: 'card', range: r, ...x });
        expect(x.stats.cubes + x.stats.flora, `card ${r}: cube budget`).toBeLessThanOrEqual(15_000);
        if (r === '7d') expect(x.stats.trees, 'card 7d: every tree of the week built (final review I2)').toBeGreaterThanOrEqual(Math.min(x.ids, 200, Math.floor((15_000 - x.stats.flora) / 70)));
      }
      // the Stats page itself at 7 d with the card scrolled away (paused): how much of the card's frame time is the page's
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForFunction(() => (window as W).__forest?.stats().paused === true, null, { timeout: 10_000 });
      await page.waitForTimeout(3000);
      out.push({ view: 'page', range: '7d', ...(await sampleFrames(page, ms)) });
      const table = out.map((o) => {
        const x = o as unknown as Awaited<ReturnType<typeof sampleFrames>> & { view: string; range: string };
        return `${x.view.padEnd(4)} ${x.range.padStart(3)}  trees ${String(x.stats.trees).padStart(3)}/${String(x.ids).padEnd(3)}  cubes ${String(x.stats.cubes).padStart(6)} + flora ${String(x.stats.flora).padStart(4)}  levels ${JSON.stringify(x.stats.levels)}  draws ${String(x.stats.drawCalls).padStart(4)}  tris ${String(x.stats.triangles).padStart(8)}  p50 ${x.p50.toFixed(1)}  p99 ${x.p99.toFixed(1)}  max ${x.max.toFixed(1)}  >25ms ${x.over25}/${x.frames}  long ${x.long.n} (max ${x.long.maxMs.toFixed(0)})  pr ${x.stats.pixelRatio}`;
      });
      const load = os.loadavg().map((v) => +v.toFixed(2));
      console.log(`perf matrix (${perf ? 'FOREST_PERF=1' : 'no FOREST_PERF'}, ${ms / 1000} s per view, ${rows} rows, load ${load.join(' ')}, ${gpu.renderer}):\n${table.join('\n')}`);
      testInfo.annotations.push({ type: 'perf', description: JSON.stringify({ rows, perf, load, renderer: gpu.renderer, out: out.map((o) => ({ ...o, stats: undefined, ...(o.stats as object) })) }) });
      if (perf) {
        test.fixme(gpu.software, `no hardware GPU (WebGL renderer: ${gpu.renderer}); frame times are meaningless`);
        for (const o of (out as { view: string; range: string; p99: number }[]).filter((o) => o.view !== 'page'))
          expect(o.p99, `${o.view} ${o.range}: p99 (load ${load.join(' ')})`).toBeLessThanOrEqual(o.view === 'card' ? 20 : 25);
      }
    } finally {
      grow.stop();
    }
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Screenshots for the far-view check (review I2): FOREST_SHOTS_DIR=<dir> runs it, otherwise it is skipped.
// ---------------------------------------------------------------------------------------------------------------

test.describe('shots', () => {
  test.skip(!process.env.FOREST_SHOTS_DIR, 'FOREST_SHOTS_DIR not set');
  // FOREST_SHOTS_TZ: the browser's time zone (the sky is always day; kept for other clock-dependent checks)
  if (process.env.FOREST_SHOTS_TZ) test.use({ timezoneId: process.env.FOREST_SHOTS_TZ });
  test('full window and card in Wide at 1h, 6h, 24h and 7d', async ({ page }) => {
    test.setTimeout(300_000);
    const dir = process.env.FOREST_SHOTS_DIR!;
    // a farm-like week for the 24 h and 7 d views (once: the week test and the perf matrix may have seeded it already)
    if (process.env.FOREST_SHOTS_FARM !== '0' && !farmSeeded(DB)) seedFarmWeek(DB);
    const grow = startGrowing(DB, SEED.keyA, {}); // trees in the last hour
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    try {
      await loginAdmin(page);
      const ranges = (process.env.FOREST_SHOTS_RANGES ?? '1h,6h,24h,7d').split(',');
      for (const r of ranges) {
        await page.goto(`/ui/forest?range=${r}&debug=1`);
        await page.waitForFunction(() => ((window as W).__forest?.stats().trees ?? 0) > 0, null, { timeout: 60_000 });
        await page.getByRole('group', { name: 'Camera' }).getByRole('button', { name: 'Wide' }).click();
        await page.waitForTimeout(6000);
        const cam = await page.evaluate(() => { const f = (window as W).__forest!; return { pos: f.cameraPos(), look: f.cameraLook(), stats: f.stats(), levelCubes: f.levelCubes() }; });
        console.log(`shot full ${r}: ${JSON.stringify(cam)}`);
        await page.screenshot({ path: path.join(dir, `shots-${r}.png`) });
        if (r === '1h') {
          // a close-up (the target screenshot's framing): the nearest tree from ~18 u, a little above, held like a drag
          await page.evaluate(() => {
            const f = (window as W).__forest!, t = f.treeDist().sort((a, b) => a.d - b.d)[0];
            const k = Number(new URL(location.href).searchParams.get('closeK') ?? 1);
            f.holdPose({ x: t.x - 6 * k, y: t.top * 0.8 + 3.5 * k, z: t.z + 17 * k }, { x: t.x, y: t.top * 0.45, z: t.z });
          });
          await page.waitForTimeout(1500);
          await page.screenshot({ path: path.join(dir, 'shots-close.png') });
        }
      }
      for (const r of process.env.FOREST_SHOTS_CARD === '0' ? [] : ranges) {
        await page.evaluate((r) => localStorage.setItem('vw.stats.range', r), r);
        await page.goto('/ui/stats?debug=1');
        await scrollCardIn(page);
        await page.waitForTimeout(6000);
        console.log(`shot card ${r}: ${JSON.stringify(await page.evaluate(() => ({ stats: (window as W).__forest!.stats(), levelCubes: (window as W).__forest!.levelCubes() })))}`);
        await page.getByTestId('forest-card').screenshot({ path: path.join(dir, `shots-card-${r}.png`) });
      }
    } finally {
      grow.stop();
    }
    expect(errors, errors.join('\n')).toEqual([]);
  });
});
