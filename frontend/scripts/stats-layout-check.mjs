// Real-browser layout check of the Stats pages (stats redesign, fixed-size rule).
//
// jsdom has no layout, so the component tests can only read the heights each
// panel DECLARES. This script opens the real page in headless Chrome and checks
// what jsdom cannot see:
//   * every fixed panel above the Requests block keeps its height while the page
//     is watched (traffic, window and tab switches), and
//   * nothing inside a fixed panel overflows it (scrollHeight <= clientHeight,
//     scrollWidth <= clientWidth): no clipped axis, legend or line,
//   * the header row fits the viewport (the window selector is reachable),
//   * opening the Prompt Lens dock (when the page has one) changes no fixed
//     panel's size or place: the dock covers the panels, never moves them.
//
// With --dock it then watches the open dock while traffic streams (several
// samples) and fails unless:
//   1. every FIXED panel keeps its box and place while the dock is open,
//   2. the dock bar is 40 px (44 below 768 px) and the filter line is one row
//      of 38 px (52 below 768 px: "Filters (n)", the search, the count),
//   3. every section box is at most 200 px (160 below 768 px), and a box that
//      reached its cap never grows (a box under the cap may grow up to it),
//   4. no dock container (dock, bar, filter line, list, request screen, section
//      boxes) reaches past the viewport sideways,
//   5. the page text never contains "cobs2" (an internal name),
//   6. below 768 px the open dock covers the whole viewport (a sheet).
// It first waits (up to 90 s) for a request to appear in the dock; on a phone
// the list fills the sheet, so the script taps the newest row to reach the
// request screen before sampling the boxes.
//
// Manual check (it needs a running warden with a few engines and some traffic,
// which CI does not have). See docs/live-stats-spec.md, "Real-browser check".
//
//   node scripts/stats-layout-check.mjs <base-url> <admin|key> <width> [samples] [outdir] [--dock]
//   env: LMW_USER, LMW_PASS (admin view) or LMW_KEY (key view)
//
// Prints one JSON line per sample and a verdict; exits 1 on any overflow or
// height change.

import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const argv = process.argv.slice(2);
const DOCK = argv.includes("--dock");
const [base, view = "admin", width = "1440", samples = "4", outdir = ""] = argv.filter((a) => a !== "--dock");
if (!base) {
  console.error("usage: stats-layout-check.mjs <base-url> <admin|key> <width> [samples] [outdir] [--dock]");
  process.exit(2);
}

const FIXED = [
  "stats-header",
  "scope-bar",
  "notice-line",
  "now-card",
  "tile-ttft",
  "tile-decode",
  "tile-prefill",
  "tile-prompt",
  "tile-duration",
  "latency-chart",
  "latency-plot",
  "tokens-card",
  "cache-card",
  "engine-details",
];
// The scope bar scrolls sideways by design when tabs do not fit.
const X_SCROLL_OK = new Set(["scope-bar"]);

async function measure(page) {
  return page.evaluate(
    ({ ids, xok }) => {
      const out = { boxes: {}, tops: {}, overflow: [], vw: window.innerWidth };
      for (const id of ids) {
        const el = document.querySelector(`[data-testid="${id}"]`);
        if (!el) continue;
        const r = el.getBoundingClientRect();
        out.boxes[id] = [Math.round(r.width), Math.round(r.height)];
        out.tops[id] = Math.round(r.top + window.scrollY);
        if (el.scrollHeight > el.clientHeight + 1) out.overflow.push(`${id}: content ${el.scrollHeight}px > box ${el.clientHeight}px`);
        if (!xok.includes(id) && el.scrollWidth > el.clientWidth + 1)
          out.overflow.push(`${id}: content ${el.scrollWidth}px wide > box ${el.clientWidth}px`);
        if (r.right > window.innerWidth + 1) out.overflow.push(`${id}: right edge ${Math.round(r.right)} > viewport ${window.innerWidth}`);
      }
      const win = document.querySelector('[aria-label="Window"]');
      if (win) {
        const r = win.getBoundingClientRect();
        if (r.right > window.innerWidth + 1) out.overflow.push(`window selector cut off at ${Math.round(r.right)}px`);
      }
      return out;
    },
    { ids: FIXED, xok: [...X_SCROLL_OK] },
  );
}

const PHONE = +width < 768;
const BAR_PX = PHONE ? 44 : 40;
const LINE_PX = PHONE ? 52 : 38;
const BOX_MAX = PHONE ? 160 : 200;
const DOCK_PARTS = ["prompt-lens-dock", "prompt-lens-bar", "prompt-lens-filter-line", "prompt-lens-list", "prompt-lens-screen", "request-screen"];

/** One sample of the open dock: the numbers checks 2-6 need. */
async function measureDock(page) {
  return page.evaluate(
    ({ parts }) => {
      const q = (id) => document.querySelector(`[data-testid="${id}"]`);
      const h = (el) => (el ? Math.round(el.getBoundingClientRect().height) : null);
      const line = q("prompt-lens-filter-line");
      const dock = q("prompt-lens-dock");
      const out = {
        bar: h(q("prompt-lens-bar")),
        line: h(line),
        lineScrollH: line ? line.scrollHeight : null,
        boxes: {},
        wide: [],
        cobs2: /cobs2/i.test(document.body.innerText),
        dock: dock ? (({ top, bottom, left, right }) => ({ top, bottom, left, right }))(dock.getBoundingClientRect()) : null,
        vw: window.innerWidth,
        vh: window.innerHeight,
      };
      for (const el of document.querySelectorAll('[data-testid^="section-box-"]')) {
        out.boxes[el.dataset.testid] = h(el);
      }
      const els = [...parts.map(q), ...document.querySelectorAll('[data-testid^="section-box-"]')];
      for (const el of els) {
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (r.width && (r.right > window.innerWidth + 1 || r.left < -1))
          out.wide.push(`${el.dataset.testid}: ${Math.round(r.left)}..${Math.round(r.right)} outside 0..${window.innerWidth}`);
      }
      return out;
    },
    { parts: DOCK_PARTS },
  );
}

const browser = await chromium.launch({ headless: true, channel: process.env.PW_CHANNEL ?? "chrome" });
let failed = false;
try {
  const ctx = await browser.newContext({ viewport: { width: +width, height: 900 }, colorScheme: "dark" });
  const page = await ctx.newPage();
  await page.goto(`${base}/ui/login/`);
  if (view === "admin") {
    await page.fill("input[name=username]", process.env.LMW_USER ?? "admin");
    await page.fill("input[name=password]", process.env.LMW_PASS ?? "");
    await page.click("button[type=submit]");
    await page.waitForURL(/\/stats/);
    await page.waitForSelector("[data-testid=now-card]");
  } else {
    await page.click("#login-tab-key");
    await page.fill("input[name=api-key]", process.env.LMW_KEY ?? "");
    await page.click("button[type=submit]");
    await page.waitForURL(/\/key/);
    await page.goto(`${base}/ui/key/stats/`);
    await page.waitForSelector("[data-testid=key-stats-page]");
  }
  await page.waitForTimeout(6000);
  const series = [];
  for (let i = 0; i < +samples; i++) {
    const m = await measure(page);
    series.push(m);
    console.log(JSON.stringify({ i, ...m }));
    if (i === 0 && outdir) await page.screenshot({ path: path.join(outdir, `check-${view}-${width}.png`), fullPage: true });
    // Switch window and tab between samples: sizes must not follow either.
    const ws = ["6h", "24h", "1h"];
    const btn = await page.$(`[aria-label=Window] button:has-text("${ws[i % ws.length]}")`);
    if (btn) await btn.click();
    const tabs = await page.$$("[data-testid=scope-tab]");
    if (tabs.length > 1) await tabs[(i + 1) % tabs.length].click();
    await page.waitForTimeout(5000);
  }
  // The Prompt Lens dock: open it from its bar; nothing above may move.
  const moved = [];
  const dockFail = [];
  const bar = await page.$('[data-testid=prompt-lens-dock][data-open=false] [data-testid=prompt-lens-bar] button');
  if (bar) {
    const before = await measure(page);
    await bar.click();
    await page.waitForSelector("[data-testid=prompt-lens-dock][data-open=true]");
    await page.waitForTimeout(1500);
    const open = await measure(page);
    series.push(open);
    console.log(JSON.stringify({ i: "dock-open", ...open }));
    if (outdir) await page.screenshot({ path: path.join(outdir, `check-${view}-${width}-dock.png`) });
    for (const id of Object.keys(before.tops)) if (before.tops[id] !== open.tops[id]) moved.push(id);
    if (DOCK) dockFail.push(...(await watchDock(page, before)));
  } else if (DOCK) {
    dockFail.push("--dock: no Prompt Lens dock on this page (turn Prompt Lens on in Settings, or VW_PROMPT_LENS_ENABLED)");
  } else {
    console.log("no Prompt Lens dock on this page (switch off): dock check skipped");
  }
  const overflow = [...new Set(series.flatMap((s) => s.overflow))];
  const ids = Object.keys(series[0].boxes);
  const changed = ids.filter((id) => new Set(series.map((s) => JSON.stringify(s.boxes[id]))).size > 1);
  for (const id of moved) if (!changed.includes(id)) changed.push(`${id} (moved when the dock opened)`);
  console.log(`verdict ${view} ${width}px: overflow ${overflow.length ? overflow.join("; ") : "none"} · changed ${changed.join(", ") || "none"}`);
  if (DOCK) console.log(`verdict dock ${view} ${width}px: ${dockFail.length ? dockFail.join("; ") : "ok"}`);
  failed = overflow.length > 0 || changed.length > 0 || dockFail.length > 0;
  if (outdir) fs.writeFileSync(path.join(outdir, `check-${view}-${width}.json`), JSON.stringify(series, null, 1));
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);

/** --dock: sample the open dock while traffic streams; return the failures. */
async function watchDock(page, before) {
  const fail = [];
  // Wait for a request that starts while the dock is open (up to 90 s): one
  // already running at open may have left the replay without its start, and
  // then has no model, so the Model preset hides it. On a desktop Follow
  // newest selects it; on a phone the list fills the sheet, so tap it.
  const row = await page
    .waitForSelector('[data-testid^="prompt-lens-row-"]', { timeout: 90000 })
    .catch(() => null);
  if (!row) fail.push("no request in the dock list (send some traffic)");
  else if (PHONE) {
    await row.click();
    await page.waitForTimeout(800);
  } else {
    await page.waitForSelector('[data-testid^="section-box-"]', { timeout: 10000 }).catch(() => {});
  }
  let prev = null;
  const n = Math.max(3, +samples);
  for (let i = 0; i < n; i++) {
    const fixed = await measure(page);
    const d = await measureDock(page);
    console.log(JSON.stringify({ i: `dock-${i}`, fixed: fixed.boxes, dock: d }));
    if (outdir) await page.screenshot({ path: path.join(outdir, `check-${view}-${width}-dock-${i}.png`) });
    for (const id of Object.keys(before.boxes)) {
      if (JSON.stringify(before.boxes[id]) !== JSON.stringify(fixed.boxes[id]) || before.tops[id] !== fixed.tops[id])
        fail.push(`1 ${id} moved or resized while the dock is open (sample ${i})`);
    }
    if (d.bar !== BAR_PX) fail.push(`2 bar ${d.bar}px, want ${BAR_PX}`);
    if (d.line !== LINE_PX || d.lineScrollH > LINE_PX) fail.push(`2 filter line ${d.line}px (scroll ${d.lineScrollH}), want one ${LINE_PX}px row`);
    for (const [id, hgt] of Object.entries(d.boxes)) {
      if (hgt > BOX_MAX) fail.push(`3 ${id} ${hgt}px > ${BOX_MAX}`);
      const was = prev?.boxes[id];
      if (was != null && was >= BOX_MAX && hgt > was) fail.push(`3 ${id} grew ${was} -> ${hgt}px at its cap`);
    }
    for (const w of d.wide) fail.push(`4 ${w}`);
    if (d.cobs2) fail.push("5 the page text says cobs2");
    if (PHONE && d.dock && (d.dock.top > 1 || d.dock.bottom < d.vh - 1 || d.dock.left > 1 || d.dock.right < d.vw - 1))
      fail.push(`6 the open dock does not cover the viewport (${JSON.stringify(d.dock)})`);
    prev = d;
    await page.waitForTimeout(3000);
  }
  if (!Object.keys(prev?.boxes ?? {}).length) fail.push("3 no section box on screen (no request selected?)");
  return [...new Set(fail)];
}
