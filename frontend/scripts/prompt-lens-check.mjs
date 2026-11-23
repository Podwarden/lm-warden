// Real-browser check of Prompt Lens against the static export, with mocked
// APIs and a mocked live stream (no warden needed). Build first:
//
//   LMW_UI_EXPORT=1 npx next build
//   node scripts/prompt-lens-check.mjs <outdir> [--only=<check>]
//
// It serves out/ at /ui, answers /api/* from a small in-memory fake (the
// section endpoint returns what the fake stream has sent so far), replaces
// EventSource with a fake the script drives, and opens headless Chrome
// (Playwright channel "chrome") at 1440 and 375 px. Checks:
//   popup-live     the section pop-up keeps streaming after it opened
//                  (thinking, reply, tool calls), Following stays on; and
//                  frame rate at ~250 events/s with the pop-up open
//   counters       the x of the text after live counters is constant over
//                  100 updates (pop-up size pill, section box header, facts)
//   page           /ui/prompt-lens and /ui/key/prompt-lens fill the screen,
//                  carry the menu entry and the breadcrumb
// Prints one JSON line per check, saves screenshots to <outdir>, exits 1 on
// any failure.

import { chromium } from "playwright";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(here, "..", "out");
const argv = process.argv.slice(2);
const outdir = path.resolve(argv.find((a) => !a.startsWith("--")) ?? ".");
const only = (argv.find((a) => a.startsWith("--only=")) ?? "").slice(7);
fs.mkdirSync(outdir, { recursive: true });

// --- static server -------------------------------------------------------
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".woff2": "font/woff2", ".json": "application/json", ".txt": "text/plain", ".ico": "image/x-icon" };
const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://x");
  let p = decodeURIComponent(u.pathname);
  if (!p.startsWith("/ui")) {
    res.writeHead(404).end();
    return;
  }
  p = p.slice(3) || "/";
  let f = path.join(OUT, p);
  if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, "index.html");
  if (!fs.existsSync(f) && fs.existsSync(f + ".html")) f += ".html";
  if (!fs.existsSync(f)) f = path.join(OUT, "404.html");
  res.writeHead(200, { "Content-Type": TYPES[path.extname(f)] ?? "application/octet-stream" });
  fs.createReadStream(f).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}`;

// --- the fake stream, in the page ------------------------------------------
// window.__lens.emit(ev) sends one event to every open Prompt Lens stream and
// records the text the "server" holds (window.__lens.text[req][kind]).
const INIT = `
(() => {
  const lens = { all: [], text: {}, calls: {} };
  window.__lens = lens;
  const Native = window.EventSource;
  class FakeES {
    constructor(url) {
      this.url = String(url);
      this.readyState = 0;
      this.closed = false;
      if (!this.url.includes('prompt-lens')) {
        // other streams: stay connecting forever, quietly
        return;
      }
      lens.all.push(this);
      setTimeout(() => { this.readyState = 1; this.onopen && this.onopen(); }, 0);
    }
    addEventListener() {}
    removeEventListener() {}
    close() { this.closed = true; this.readyState = 2; }
  }
  window.EventSource = FakeES;
  const KIND = { reasoning: 'thinking', content: 'reply' };
  lens.serverOnly = (req, kind, text) => {
    const t = (lens.text[req] ??= { thinking: '', reply: '' });
    t[kind] += text;
  };
  lens.emit = (ev) => {
    if (ev.type === 'delta') {
      const t = (lens.text[ev.req_id] ??= { thinking: '', reply: '' });
      const k = KIND[ev.channel];
      if (k) t[k] += ev.text;
      if (ev.channel === 'tool_call') {
        const c = (lens.calls[ev.req_id] ??= []);
        const i = ev.index ?? 0;
        c[i] ??= { index: i, call_id: ev.call_id ?? null, name: ev.name ?? '', args: '' };
        if (ev.name) c[i].name = ev.name;
        c[i].args += ev.text;
      }
    }
    const data = JSON.stringify(ev);
    for (const s of lens.all) if (!s.closed && s.onmessage) s.onmessage({ data });
  };
})();
`;

let T = Math.floor(Date.now() / 1000) - 30;
const ts = () => (T += 0.004);
const startEv = (req, o = {}) => ({
  type: "request_start", req_id: req, ts: ts(), token_id: "tok-a", token_label: "ci-runner", request_id: `rq-${req}`,
  model: "qwen3.8-27b", served_name: "qwen3.8-27b", stream: true, session_id: "sess0001aaaa", turn: 3,
  client_ip: "10.0.0.7", prompt: "Summarise the build log", seq: 1, ...o,
});

async function api(route, view) {
  const req = route.request();
  const u = new URL(req.url());
  const p = u.pathname;
  const ok = (body) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  if (p === "/api/auth/refresh") return ok({ access_token: "jwt", expires_in: 3600 });
  if (p === "/api/csrf") return ok({ csrf: "c" });
  if (p === "/api/auth/sse-ticket" || p.endsWith("/prompt-lens/ticket")) return ok({ ticket: "t" });
  if (p === "/api/admin/prompt-lens/status") return ok({ enabled: true, default_on: true });
  if (p.endsWith("/prompt-lens/kept")) return ok({ kept: [] });
  if (p === "/api/me/key") return ok({ id: "tok-a", name: "ci-runner", prompt_lens_view_available: true, keep_captures: false });
  if (p === "/api/auth/me" || p === "/api/me") return ok({ username: "admin", role: "admin" });
  const m = p.match(/\/prompt-lens\/requests\/([^/]+)\/section\/([a-z_]+)$/);
  if (m) {
    const [, id, kind] = m;
    const page = route.request().frame().page();
    const st = await page.evaluate(([i]) => ({ t: window.__lens.text[i] ?? { thinking: "", reply: "" }, c: window.__lens.calls[i] ?? [] }), [id]);
    // As on a live server: deltas sent after the snapshot reach the page
    // before the response does.
    const CH = { thinking: "reasoning", reply: "content", tool_calls: "tool_call" };
    if (CH[kind]) {
      await page.evaluate(([i, ch]) => {
        for (let n = 0; n < 6; n++) window.__lens.emit({ type: "delta", req_id: i, ts: Date.now() / 1000, token_id: "tok-a", channel: ch, text: ch === "tool_call" ? "in-flight/" : `in-flight ${n} `, ...(ch === "tool_call" ? { index: 0 } : {}) });
      }, [id, CH[kind]]);
      await new Promise((r) => setTimeout(r, 120));
    }
    const from = Number(u.searchParams.get("from") ?? 0);
    const base = { kind, seq: view === "admin" ? 5 : null, live: true, dropped_chars: 0, dropped_calls: 0, kept: false };
    if (kind === "thinking" || kind === "reply") {
      const text = st.t[kind];
      return ok({ ...base, offset: Math.min(from, text.length), text: text.slice(from) });
    }
    if (kind === "tool_calls") return ok({ ...base, offset: 0, calls: st.c.filter(Boolean) });
    return ok({ ...base, offset: 0, messages: [] });
  }
  return ok({});
}

async function openPage(browser, width, view, url) {
  const ctx = await browser.newContext({ viewport: { width, height: width < 768 ? 812 : 900 }, deviceScaleFactor: 1 });
  await ctx.addInitScript(INIT);
  if (view === "key") {
    await ctx.addInitScript(() => {
      sessionStorage.setItem("vw-sessions-exp", String(Date.now() + 3600_000));
      sessionStorage.setItem("vw-sessions-token", "key-signin");
    });
  }
  await ctx.route("**/api/**", (r) => api(r, view));
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  if (process.env.LENS_DEBUG) {
    page.on("pageerror", (e) => console.error("pageerror", String(e).slice(0, 300)));
    page.on("request", (r) => r.url().includes("/api/") && console.error("api", r.url().replace(BASE, "")));
  }
  await page.goto(`${BASE}${url}`, { waitUntil: "networkidle" });
  if (process.env.LENS_DEBUG) await page.screenshot({ path: path.join(outdir, `debug-${view}-${width}.png`) });
  return { ctx, page, errors };
}

const results = [];
function report(name, pass, info) {
  results.push({ name, pass });
  console.log(JSON.stringify({ check: name, pass, info }));
}

const emit = (page, ev) => page.evaluate((e) => window.__lens.emit(e), ev);
const emitMany = (page, evs) => page.evaluate((es) => es.forEach((e) => window.__lens.emit(e)), evs);
const frame = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

/** Open the Prompt Lens view on `url` and select request `req` there. */
async function lensOn(page, width, req) {
  // Rows in the list; on a phone, tap the row to reach the screen.
  await page.getByTestId(`prompt-lens-row-${req}`).click();
}

const delta = (req, channel, text, o = {}) => ({ type: "delta", req_id: req, ts: ts(), token_id: "tok-a", channel, text, ...o });

// --- checks ------------------------------------------------------------------

async function checkPopupLive(browser, width, view, url) {
  const { ctx, page, errors } = await openPage(browser, width, view, url);
  const tag = `${view}-${width}`;
  await page.waitForFunction(() => window.__lens.all.length > 0);
  if (process.env.LENS_MIDSTREAM) {
    await page.evaluate(() => window.__lens.serverOnly("r1", "thinking", "Early thinking the dock never saw.\n".repeat(50)));
    await emit(page, { ...startEv("r1"), restored: true });
  } else await emit(page, startEv("r1"));
  await emit(page, delta("r1", "reasoning", "Reading the log.\n" + Array.from({ length: 120 }, (_, i) => `line ${i} of the earlier thinking, long enough to wrap a little on a phone`).join("\n") + "\n"));
  await frame(page);
  await lensOn(page, width, "r1");
  for (const [kind, channel, title] of [["thinking", "reasoning", "Thinking"], ["reply", "content", "Reply"], ["tool_calls", "tool_call", "Tool calls"]]) {
    if (channel !== "reasoning") {
      const first = channel === "tool_call" ? delta("r1", channel, '{"path": "', { index: 0, name: "read_file", call_id: "c1" }) : delta("r1", channel, Array.from({ length: 120 }, (_, i) => `reply line ${i}`).join("\n") + "\n");
      await emit(page, first);
      await frame(page);
    }
    await page.getByRole("button", { name: `Open ${title} in a large view` }).click();
    const popup = page.getByTestId("popup-body");
    await popup.waitFor();
    await page.waitForTimeout(150);
    const before = (await popup.textContent()).length;
    const lens = [];
    for (let i = 0; i < 40; i++) {
      await emit(page, delta("r1", channel, channel === "tool_call" ? `seg${i}/` : `word${i} more text here.\n`, channel === "tool_call" ? { index: 0 } : {}));
      await frame(page);
      if (i % 10 === 9) lens.push((await popup.textContent()).length);
    }
    // Token-sized deltas at ~60/s for 3 s, as a model streams.
    const samples = await page.evaluate(async ([ch]) => {
      const b = document.querySelector('[data-testid="popup-body"]');
      const out = [];
      for (let i = 0; i < 180; i++) {
        window.__lens.emit({ type: "delta", req_id: "r1", ts: Date.now() / 1000, token_id: "tok-a", channel: ch, text: ch === "tool_call" ? "ab" : (i % 12 === 11 ? " tok\n" : " tok"), ...(ch === "tool_call" ? { index: 0 } : {}) });
        await new Promise((r) => setTimeout(r, 16));
        if (i % 30 === 29) out.push(b.textContent.length);
      }
      return out;
    }, [channel]);
    lens.push(...samples);
    const after = (await popup.textContent()).length;
    const following = await page.getByTestId("popup-body").getAttribute("data-following");
    const grew = lens.every((n, i) => i === 0 || n > lens[i - 1]) && after > before;
    const lastWord = channel === "tool_call" ? "seg39/" : "word39";
    const visible = await page.evaluate((w) => {
      const b = document.querySelector('[data-testid="popup-body"]');
      return b.scrollHeight - b.clientHeight - b.scrollTop < 30 && b.textContent.includes(w);
    }, lastWord);
    await page.screenshot({ path: path.join(outdir, `popup-live-${kind}-${tag}.png`) });
    report(`popup-live ${kind} ${tag}`, grew && visible && following === "true", { before, lens, after, following, visible });
    await page.keyboard.press("Escape");
    await frame(page);
  }
  // fps at ~250 events/s with the thinking pop-up open (8 s)
  await page.getByRole("button", { name: "Open Thinking in a large view" }).click();
  await page.getByTestId("popup-body").waitFor();
  const fps = await page.evaluate(async () => {
    let frames = 0;
    let run = true;
    const tick = () => {
      frames++;
      if (run) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    let n = 0;
    const t0 = performance.now();
    const iv = setInterval(() => {
      for (let i = 0; i < 5; i++) {
        n++;
        window.__lens.emit({ type: "delta", req_id: "r1", ts: Date.now() / 1000, token_id: "tok-a", channel: "reasoning", text: `tok${n} ` + (n % 40 === 0 ? "\n" : "") });
      }
    }, 20);
    await new Promise((r) => setTimeout(r, 6000));
    clearInterval(iv);
    run = false;
    const s = (performance.now() - t0) / 1000;
    return { fps: frames / s, eventsPerS: n / s };
  });
  report(`popup-fps ${tag}`, fps.fps >= 50, fps);
  await page.keyboard.press("Escape");
  report(`popup-no-page-errors ${tag}`, errors.length === 0, errors.slice(0, 3));
  await ctx.close();
}


/** The x of each element `sel` names (a function run in the page returning
 *  elements), across `n` deltas: every x must stay the same. */
async function watchX(page, n, makeDelta, probes) {
  const xs = Object.fromEntries(Object.keys(probes).map((k) => [k, []]));
  for (let i = 0; i < n; i++) {
    await emit(page, makeDelta(i));
    await frame(page);
    const got = await page.evaluate((ps) => {
      const out = {};
      for (const [k, src] of Object.entries(ps)) {
        // eslint-disable-next-line no-new-func
        const el = new Function(`return (${src})()`)();
        out[k] = el ? Math.round(el.getBoundingClientRect().x * 100) / 100 : null;
      }
      return out;
    }, probes);
    for (const k of Object.keys(probes)) xs[k].push(got[k]);
  }
  return Object.fromEntries(
    Object.entries(xs).map(([k, v]) => {
      const nums = v.filter((x) => x !== null);
      return [k, { found: nums.length, spread: nums.length ? Math.max(...nums) - Math.min(...nums) : null, first: nums[0], last: nums[nums.length - 1] }];
    }),
  );
}

async function checkCounters(browser, width, view, url) {
  const { ctx, page, errors } = await openPage(browser, width, view, url);
  const tag = `${view}-${width}`;
  await page.waitForFunction(() => window.__lens.all.length > 0);
  await emit(page, startEv("c1"));
  await emit(page, delta("c1", "reasoning", "x".repeat(950)));
  await frame(page);
  await lensOn(page, width, "c1");
  // Deltas of uneven size: the counters cross 1,000 and 10,000 chars and
  // change most digits on every update.
  const sizes = (i) => 1 + ((i * 37) % 211);
  const inline = await watchX(page, 100, (i) => delta("c1", "reasoning", "y".repeat(sizes(i)) + (i % 7 === 0 ? "\n" : "")), {
    "after thinking size": `() => document.querySelector('[data-testid="section-size-thinking"]')?.nextElementSibling`,
    "after facts 'out'": `() => [...document.querySelectorAll('[data-testid="rs-facts"] > span')].find((s) => s.textContent.startsWith("now") || s.textContent.startsWith("client"))`,
  });
  await page.screenshot({ path: path.join(outdir, `counters-inline-${tag}.png`) });
  for (const [k, v] of Object.entries(inline)) report(`counters ${k} ${tag}`, v.found > 0 && v.spread < 0.5, v);
  await page.getByRole("button", { name: "Open Thinking in a large view" }).click();
  await page.getByTestId("popup-body").waitFor();
  await page.waitForTimeout(200);
  const popup = await watchX(page, 100, (i) => delta("c1", "reasoning", "z".repeat(sizes(i + 3))), {
    "after pop-up size": `() => document.querySelector('[data-testid="popup-size"]')?.nextElementSibling`,
  });
  await page.screenshot({ path: path.join(outdir, `counters-popup-${tag}.png`) });
  for (const [k, v] of Object.entries(popup)) report(`counters ${k} ${tag}`, v.found > 0 && v.spread < 0.5, v);
  await page.keyboard.press("Escape");
  report(`counters-no-page-errors ${tag}`, errors.length === 0, errors.slice(0, 3));
  await ctx.close();
}

async function checkPage(browser, width, view, url) {
  const { ctx, page, errors } = await openPage(browser, width, view, url);
  const tag = `${view}-${width}`;
  const phone = width < 768;
  const el = page.getByTestId("prompt-lens-page");
  await el.waitFor();
  await page.waitForFunction(() => window.__lens.all.length > 0);
  await emit(page, startEv("p1", { session_id: "sessA0000001" }));
  await emit(page, delta("p1", "reasoning", "Reading the build log.\n".repeat(20)));
  await emit(page, startEv("p2", { session_id: "sessB0000002", turn: 1 }));
  await emit(page, delta("p2", "content", "The build failed at step 3.\n".repeat(8)));
  await frame(page);
  const vp = page.viewportSize();
  const box = await el.boundingBox();
  const above = await page.evaluate(() => {
    const nav = document.querySelector("header");
    const crumb = document.querySelector('nav[aria-label="Breadcrumb"]');
    // The breadcrumb strip is the trail's row (its parent box).
    const strip = crumb ? crumb.closest("div") ?? crumb : null;
    return { nav: nav ? nav.getBoundingClientRect().bottom : 0, crumb: strip ? strip.getBoundingClientRect().bottom : 0 };
  });
  const fills =
    Math.abs(box.x) < 1 && Math.abs(box.width - vp.width) < 1 && Math.abs(box.y + box.height - vp.height) < 1 && box.y <= Math.max(above.nav, above.crumb) + 1;
  const scroll = await page.evaluate(() => ({ x: document.documentElement.scrollWidth - window.innerWidth, y: document.documentElement.scrollHeight - window.innerHeight }));
  report(`page fills the screen ${tag}`, fills && scroll.x <= 0 && scroll.y <= 0, { box, vp, above, scroll });
  // Desktop: list and request screen side by side; phone: the list alone.
  const list = await page.getByTestId("prompt-lens-list").boundingBox();
  const scr = phone ? null : await page.getByTestId("prompt-lens-screen").boundingBox();
  if (phone) report(`page list fills the phone ${tag}`, Math.abs(list.width - vp.width) < 1, { list });
  else report(`page list beside request screen ${tag}`, list.x + list.width <= scr.x + 1 && Math.abs(scr.x + scr.width - vp.width) < 1 && scr.height > vp.height * 0.6, { list, scr });
  // Menu and breadcrumbs.
  if (view === "admin") {
    const crumbs = await page.evaluate(() => {
      const n = document.querySelector('nav[aria-label="Breadcrumb"]');
      return n ? [...n.querySelectorAll("li, a, span")].map((x) => x.textContent.trim()).filter(Boolean) : null;
    });
    const crumbText = await page.locator('nav[aria-label="Breadcrumb"]').innerText().catch(() => "");
    report(`breadcrumb Home > Prompt Lens ${tag}`, /Home\s*(›|>|\n)?\s*Prompt Lens\s*$/.test(crumbText.trim()) && (crumbText.match(/Prompt Lens/g) ?? []).length === 1, { crumbText, crumbs });
    await page.getByRole("button", { name: "Open menu" }).click();
    const item = page.getByRole("menuitem", { name: "Prompt Lens" });
    const href = await item.getAttribute("href");
    const current = await item.getAttribute("aria-current");
    await page.screenshot({ path: path.join(outdir, `page-menu-${tag}.png`) });
    report(`menu entry ${tag}`, href === "/ui/prompt-lens/" || href === "/ui/prompt-lens", { href, current });
    await page.keyboard.press("Escape");
  } else {
    const navCurrent = await page.locator('nav[aria-label="Key"] a[aria-current="page"]').innerText();
    const crumb = await page.getByTestId("key-breadcrumb").innerText();
    report(`key nav and breadcrumb ${tag}`, navCurrent.trim() === "Prompt Lens" && /Home\s*›\s*Prompt Lens/.test(crumb), { navCurrent, crumb });
  }
  await page.screenshot({ path: path.join(outdir, `page-${tag}.png`) });
  // Pick a request: the address carries it; on a phone the back gesture
  // returns to the list.
  await page.getByTestId("prompt-lens-row-p1").click();
  await frame(page);
  const search = await page.evaluate(() => location.search);
  report(`pick writes ?req= ${tag}`, /req=rq-p1/.test(search), { search });
  await page.screenshot({ path: path.join(outdir, `page-request-${tag}.png`) });
  if (phone) {
    await page.goBack();
    await frame(page);
    const back = await page.getByTestId("prompt-lens-list").isVisible().catch(() => false);
    const still = page.url().includes("prompt-lens");
    report(`phone back gesture returns to the list ${tag}`, back && still, { url: page.url() });
  }
  report(`page-no-page-errors ${tag}`, errors.length === 0, errors.slice(0, 3));
  await ctx.close();

  // A deep link opens the request (the Stats eye).
  const d = await openPage(browser, width, view, `${url}?req=rq-d1`);
  await d.page.waitForFunction(() => window.__lens.all.length > 0);
  await emit(d.page, startEv("d1"));
  await frame(d.page);
  const cur = await d.page.getByTestId("prompt-lens-row-d1").getAttribute("aria-current").catch(() => null);
  const shown = await d.page.getByTestId("request-screen").isVisible().catch(() => false);
  report(`deep link ?req= opens the request ${tag}`, shown && (phone || cur === "true"), { cur, shown });
  await d.ctx.close();
}

const viewsFor = (name) => [
  [1440, "admin", process.env.LENS_ADMIN_URL ?? "/ui/prompt-lens/"],
  [375, "admin", process.env.LENS_ADMIN_URL ?? "/ui/prompt-lens/"],
  [1440, "key", process.env.LENS_KEY_URL ?? "/ui/key/prompt-lens/"],
  [375, "key", process.env.LENS_KEY_URL ?? "/ui/key/prompt-lens/"],
].map((v) => [name, ...v]);

const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const checks = { page: checkPage, "popup-live": checkPopupLive, counters: checkCounters };
  for (const [name, fn] of Object.entries(checks)) {
    if (only && only !== name) continue;
    const vf = (process.env.LENS_VIEWS ?? "").split(",").filter(Boolean);
    for (const [, width, view, url] of viewsFor(name)) {
      if (vf.length && !vf.includes(`${view}-${width}`)) continue;
      try {
        await fn(browser, width, view, url);
      } catch (e) {
        report(`${name} ${view}-${width}`, false, String(e).slice(0, 400));
      }
    }
  }
} finally {
  await browser.close();
  server.close();
}
const failed = results.filter((r) => !r.pass);
console.log(JSON.stringify({ verdict: failed.length ? "FAIL" : "PASS", failed: failed.map((f) => f.name) }));
process.exit(failed.length ? 1 : 0);
