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
//   sessions       ?view=sessions&session=... opens the scorecard: the score
//                  tiles (two a row on a phone), the top 3 fixes and the turn
//                  strip, no horizontal scroll at 1440 and 375, "not scored"
//                  tiles carry no number
//   trends         ?view=trends (admin: &key=...) draws the chart with the
//                  version markers and a gap for an unscored week, the
//                  "Did the last change help?" verdict and the findings by
//                  version, no horizontal page scroll at 1440 and 375
//   filters        the one filter bar sits on Requests, Sessions and Trends
//                  (admin: Key picker with every named key, even one with no
//                  traffic; key holder: none); Trends with no key offers an
//                  inline "Pick a key" select; no horizontal scroll at 1440 and 375
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

const SESSION_A = "s_" + "a".repeat(32);
const SESSION_B = "s_" + "b".repeat(32);
const sc = (layer, value, o = {}) => ({ layer, scored: true, value, lo: value - 7, hi: value + 6, confidence: "high", turns: 24, inputs: {}, excluded_rules: [], ...o });
const noSc = (layer, reason) => ({ layer, scored: false, reason, turns: 2 });
const fnd = (rule, layer, turns, tokens, extra = {}) => ({ rule, layer, turns, counts: {}, tool: null, cause: null, saving: { tokens, prefill_s: null, decode_s: null, money: null }, confidence: "high", muted: false, part: null, ...extra });
const CARD = (short) => ({
  session: { harness: "other", model: "qwen3.8-27b", route: "local", turns: short ? 3 : 24, duration_s: 1080, version: null },
  scores: short
    ? { overall: noSc("overall", "1 of 2 layers scored"), E: noSc("E", "2 of 3 continuing turns"), C: sc("C", 90, { turns: 3 }), R: noSc("R", "3 of 5 assistant turns") }
    : { overall: sc("overall", 62, { lo: 55, hi: 68, confidence: "medium", inputs: { from: ["E", "C", "R"] } }), E: sc("E", 48, { inputs: { ideal: 185000, lost: 96200 } }), C: sc("C", 71, { inputs: { waste: 89000, prompt: 612000 } }), R: sc("R", 74, { confidence: "medium", inputs: { turns: 24, bad: 5, completion: 4800, waste_out: 430 } }) },
  top: short ? [] : [fnd("E-01", "E", [2, 3, 9], 61400, { counts: { line: 2, class: 1, turns: 3, lost: 61400, sessions: 1 }, cause: "sys_changed", part: "sys" }), fnd("C-01", "C", [1, 2, 3], 139000, { counts: { unused: 7, tools: 12, tokens: 5800 } }), fnd("R-04", "R", [7, 9, 12, 14], 2100, { counts: { bad: 4, calls: 6, code: 3 }, tool: "fa000003", cause: "schema_undocumented", confidence: "medium" })],
  notice: null,
  findings: short ? [] : [fnd("E-01", "E", [2, 3, 9], 61400, { counts: { line: 2, class: 1, turns: 3, lost: 61400, sessions: 1 }, cause: "sys_changed", part: "sys" }), fnd("C-01", "C", [1, 2, 3], 139000, { counts: { unused: 7, tools: 12, tokens: 5800 } }), fnd("R-04", "R", [7, 9, 12, 14], 2100, { counts: { bad: 4, calls: 6, code: 3 }, tool: "fa000003", cause: "schema_undocumented", confidence: "medium" })],
  not_scored: short ? { E: "2 of 3 continuing turns", R: "3 of 5 assistant turns" } : {},
  turns: Array.from({ length: short ? 3 : 24 }, (_, i) => ({ n: i + 1, req_id: `req-${i + 1}`, reused: i ? 6000 + 400 * (i - 1) : 0, recomputed: i === 8 ? 8800 : 0, new: i ? 400 : 6000, markers: i === 8 ? ["E-01", "R-04"] : [] })),
  cost: { prompt_tokens: 612000, reused_tokens: 309000, recomputed_tokens: 96000, fixable_lost_tokens: 61400, card: null, actual_input: null, saving: null, saving_pct: null },
  score_version: 1, weights: { E: 0.45, C: 0.15, R: 0.4 }, review: null, names: { fa000003: "search_docs" },
});
const WEEK = 7 * 86400;
const T_NOW = Math.floor(Date.now() / 1000);
const med = (o) => (o === null ? { overall: null, E: null, C: null, R: null } : { overall: o, E: o - 3, C: o + 5, R: o + 8 });
const TRENDS = {
  by: "week", model: "qwen3.8-27b", models: ["qwen3.8-27b"], weights: { E: 0.45, C: 0.15, R: 0.4 }, score_version: 1,
  points: [44, 46, 47, null, 52, 50, 53, 62].map((o, i) => ({ bucket: `w${i}`, n: o === null ? 2 : 14 + i, medians: med(o), band: o === null ? null : [o - 7, o + 8], start: T_NOW - (8 - i) * WEEK })),
  versions: [
    { id: "aaaa1111", label: "v1", since: T_NOW - 8 * WEEK, last_at: T_NOW - 5 * WEEK, n: 52, medians: med(46), rules: { "E-01": 49, "E-02": 18, "C-01": 52 }, resolved: [], new: [] },
    { id: "bbbb2222", label: "v2", since: T_NOW - 4 * WEEK, last_at: T_NOW - 2 * WEEK, n: 31, medians: med(51), rules: { "E-01": 28, "C-01": 31 }, resolved: ["E-02"], new: [] },
    { id: "cccc3333", label: "v3", since: T_NOW - 2 * WEEK, last_at: T_NOW, n: 23, medians: med(62), rules: { "E-01": 21, "C-01": 14, "R-04": 9 }, resolved: [], new: ["R-04"] },
  ],
};
const COMPARE = { model: "qwen3.8-27b", a: "bbbb2222", b: "cccc3333", a_label: "v2", b_label: "v3", n_a: 31, n_b: 23, diffs: { overall: 11, E: 3, C: 24, R: 6 }, verdict: "better_clear", like_for_like: true, note: null, resolved: [], new: ["R-04"], p: 0.004 };
const LIST = { sessions: [
  { session_id: SESSION_A, model: "qwen3.8-27b", route: "local", harness: "other", turns: 24, started: Date.now() / 1000 - 4000, last_at: Date.now() / 1000 - 3000, overall: 62, e: 48, c: 71, r: 74, confidence: "medium", reasons: {}, score_version: 1, key_name: "alpha-key", key_id: "tok-a" },
  { session_id: SESSION_B, model: "qwen3.8-27b", route: "local", harness: "other", turns: 3, started: Date.now() / 1000 - 9000, last_at: Date.now() / 1000 - 8000, overall: null, e: null, c: 90, r: null, confidence: null, reasons: { E: "2 of 3 continuing turns", R: "3 of 5 assistant turns" }, score_version: 1, key_name: "alpha-key", key_id: "tok-a" },
], next_cursor: null };

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
  if (p === "/api/tokens") return ok({ items: [{ id: "tok-a", name: "alpha-key" }, { id: "tok-b", name: "beta-key" }] });
  if (p === "/api/me/key") return ok({ id: "tok-a", name: "ci-runner", prompt_lens_view_available: true, keep_captures: false });
  if (p === "/api/auth/me" || p === "/api/me") return ok({ username: "admin", role: "admin" });
  if (/\/prompt-lens\/sessions$/.test(p)) return ok(LIST);
  if (/\/prompt-lens\/trends$/.test(p)) return ok(TRENDS);
  if (/\/prompt-lens\/compare$/.test(p)) return ok(COMPARE);
  const sm = p.match(/\/prompt-lens\/sessions\/([^/]+)\/scorecard$/);
  if (sm) return ok(CARD(sm[1] === SESSION_B));
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
    page.on("pageerror", (e) => console.error("pageerror", String(e.stack ?? e).slice(0, 900)));
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

async function checkSessions(browser, width, view, url) {
  const tag = `${view}-${width}`;
  const phone = width < 768;
  const { ctx, page, errors } = await openPage(browser, width, view, `${url}?view=sessions${process.env.BIS_NO_SESSION ? "" : `&session=${SESSION_A}`}`);
  const card = page.getByTestId("scorecard");
  await card.waitFor();
  const tiles = await Promise.all(["overall", "E", "C", "R", "review"].map(async (k) => page.getByTestId(`score-tile-${k}`).boundingBox()));
  const nums = await page.getByTestId("score-value").allInnerTexts();
  report(`sessions scorecard shows the scores ${tag}`, nums.join(",") === "62,48,71,74" && tiles.every(Boolean), { nums });
  // Phone: two tiles a row (E and C share a row; the overall spans the width).
  const [overall, e, c, r] = tiles;
  if (phone) report(`sessions tiles two a row ${tag}`, Math.abs(e.y - c.y) < 2 && r.y > c.y && overall.width > e.width * 1.6, { overall, e, c, r });
  else report(`sessions tiles in a row ${tag}`, Math.abs(e.y - c.y) < 2 && Math.abs(c.y - r.y) < 2, { e, c, r });
  const scroll = await page.evaluate(() => ({ x: document.documentElement.scrollWidth - window.innerWidth, pane: (() => { const p = document.querySelector('[data-testid="lens-sessions-pane"]'); return p ? p.scrollWidth - p.clientWidth : 0; })() }));
  report(`sessions no horizontal scroll ${tag}`, scroll.x <= 0 && scroll.pane <= 0, scroll);
  const fixes = await page.getByTestId("fix-card").count();
  const bars = await page.getByTestId("turn-bar").count();
  report(`sessions top 3 fixes and the turn strip ${tag}`, fixes === 3 && bars === 24, { fixes, bars });
  await page.screenshot({ path: path.join(outdir, `sessions-${tag}.png`) });
  // A session below its minimums: "not scored: <reason>" and no number.
  await ctx.close();
  const b = await openPage(browser, width, view, `${url}?view=sessions&session=${SESSION_B}`);
  await b.page.getByTestId("scorecard").waitFor();
  const e2 = b.page.getByTestId("score-tile-E");
  const text = await e2.innerText();
  const hasNum = await e2.getByTestId("score-value").count();
  report(`sessions not scored carries the reason, no number ${tag}`, /not scored: 2 of 3 continuing turns/.test(text) && hasNum === 0, { text });
  await b.page.screenshot({ path: path.join(outdir, `sessions-not-scored-${tag}.png`) });
  // The list itself: title with the key name, no model, no horizontal scroll.
  const l = await openPage(browser, width, view, `${url}?view=sessions`);
  await l.page.getByTestId("lens-session-row").first().waitFor();
  const rows = await l.page.getByTestId("lens-session-row").allInnerTexts();
  const lscroll = await l.page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  report(`sessions list rows read in words ${tag}`, rows.length === 2 && rows.every((t) => !/qwen/.test(t) && /Efficiency/.test(t)) && (view === "key" || /alpha-key/.test(rows[0])) && lscroll <= 0, { rows, lscroll });
  await l.page.screenshot({ path: path.join(outdir, `sessions-list-${tag}.png`) });
  await l.ctx.close();
  report(`sessions no page errors ${tag}`, errors.length === 0 && b.errors.length === 0, [...errors, ...b.errors].slice(0, 3));
  await b.ctx.close();
}

async function checkTrends(browser, width, view, url) {
  const tag = `${view}-${width}`;
  const sep = url.includes("?") ? "&" : "?";
  const { ctx, page, errors } = await openPage(browser, width, view, `${url}${sep}view=trends${view === "admin" ? "&key=tok-a" : ""}`);
  await page.getByTestId("trend-chart").waitFor();
  const markers = await page.getByTestId("version-marker").count();
  const gaps = await page.getByTestId("trend-gap").count();
  report(`trends chart with version markers and a gap ${tag}`, markers === 2 && gaps === 1, { markers, gaps });
  const verdict = await page.getByTestId("verdict").innerText();
  report(`trends verdict card ${tag}`, /Better, clear · 23 vs 31 sessions/.test(verdict), { verdict });
  const rows = await page.getByTestId("version-rule-row").count();
  report(`trends findings by version ${tag}`, rows === 4, { rows });
  const scroll = await page.evaluate(() => ({ x: document.documentElement.scrollWidth - window.innerWidth }));
  report(`trends no horizontal page scroll ${tag}`, scroll.x <= 0, scroll);
  const text = await page.getByTestId("lens-trends-view").innerText();
  report(`trends a key sees no fleet figure ${tag}`, view === "admin" || !/rank|fleet|leaderboard/i.test(text), {});
  await page.screenshot({ path: path.join(outdir, `trends-${tag}.png`) });
  report(`trends no page errors ${tag}`, errors.length === 0, errors.slice(0, 3));
  await ctx.close();
}

async function checkFilters(browser, width, view, url) {
  const tag = `${view}-${width}`;
  const phone = width < 768;
  const sep = url.includes("?") ? "&" : "?";
  for (const v of ["requests", "sessions", "trends"]) {
    const { ctx, page, errors } = await openPage(browser, width, view, v === "requests" ? url : `${url}${sep}view=${v}`);
    const bar = page.getByTestId("prompt-lens-filter-line");
    await bar.waitFor();
    const bars = await page.getByTestId("prompt-lens-filter-line").count();
    report(`filters one bar on ${v} ${tag}`, bars === 1, { bars });
    if (!phone) {
      const key = await bar.getByRole("button", { name: /^Key/ }).count();
      report(`filters Key picker only for the admin on ${v} ${tag}`, view === "admin" ? key === 1 : key === 0, { key });
      if (view === "admin") {
        await bar.getByRole("button", { name: /^Key/ }).click();
        const names = await page.getByRole("menuitemcheckbox").allInnerTexts();
        report(`filters Key picker lists every named key on ${v} ${tag}`, names.some((n) => /beta-key/.test(n)) && names.some((n) => /alpha-key/.test(n)), { names });
        await page.keyboard.press("Escape");
      }
    } else {
      const n = await bar.getByRole("button", { name: /^Filters/ }).count();
      report(`filters phone button on ${v} ${tag}`, n === 1, { n });
    }
    if (v === "trends" && view === "admin") {
      const pick = await page.getByRole("combobox", { name: "Pick a key" }).count();
      report(`filters Trends offers an inline key picker ${tag}`, pick === 1, { pick });
    }
    const scroll = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    report(`filters no horizontal page scroll on ${v} ${tag}`, scroll <= 0, { scroll });
    await page.screenshot({ path: path.join(outdir, `filters-${v}-${tag}.png`) });
    report(`filters no page errors on ${v} ${tag}`, errors.length === 0, errors.slice(0, 3));
    await ctx.close();
  }
}

const viewsFor = (name) => [
  [1440, "admin", process.env.LENS_ADMIN_URL ?? "/ui/prompt-lens/"],
  [375, "admin", process.env.LENS_ADMIN_URL ?? "/ui/prompt-lens/"],
  [1440, "key", process.env.LENS_KEY_URL ?? "/ui/key/prompt-lens/"],
  [375, "key", process.env.LENS_KEY_URL ?? "/ui/key/prompt-lens/"],
].map((v) => [name, ...v]);

const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const checks = { page: checkPage, sessions: checkSessions, trends: checkTrends, filters: checkFilters, "popup-live": checkPopupLive, counters: checkCounters };
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
