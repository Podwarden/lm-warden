// Connect a client (plan docs/superpowers/plans/2026-10-04-connect-clients.md).
// Pure helpers over `GET /api/connect/clients`: template rendering, the page
// strip, the requirements checklist, the Verify request and its explanation.
// No fetch, no SWR, no React — the page owns those.

import { getPublicBaseUrl } from "@/lib/public-url";
import type { components } from "@/lib/api-types.generated";
import { KEY_PLACEHOLDER, type DecisionOut, type TokenListPage } from "@/lib/router";

// ── Types (§2) ─────────────────────────────────────────────────────────────
// Aliases of the API's schemas (`GET /api/connect/clients`), so a backend
// change surfaces here as a type error.

type Schemas = components["schemas"];

export type ConnectClientsOut = Schemas["ConnectClientsOut"];
export type ConnectClient = Schemas["ConnectClient"];
export type ConnectMode = Schemas["ConnectMode"];
export type ConnectFile = Schemas["ConnectFile"];
export type ConnectVerify = Schemas["ConnectVerify"];
export type ConnectVerified = Schemas["ConnectVerified"];
export type ConnectDocs = Schemas["ConnectDocs"];
export type ConnectRequirement = Schemas["ConnectRequirement"];
export type ConnectRouter = Schemas["ConnectRouter"];
export type ConnectRule = Schemas["ConnectRule"];
export type ConnectModel = Schemas["ConnectModel"];
export type ConnectGroup = ConnectClient["group"];
export type ConnectProtocol = ConnectClient["protocol"];
export type ConnectSupport = ConnectClient["support"];
export type ConnectFileLanguage = ConnectFile["language"];

type TokenItem = TokenListPage["items"][number];

// ── Templates ──────────────────────────────────────────────────────────────

export const MODEL_PLACEHOLDER = "your-served-model-name";
export const DEFAULT_HEADER = "X-LMWarden-Key";
/** `{{context}}` when the model's window is unknown (the template says it is a guess). */
export const CONTEXT_FALLBACK = 32768;

export interface TemplateVars {
  origin?: string | null;
  key?: string | null;
  model?: string | null;
  header?: string | null;
  context?: number | null;
  /** verify bodies only; defaults to `model`. */
  verify_model?: string | null;
}

const PLACEHOLDER_RE = /\{\{(origin|key|model|header|context|verify_model)\}\}/g;

/** Pure string replace of the five placeholders (+ `verify_model`). A var
 *  that is missing renders as the server would: `vw_YOUR_KEY`,
 *  `your-served-model-name`, `32768`. Anything else is left as written. */
export function renderTemplate(template: string, vars: TemplateVars): string {
  const model = vars.model || MODEL_PLACEHOLDER;
  const values: Record<string, string> = {
    origin: vars.origin || getPublicBaseUrl(null),
    key: vars.key || KEY_PLACEHOLDER,
    model,
    header: vars.header || DEFAULT_HEADER,
    context: String(vars.context ?? CONTEXT_FALLBACK),
    verify_model: vars.verify_model || model,
  };
  return template.replace(PLACEHOLDER_RE, (_, name: string) => values[name]);
}

/** The files to show for a model: a `requires_tools` variant appears only when
 *  it matches the model's tool support (no model → the no-tools variant). */
export function filesFor(mode: ConnectMode, model: ConnectModel | null): ConnectFile[] {
  const tools = model?.supports_tools ?? false;
  return mode.files.filter((f) => f.requires_tools == null || f.requires_tools === tools);
}

// ── Keys and models ────────────────────────────────────────────────────────

/** A key still authenticates. */
export function isLiveKey(t: TokenItem): boolean {
  return !t.is_revoked && !t.is_expired && !t.is_paused;
}

export function isLoaded(m: ConnectModel): boolean {
  return m.status === "loaded";
}

// ── Page strip (§5.5) ──────────────────────────────────────────────────────

export type ConnectStateKind = "loading" | "error" | "no_model" | "no_key" | "tokens_error" | "ok";
export type ConnectStripAction = "retry" | "models" | "create_key" | "retry_tokens";

export interface ConnectStrip {
  tone: "neutral" | "amber" | "danger";
  role: "status" | "alert";
  text: string;
  action: ConnectStripAction;
}

export interface ConnectStateInput {
  clients: ConnectClientsOut | undefined;
  tokens: TokenListPage | undefined;
  errors: { clients?: boolean; tokens?: boolean };
}

export function deriveConnectState({ clients, tokens, errors }: ConnectStateInput): {
  kind: ConnectStateKind;
  strip: ConnectStrip | null;
} {
  if (errors.clients) {
    return {
      kind: "error",
      strip: { tone: "danger", role: "alert", text: "Couldn't load the client catalogue. Nothing was changed.", action: "retry" },
    };
  }
  if (!clients) return { kind: "loading", strip: null };
  if (!clients.models.some(isLoaded)) {
    return {
      kind: "no_model",
      strip: {
        tone: "amber",
        role: "alert",
        text: "No model is loaded. Snippets use a placeholder model name until one is.",
        action: "models",
      },
    };
  }
  if (errors.tokens) {
    return {
      kind: "tokens_error",
      strip: { tone: "neutral", role: "status", text: "Couldn't load your keys; snippets keep the placeholder.", action: "retry_tokens" },
    };
  }
  if (tokens && !tokens.items.some(isLiveKey)) {
    return {
      kind: "no_key",
      strip: { tone: "neutral", role: "status", text: "You have no API key yet. Create one to fill the snippets.", action: "create_key" },
    };
  }
  return { kind: "ok", strip: null };
}

// ── Requirements (§5.3) ────────────────────────────────────────────────────

export type CheckState = "ok" | "warn" | "fail" | "unknown";

export interface RequirementCheck {
  id: string;
  state: CheckState;
  text: string;
  /** The measured value, e.g. "qwen: 65,536 tokens (needs ≥ 49,152)". */
  detail?: string;
}

export interface RequirementContext {
  model: ConnectModel | null;
  /** The picked key's list row; null when none is picked. */
  key: Pick<TokenItem, "anthropic_relay"> | null;
  router: ConnectRouter | null;
  /** Every model, for router mode (min context is checked per rule target). */
  models?: ConnectModel[];
}

const fmt = (n: number) => n.toLocaleString("en-US");

function isRouterMode(mode: ConnectMode): boolean {
  return mode.id === "router" || mode.needs_relay_key;
}

function contextCheck(
  r: ConnectRequirement,
  min: number,
  mode: ConnectMode,
  ctx: RequirementContext,
): RequirementCheck {
  const base = { id: r.id, text: r.text };
  const needs = `needs ≥ ${fmt(min)}`;
  let model = ctx.model;
  if (isRouterMode(mode) && ctx.router) {
    // Rules decide the model: the smallest known target is the binding one.
    const targets = ctx.router.rules
      .map((rule) => (ctx.models ?? []).find((m) => m.served_name === rule.target_served_name))
      .filter((m): m is ConnectModel => !!m);
    if (targets.length === 0) return { ...base, state: "unknown", detail: "no rule targets a model" };
    const known = targets.filter((m) => m.context_window != null);
    if (known.length === 0) return { ...base, state: "unknown", detail: `${targets[0].served_name}: context unknown` };
    model = known.reduce((a, b) => (b.context_window! < a.context_window! ? b : a));
  }
  if (!model) return { ...base, state: "unknown", detail: "no model picked" };
  if (model.context_window == null) return { ...base, state: "unknown", detail: `${model.served_name}: context unknown` };
  const detail = `${model.served_name}: ${fmt(model.context_window)} tokens (${needs})`;
  // Below the minimum is a warning, never a block: the model stays usable for short sessions.
  return { ...base, state: model.context_window >= min ? "ok" : "warn", detail };
}

export function requirementChecks(
  client: ConnectClient,
  mode: ConnectMode,
  ctx: RequirementContext,
): RequirementCheck[] {
  const out: RequirementCheck[] = [];
  const router = isRouterMode(mode);
  for (const r of client.requirements) {
    const base = { id: r.id, text: r.text };
    // `modes` null = every mode; the router-only flags never apply outside router mode.
    if (r.modes && !r.modes.includes(mode.id)) continue;
    if ((r.router_on || r.relay_key) && !router) continue;
    if (r.min_context_tokens != null) {
      out.push(contextCheck(r, r.min_context_tokens, mode, ctx));
    } else if (r.needs_tools) {
      const m = ctx.model;
      out.push(
        !m
          ? { ...base, state: "unknown", detail: "no model picked" }
          : m.supports_tools == null
            ? { ...base, state: "unknown", detail: `${m.served_name}: tool calling unknown` }
            : {
              ...base,
              state: m.supports_tools ? "ok" : "fail",
              detail: `${m.served_name}: ${m.supports_tools ? "tool calling" : "no tool calling"}`,
            },
      );
    } else if (r.router_on) {
      out.push(
        !ctx.router
          ? { ...base, state: "unknown" }
          : { ...base, state: ctx.router.enabled ? "ok" : "fail", detail: ctx.router.enabled ? "routing is on" : "routing is off" },
      );
    } else if (r.relay_key) {
      out.push(
        !ctx.key
          ? { ...base, state: "unknown", detail: "no key picked" }
          : {
              ...base,
              state: ctx.key.anthropic_relay ? "ok" : "fail",
              detail: ctx.key.anthropic_relay ? "this key may relay" : "this key may not relay",
            },
      );
    } else {
      out.push({ ...base, state: "unknown" });
    }
  }
  return out;
}

// ── Verify (§5.4) ──────────────────────────────────────────────────────────

export const VERIFY_TIMEOUT_MS = 15_000;

/** Router mode: a Claude id the first enabled rule matches; else the picked model. */
export function verifyModelFor(
  _client: ConnectClient,
  mode: ConnectMode,
  router: ConnectRouter | null,
  selected: string | null,
): string {
  if (isRouterMode(mode)) {
    const ex = router?.rules.find((r) => r.example_model)?.example_model;
    if (ex) return ex;
  }
  return selected || MODEL_PLACEHOLDER;
}

function renderDeep(value: unknown, vars: TemplateVars): unknown {
  if (typeof value === "string") return renderTemplate(value, vars);
  if (Array.isArray(value)) return value.map((v) => renderDeep(v, vars));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [renderTemplate(k, vars), renderDeep(v, vars)]));
  }
  return value;
}

/** The request the Verify button sends. Same origin as this tab (no CORS,
 *  CSRF-exempt under /v1), cookies omitted, headers exactly as the template
 *  names them — so no `Authorization` unless it says so. Call `done()` when
 *  the response is in, to clear the 15 s timer. */
export function buildVerifyRequest(
  verify: ConnectVerify,
  vars: TemplateVars,
  origin: string = window.location.origin,
): { url: string; init: RequestInit; done: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), VERIFY_TIMEOUT_MS);
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(verify.headers)) headers[renderTemplate(k, vars)] = renderTemplate(v, vars);
  return {
    // The path is never templated: the key cannot reach the URL.
    url: `${origin.replace(/\/+$/, "")}${verify.path}`,
    init: {
      method: verify.method,
      headers,
      body: JSON.stringify(renderDeep(verify.body, vars)),
      credentials: "omit",
      cache: "no-store",
      signal: controller.signal,
    },
    done: () => clearTimeout(timer),
  };
}

export interface VerifyContext {
  latencyMs?: number;
  /** The model the request asked for. */
  model?: string;
  /** The origin the tab tried to reach. */
  origin?: string;
  /** Scrubbed from any server text before it is shown. */
  key?: string | null;
  /** What a good answer looks like (`ConnectVerify.expect`). */
  expect?: ConnectVerify["expect"];
}

export interface VerifyExplanation {
  tone: "ok" | "danger";
  text: string;
}

function rec(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** The server's message from an Anthropic, OpenAI or FastAPI error body. */
function serverMessage(body: unknown): string | null {
  if (typeof body === "string") return body || null;
  const b = rec(body);
  if (!b) return null;
  const err = rec(b.error);
  if (typeof err?.message === "string") return err.message;
  if (typeof b.error === "string") return b.error;
  if (typeof b.detail === "string") return b.detail;
  const det = rec(b.detail);
  if (typeof det?.message === "string") return det.message;
  if (typeof b.message === "string") return b.message;
  return null;
}

/** The text of a Responses API answer: the `output_text` parts of its `message` items. */
function responsesText(b: Record<string, unknown>): string {
  if (!Array.isArray(b.output)) return "";
  return b.output
    .map((item) => rec(item))
    .filter((item) => item?.type === "message" && Array.isArray(item.content))
    .flatMap((item) => (item!.content as unknown[]).map((c) => rec(c)))
    .filter((c) => c?.type === "output_text" && typeof c.text === "string")
    .map((c) => c!.text as string)
    .join("");
}

function answerText(body: unknown, protocol: ConnectProtocol): string {
  const b = rec(body);
  if (!b) return "";
  if (Array.isArray(b.output)) return responsesText(b);
  if (protocol === "anthropic" && Array.isArray(b.content)) {
    return b.content
      .map((c) => rec(c))
      .filter((c) => c?.type === "text" && typeof c.text === "string")
      .map((c) => c!.text as string)
      .join("");
  }
  const choice = Array.isArray(b.choices) ? rec(b.choices[0]) : null;
  const content = rec(choice?.message)?.content;
  return typeof content === "string" ? content : typeof choice?.text === "string" ? choice.text : "";
}

const clip = (s: string, n = 80) => (s.length > n ? `${s.slice(0, n)}…` : s);

/** One sentence for a Verify outcome. Never echoes headers or the key. */
export function explainVerify(
  status: number | "abort" | "network",
  body: unknown,
  protocol: ConnectProtocol,
  ctx: VerifyContext = {},
): VerifyExplanation {
  const scrub = (s: string) => (ctx.key ? s.split(ctx.key).join("vw_…") : s).replace(/vw_[A-Za-z0-9_-]{6,}/g, "vw_…");
  const danger = (text: string): VerifyExplanation => ({ tone: "danger", text: scrub(text) });
  if (status === "abort") return danger("No answer in 15 s.");
  if (status === "network") return danger(`Couldn't reach ${ctx.origin ?? "the warden"} from this tab.`);
  const msg = serverMessage(body);
  if (status >= 200 && status < 300) {
    const model = typeof rec(body)?.model === "string" ? (rec(body)!.model as string) : ctx.model;
    const parts = [`${status} in ${Math.round(ctx.latencyMs ?? 0)} ms`];
    if (model) parts.push(`answered by ${model}`);
    const answer = answerText(body, protocol).trim();
    // A 200 that is not the expected shape (a Responses reply with no message)
    // is not a pass: the client would get nothing to show.
    if (ctx.expect === "openai_responses" && !answer) {
      return danger("The warden answered 200 but the response has no message text.");
    }
    if (answer) parts.push(`'${clip(answer)}'`);
    return { tone: "ok", text: scrub(parts.join(" · ")) };
  }
  if (status === 401) return danger("Key not accepted: unknown, expired or revoked.");
  if (status === 403) {
    const raw = `${msg ?? ""} ${JSON.stringify(body ?? "")}`;
    if (raw.includes("relay_not_allowed")) return danger("This key may not relay to Anthropic — tick May relay on the key.");
    if (raw.includes("token_not_allowed")) return danger("This key's model allow list excludes this model.");
    return danger(msg ? `Refused (403): ${msg}` : "Refused (403).");
  }
  if (status === 404) {
    return danger(`Model ${ctx.model ?? MODEL_PLACEHOLDER} is not served here (not loaded, or not a served name).`);
  }
  if (status === 502 || status === 529 || status === 503) {
    return danger(`The local model failed: ${msg ?? `HTTP ${status}`}`);
  }
  return danger(msg ?? `The warden answered ${status}.`);
}

// ── Deep links (§3) ────────────────────────────────────────────────────────

/** `?tool=&mode=` → a valid pick. `corrected` asks the page to rewrite the URL
 *  (router.replace) because a value named something that does not exist. */
export function parseConnectParams(
  params: URLSearchParams,
  clients: ConnectClient[],
): { tool: string | null; mode: string | null; corrected: boolean } {
  if (clients.length === 0) return { tool: null, mode: null, corrected: false };
  const wantTool = params.get("tool");
  const wantMode = params.get("mode");
  const found = wantTool ? clients.find((c) => c.id === wantTool) : undefined;
  const client = found ?? clients[0];
  let corrected = !!wantTool && !found;
  const modeHit = wantMode ? client.modes.find((m) => m.id === wantMode) : undefined;
  if (wantMode && !modeHit && found) corrected = true;
  const mode = modeHit ?? client.modes[0];
  return { tool: client.id, mode: mode?.id ?? null, corrected };
}

// ── Page presentation (§3, §6) ─────────────────────────────────────────────

export type SupportTone = "ok" | "info" | "idle" | "danger";

/** The support pill: always a word, never colour alone. */
export const SUPPORT: Record<ConnectSupport, { label: string; tone: SupportTone }> = {
  full: { label: "Full", tone: "ok" },
  local_only: { label: "Local models only", tone: "info" },
  documented: { label: "Documented, not run", tone: "idle" },
  unsupported: { label: "Not supported", tone: "danger" },
};

export type ToolSection = "anthropic" | "openai" | "editors" | "unsupported";

export const SECTION_LABEL: Record<ToolSection, string> = {
  anthropic: "Anthropic protocol",
  openai: "OpenAI protocol",
  editors: "Editors",
  unsupported: "Not supported",
};

/** The tool list's section: what the tool can reach, not where it runs. */
export function toolSection(c: ConnectClient): ToolSection {
  if (c.support === "unsupported") return "unsupported";
  if (c.group === "editors") return "editors";
  return c.protocol;
}

/** Sections in order of first appearance; tools keep catalogue order inside. */
export function groupTools(clients: ConnectClient[]): { section: ToolSection; label: string; clients: ConnectClient[] }[] {
  const out: { section: ToolSection; label: string; clients: ConnectClient[] }[] = [];
  for (const c of clients) {
    const s = toolSection(c);
    let g = out.find((x) => x.section === s);
    if (!g) {
      g = { section: s, label: SECTION_LABEL[s], clients: [] };
      out.push(g);
    }
    g.clients.push(c);
  }
  return out;
}

const COMMENT: Partial<Record<ConnectFileLanguage, string>> = {
  bash: "#",
  toml: "#",
  yaml: "#",
  python: "#",
  typescript: "//",
};

/** "# key: laptop (vw_ab12…)" in the file's own comment syntax; null where
 *  the format has none (json, fields). */
export function keyComment(language: ConnectFileLanguage, name: string, preview: string): string | null {
  const c = COMMENT[language];
  return c ? `${c} key: ${name} (${preview})` : null;
}

/** The smallest hard (non-soft) context minimum that applies to this mode. */
export function hardMinContext(client: ConnectClient, mode: ConnectMode | null): number | null {
  const mins = client.requirements
    .filter((r) => r.min_context_tokens != null && !r.soft && (!r.modes || !mode || r.modes.includes(mode.id)))
    .map((r) => r.min_context_tokens as number);
  return mins.length ? Math.min(...mins) : null;
}

/** Why Verify cannot run, or null. Order: what the operator cannot fix here first. */
export function verifyBlocker(mode: ConnectMode, router: ConnectRouter | null, hasKey: boolean): string | null {
  if (isRouterMode(mode)) {
    if (router && !router.enabled) return "Routing is off, so there is nothing to test in router mode. Turn it on on the Router page.";
    if (router && !router.rules.some((r) => r.example_model)) {
      return "No enabled rule names a Claude model to test with. Add a rule on the Router page.";
    }
  }
  if (!hasKey) return "Paste or create a key to send a test request.";
  return null;
}

/** The decision log's verdict on a Verify request, as a clause for the result line. */
export function describeRoute(
  d: Pick<DecisionOut, "route" | "reason"> & Partial<Pick<DecisionOut, "model_out">> | undefined,
): string | null {
  if (!d) return null;
  const why = d.reason ? ` (${d.reason})` : "";
  switch (d.route) {
    case "local":
      // The answer echoes the Claude id it was asked for; the log names the model that ran.
      return d.model_out ? `route: local on ${d.model_out}` : "route: local";
    case "fallback":
      return `fell back to Anthropic${why}`;
    case "passthrough":
      return "passed through to Anthropic";
    case "refused":
      return `refused${why}`;
    default:
      return `route: ${d.route}${why}`;
  }
}

export { isRouterMode };
