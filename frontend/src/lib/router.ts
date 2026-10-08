import type { components } from "@/lib/api-types.generated";

export type RuleOut = components["schemas"]["RuleOut"];
export type RuleIn = components["schemas"]["RuleIn"];
export type RulePatch = components["schemas"]["RulePatch"];
export type RulesOut = components["schemas"]["RulesOut"];
export type RouterSettingsOut = components["schemas"]["RouterSettingsOut"];
export type RouterSettingsPatch = components["schemas"]["RouterSettingsPatch"];

export const KEY_PLACEHOLDER = "vw_YOUR_KEY";

/** Shell block for router mode. The subscription login stays in place; only
 *  the base URL and our own key header are added. */
export function envSnippet(baseUrl: string, headerName: string, keyPlaceholder: string): string {
  return [
    `export ANTHROPIC_BASE_URL=${baseUrl}`,
    `export ANTHROPIC_CUSTOM_HEADERS="${headerName}: ${keyPlaceholder}"`,
    "claude",
  ].join("\n");
}

export function settingsJsonSnippet(
  baseUrl: string,
  headerName: string,
  keyPlaceholder: string,
): string {
  return JSON.stringify(
    {
      env: {
        ANTHROPIC_BASE_URL: baseUrl,
        ANTHROPIC_CUSTOM_HEADERS: `${headerName}: ${keyPlaceholder}`,
      },
    },
    null,
    2,
  );
}

/** Local-only mode (no Anthropic account): every Claude alias maps to one
 *  served model and the warden key is the auth token. */
export function localOnlySnippet(baseUrl: string, servedName: string, key: string = KEY_PLACEHOLDER): string {
  return [
    `export ANTHROPIC_BASE_URL=${baseUrl}`,
    `export ANTHROPIC_AUTH_TOKEN=${key}`,
    `export ANTHROPIC_MODEL=${servedName}`,
    `export ANTHROPIC_DEFAULT_OPUS_MODEL=${servedName}`,
    `export ANTHROPIC_DEFAULT_SONNET_MODEL=${servedName}`,
    `export ANTHROPIC_DEFAULT_HAIKU_MODEL=${servedName}`,
  ].join("\n");
}

// ── Router overview: derived state (router page redesign, plan §4.1) ───────
// Pure functions only. The page reads the router's public endpoints and these
// turn them into one answer: what state is it in, what is wrong, what next.

export type RouterStatsOut = components["schemas"]["RouterStatsOut"];
export type RuleStats = components["schemas"]["RuleStats"];
export type TargetStats = components["schemas"]["TargetStats"];
export type DecisionOut = components["schemas"]["DecisionOut"];
export type TokenListPage = components["schemas"]["TokenListPage"];

/** Whether any key may relay to Anthropic. `relayKnown` is false while the
 *  token list is loading, failed, or was truncated with none found — the UI
 *  then makes no claim either way. */
export interface RelayInfo {
  relayKeys: number;
  relayKnown: boolean;
  relayLoading: boolean;
}

/** Text with inline code: rendered as `<code>` by the strip. */
export type StripPart = string | { code: string };
export type StripTone = "neutral" | "ok" | "amber" | "danger";
export type RouterStateKind =
  | "loading" | "error" | "setup" | "off" | "danger" | "amber" | "waiting" | "ok";

export type StripAction =
  | { kind: "retry"; label: string }
  | { kind: "turn_on"; label: string }
  | { kind: "open_model"; label: string; href: string; modelId: string }
  | { kind: "see_fallbacks"; label: string; href: string }
  | { kind: "pause_rule"; label: string; ruleId: string }
  | { kind: "edit_rule"; label: string; ruleId: string }
  | { kind: "create_relay_key"; label: string }
  | { kind: "api_keys"; label: string; href: string }
  | { kind: "connect"; label: string; href: string };

export type IssueCode =
  | "stats_error" | "breaker" | "target_missing" | "target_unloaded" | "no_relay" | "recent_fallbacks";

export interface RouterIssue {
  code: IssueCode;
  tone: "danger" | "amber";
  title: StripPart[];
  body: StripPart[];
  actions: StripAction[];
  ruleId?: string;
  modelId?: string;
}

export interface RouterStrip {
  tone: StripTone;
  /** alert for danger/amber (content changes in place, so it announces);
   *  status for the calm states. */
  role: "status" | "alert";
  title: StripPart[];
  body: StripPart[];
  actions: StripAction[];
  /** waiting: the indicator dot pulses (not under reduced motion). */
  pulse?: boolean;
  /** "On with no rules: …" — a second, quieter line. */
  note?: string;
  /** Issues below the headline ("and N more"). */
  more: RouterIssue[];
}

export interface RouterState {
  kind: RouterStateKind;
  /** Every applicable issue, in priority order; issues[0] is the headline
   *  when kind is error/danger/amber. */
  issues: RouterIssue[];
  /** null while settings (or, with nothing else to say, stats) load. */
  strip: RouterStrip | null;
}

export interface RouterStateInput {
  settings: RouterSettingsOut | undefined;
  rules: RuleOut[] | undefined;
  stats: RouterStatsOut | undefined;
  /** The stats (or decisions) fetch failed; `stats` may hold the last good data. */
  statsError?: boolean;
  /** Newest first, as `GET /api/router/decisions` returns them. */
  decisions: DecisionOut[] | undefined;
  relay: RelayInfo | undefined;
  now: number;
}

export const ACTIVITY_HREF = "/router/stats";
export const FALLBACKS_HREF = "/router/stats?route=fallback";
const RECENT_WINDOW = 20;
const NOT_LOCAL = new Set(["fallback", "refused", "error"]);

/** Concatenate strip parts to plain text (tests, aria-labels, titles). */
export function partsText(parts: StripPart[]): string {
  return parts.map((p) => (typeof p === "string" ? p : p.code)).join("");
}

export function modelHref(modelId: string): string {
  return `/models/${encodeURIComponent(modelId)}`;
}

export function fmtInt(n: number): string {
  return n.toLocaleString("en-US");
}

/** "76 %", "9.6 %", "0.1 %", "0 %". */
export function fmtPct(p: number): string {
  if (!(p > 0)) return "0 %";
  if (p >= 10) return `${Math.round(p)} %`;
  return `${Math.max(0.1, Math.round(p * 10) / 10)} %`;
}

/** "4 s ago" / "3 min ago" / "2 h ago" / "3 d ago"; "just now" for < 1 s or a
 *  timestamp ahead of this clock; "—" when unparseable. */
export function fmtAgo(iso: string, now: number): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const s = Math.floor((now - t) / 1000);
  if (s < 1) return "just now";
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

/** Local wall-clock "HH:MM" for "since 09:12". */
export function fmtClock(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

/** "41 s", "10 min", "2 h" until `iso`; null when past or unparseable. */
export function fmtIn(iso: string | null, now: number): string | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const s = Math.round((t - now) / 1000);
  if (s <= 0) return null;
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  return `${Math.round(s / 3600)} h`;
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

// ── Pattern preview ────────────────────────────────────────────────────────

/** Claude model ids Claude Code is known to ask for; the rule dialog's
 *  pattern preview ("Matches e.g. …") picks from these. Static on purpose. */
export const KNOWN_CLAUDE_IDS: readonly string[] = [
  "claude-haiku-4-5",
  "claude-haiku-4-5-20251001",
  "claude-3-5-haiku-latest",
  "claude-3-5-haiku-20241022",
  "claude-sonnet-4-5",
  "claude-sonnet-4-5-20250929",
  "claude-sonnet-4-20250514",
  "claude-3-7-sonnet-latest",
  "claude-opus-4-1",
  "claude-opus-4-1-20250805",
  "claude-opus-4-20250514",
];

/** Client mirror of the server's `fnmatchcase` for the characters a pattern
 *  may contain (`[A-Za-z0-9._:*?-]`): `*` any run, `?` one character,
 *  everything else literal, case-sensitive, whole string. */
export function globMatches(pattern: string, s: string): boolean {
  const re = pattern
    .split("")
    .map((c) => (c === "*" ? ".*" : c === "?" ? "." : c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    .join("");
  return new RegExp(`^${re}$`, "s").test(s);
}

// ── Traffic ────────────────────────────────────────────────────────────────

export type TrafficKey = "local" | "passthrough" | "fallback" | "refused";

export interface TrafficSegment {
  key: TrafficKey;
  label: string;
  n: number;
  /** 0..100, unrounded; format with fmtPct. */
  pct: number;
}

const SEGMENTS: { key: TrafficKey; label: string }[] = [
  { key: "local", label: "Local" },
  { key: "passthrough", label: "Anthropic (pass-through)" },
  { key: "fallback", label: "Fell back to Anthropic" },
  { key: "refused", label: "Refused or error" },
];

/** The four-way split of `stats.totals`; "Refused or error" folds both. */
export function trafficSplit(totals: Record<string, number>): {
  total: number;
  segments: TrafficSegment[];
} {
  const get = (k: string) => totals[k] ?? 0;
  const counts: Record<TrafficKey, number> = {
    local: get("local"),
    passthrough: get("passthrough"),
    fallback: get("fallback"),
    refused: get("refused") + get("error"),
  };
  const total = counts.local + counts.passthrough + counts.fallback + counts.refused;
  return {
    total,
    segments: SEGMENTS.map(({ key, label }) => ({
      key,
      label,
      n: counts[key],
      pct: total > 0 ? (counts[key] / total) * 100 : 0,
    })),
  };
}

export interface RuleTraffic {
  local: number;
  fallback: number;
  refused: number;
  total: number;
  /** "1,284 local · 6 fell back · 2 refused" — zero parts omitted. */
  label: string;
}

/** Per-rule counters, or null when there are none (render "—"). */
export function ruleTraffic(stats: RouterStatsOut | undefined, ruleId: string): RuleTraffic | null {
  const r = stats?.rules.find((x) => x.rule_id === ruleId);
  if (!r) return null;
  const total = r.local + r.fallback + r.refused;
  if (total === 0) return null;
  const parts: string[] = [];
  if (r.local) parts.push(`${fmtInt(r.local)} local`);
  if (r.fallback) parts.push(`${fmtInt(r.fallback)} fell back`);
  if (r.refused) parts.push(`${fmtInt(r.refused)} refused`);
  return { local: r.local, fallback: r.fallback, refused: r.refused, total, label: parts.join(" · ") };
}

// ── Relay keys ─────────────────────────────────────────────────────────────

/** A key may relay when it has the flag and still authenticates. */
export function relayFromTokens(page: TokenListPage | undefined, error: boolean): RelayInfo {
  if (error) return { relayKeys: 0, relayKnown: false, relayLoading: false };
  if (!page) return { relayKeys: 0, relayKnown: false, relayLoading: true };
  const relayKeys = page.items.filter(
    (t) => t.anthropic_relay && !t.is_revoked && !t.is_expired && !t.is_paused,
  ).length;
  // One found is proof; none found is proof only when we saw every key.
  const relayKnown = relayKeys > 0 || page.total <= page.items.length;
  return { relayKeys, relayKnown, relayLoading: false };
}

// ── Setup checklist (plan §2.1, §4.3) ──────────────────────────────────────

export type SetupStepId = "rule" | "relay" | "connect" | "enable" | "first_request";

export interface SetupStep {
  id: SetupStepId;
  done: boolean;
  /** The first step that is neither done nor loading. */
  current: boolean;
  /** Its data is still loading: neutral mark, no tick. */
  loading: boolean;
  /** relay only: the token list could not prove the answer either way. */
  unverified?: boolean;
}

export interface SetupInput {
  settings: RouterSettingsOut | undefined;
  rules: RuleOut[] | undefined;
  stats: RouterStatsOut | undefined;
  decisions: DecisionOut[] | undefined;
  relay: RelayInfo | undefined;
}

export function setupSteps({ settings, rules, stats, decisions, relay }: SetupInput): SetupStep[] {
  const seenLoading = stats === undefined && decisions === undefined;
  const seen = (stats?.since ?? null) !== null || (decisions?.length ?? 0) > 0;
  const relayLoading = !relay || relay.relayLoading;
  const raw: Omit<SetupStep, "current">[] = [
    { id: "rule", loading: rules === undefined, done: !!rules?.some((r) => r.enabled) },
    {
      id: "relay",
      loading: relayLoading,
      done: !!relay && relay.relayKnown && relay.relayKeys > 0,
      ...(relay && !relay.relayLoading && !relay.relayKnown ? { unverified: true } : {}),
    },
    { id: "connect", loading: seenLoading, done: seen },
    { id: "enable", loading: settings === undefined, done: !!settings?.enabled },
    { id: "first_request", loading: seenLoading, done: seen },
  ];
  const cur = raw.findIndex((s) => !s.done && !s.loading);
  return raw.map((s, i) => ({ ...s, current: i === cur }));
}

export interface SetupVisibilityInput {
  settings: RouterSettingsOut | undefined;
  rules: RuleOut[] | undefined;
  stats: RouterStatsOut | undefined;
  decisions: DecisionOut[] | undefined;
}

/** Whether the setup checklist (and the explainer) replace the day-to-day
 *  view — decided on durable facts only (plan §4.1 #2), never on "no request
 *  since this process started", which every restart or counter reset makes
 *  true again:
 *  - no enabled rule → setup, on or off;
 *  - a rule, routing off and no request seen yet → still onboarding;
 *  - routing on with a rule → never, even with no traffic (waiting strip);
 *  - routing off with traffic seen (the kill switch mid-incident) → never.
 *  While settings, rules or traffic load it says no, so a configured router
 *  never flashes the checklist. */
export function showSetupChecklist({ settings, rules, stats, decisions }: SetupVisibilityInput): boolean {
  if (!settings || !rules) return false;
  if (!rules.some((r) => r.enabled)) return true;
  if (settings.enabled) return false;
  if (stats === undefined && decisions === undefined) return false;
  const seen = (stats?.since ?? null) !== null || (decisions?.length ?? 0) > 0;
  return !seen;
}

// ── deriveRouterState (plan §4.1) ──────────────────────────────────────────

function targetName(rule: RuleOut, target?: TargetStats): string {
  return rule.target_served_name ?? target?.served_name ?? rule.target_model_id;
}

function failurePath(rule: RuleOut): StripPart {
  return rule.fallback
    ? "are going to Anthropic instead (the rule falls back)."
    : "are refused with 529 (Claude Code retries).";
}

function breakerIssue(
  rule: RuleOut,
  target: TargetStats,
  stats: RouterStatsOut,
  now: number,
): RouterIssue {
  const name = targetName(rule, target);
  // half_open: open_until has passed and no probe has run yet — nothing is
  // failing over right now; the next request is the probe (amber, like the
  // Activity page's breaker pill).
  const half = target.breaker === "half_open";
  const body: StripPart[] = half
    ? [
        "It failed recently. The next request for ",
        { code: rule.pattern },
        " tries it again; the breaker closes if it answers.",
      ]
    : ["Requests for ", { code: rule.pattern }, " ", failurePath(rule)];
  if (!half) {
    const inS = fmtIn(target.open_until, now);
    body.push(inS ? ` Next probe in ${inS}.` : " Next probe on the next request.");
  }
  if (target.last_reason) {
    body.push(" Last failure: ", { code: target.last_reason });
    body.push(target.consecutive_failures > 0 ? `, ${target.consecutive_failures} in a row.` : ".");
  }
  // The counter runs since process start; Activity lists only the last 100
  // decisions, so the count stays here and the link promises "recent" only.
  const fb = stats.rules.find((r) => r.rule_id === rule.id)?.fallback ?? 0;
  if (rule.fallback && fb > 0 && stats.since) {
    body.push(` ${fmtInt(fb)} fell back since ${fmtClock(stats.since)} — Activity lists the last 100 requests.`);
  }
  const actions: StripAction[] = [
    { kind: "open_model", label: `Open ${name}`, href: modelHref(rule.target_model_id), modelId: rule.target_model_id },
  ];
  if (rule.fallback) {
    actions.push({ kind: "see_fallbacks", label: "See recent fallbacks", href: FALLBACKS_HREF });
  }
  actions.push({ kind: "pause_rule", label: "Pause this rule", ruleId: rule.id });
  return {
    code: "breaker",
    tone: half ? "amber" : "danger",
    title: half
      ? [`${name}'s breaker is half-open — the next request will probe it.`]
      : [`${name} is failing — its breaker is open.`],
    body,
    actions,
    ruleId: rule.id,
    modelId: rule.target_model_id,
  };
}

function missingIssue(rule: RuleOut): RouterIssue {
  return {
    code: "target_missing",
    tone: "danger",
    title: ["The local model for ", { code: rule.pattern }, " was deleted."],
    body: [
      "Its requests are handled as ",
      { code: "target_missing" },
      rule.fallback ? " (they fall back to Anthropic)." : " (refused with 529).",
    ],
    actions: [
      { kind: "edit_rule", label: "Edit rule", ruleId: rule.id },
      { kind: "pause_rule", label: "Pause this rule", ruleId: rule.id },
    ],
    ruleId: rule.id,
  };
}

function unloadedIssue(rule: RuleOut): RouterIssue {
  const name = targetName(rule);
  return {
    code: "target_unloaded",
    tone: "amber",
    title: [`${name} is not loaded.`],
    body: [
      { code: rule.pattern },
      rule.fallback ? " requests are going to Anthropic." : " requests are refused (529).",
    ],
    actions: [
      { kind: "open_model", label: `Open ${name}`, href: modelHref(rule.target_model_id), modelId: rule.target_model_id },
    ],
    ruleId: rule.id,
    modelId: rule.target_model_id,
  };
}

function recentIssue(decisions: DecisionOut[]): RouterIssue | null {
  const window = decisions.slice(0, RECENT_WINDOW);
  const bad = window.filter((d) => NOT_LOCAL.has(d.route));
  if (bad.length === 0) return null;
  const reasons = new Map<string, number>();
  for (const d of bad) if (d.reason) reasons.set(d.reason, (reasons.get(d.reason) ?? 0) + 1);
  const top = [...reasons.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  const allFallback = bad.every((d) => d.route === "fallback");
  return {
    code: "recent_fallbacks",
    tone: "amber",
    title: [`${bad.length} of the last ${window.length} requests ${allFallback ? "fell back" : "fell back or failed"}.`],
    body: top ? ["Top reason: ", { code: top }, "."] : [],
    actions: [{ kind: "see_fallbacks", label: "See them", href: allFallback ? FALLBACKS_HREF : ACTIVITY_HREF }],
  };
}

function issueStrip(issues: RouterIssue[]): RouterStrip {
  const [head, ...more] = issues;
  return { tone: head.tone, role: "alert", title: head.title, body: head.body, actions: head.actions, more };
}

/** One answer for the whole page: the router's state, every applicable
 *  issue in priority order, and the strip's headline (plan §4.1). */
export function deriveRouterState(input: RouterStateInput): RouterState {
  const { settings, rules, stats, statsError, decisions, relay, now } = input;
  if (!settings) return { kind: "loading", issues: [], strip: null };

  const issues: RouterIssue[] = [];
  // 1. The stats fetch failed: say so, never pretend there is no traffic.
  if (statsError) {
    issues.push({
      code: "stats_error",
      tone: "danger",
      title: ["Can't load router status."],
      body: [stats ? "Showing the last good data." : "Traffic counts are unavailable."],
      actions: [{ kind: "retry", label: "Retry" }],
    });
  }

  const enabledRules = (rules ?? []).filter((r) => r.enabled);

  // 2. Off: nothing is routed, so engine trouble is not the headline.
  if (!settings.enabled) {
    if (issues.length) return { kind: "error", issues, strip: issueStrip(issues) };
    if (rules !== undefined && enabledRules.length === 0) {
      return {
        kind: "setup",
        issues,
        strip: {
          tone: "neutral",
          role: "status",
          title: ["Not set up yet."],
          body: [
            "Add a rule, give Claude Code a key that may relay to Anthropic, then turn routing on. " +
              "Until then nothing changes: requests are answered by served model name only, and nothing is relayed.",
          ],
          actions: [],
          more: [],
        },
      };
    }
    const n = enabledRules.length;
    return {
      kind: "off",
      issues,
      strip: {
        tone: "neutral",
        role: "status",
        title: ["Routing is off."],
        body: [
          `${rules === undefined ? "Your rules are" : n === 1 ? "Your rule is" : `Your ${n} rules are`} kept; Claude model names get 404 unless a model is ` +
            "served under that name. Nothing is relayed.",
        ],
        actions: [{ kind: "turn_on", label: "Turn on" }],
        more: [],
      },
    };
  }

  // 3–5. Per enabled rule, its single worst problem, grouped by check.
  // Danger before amber: an open breaker, then a deleted target, then a
  // half-open breaker (the next request probes), then an unloaded target.
  const breaker: RouterIssue[] = [];
  const missing: RouterIssue[] = [];
  const probing: RouterIssue[] = [];
  const unloaded: RouterIssue[] = [];
  for (const rule of enabledRules) {
    const target = stats?.targets.find((t) => t.model_id === rule.target_model_id);
    if (stats && target && (target.breaker === "open" || target.breaker === "half_open")) {
      (target.breaker === "open" ? breaker : probing).push(breakerIssue(rule, target, stats, now));
    } else if (rule.target_served_name === null) {
      missing.push(missingIssue(rule));
    } else if (rule.target_status !== "loaded") {
      unloaded.push(unloadedIssue(rule));
    }
  }
  issues.push(...breaker, ...missing, ...probing, ...unloaded);

  // 6. Relaying is needed but no key may relay (only when verified).
  const needsRelay = settings.passthrough_unmatched || enabledRules.some((r) => r.fallback);
  if (relay?.relayKnown && relay.relayKeys === 0 && needsRelay) {
    issues.push({
      code: "no_relay",
      tone: "amber",
      title: ["No key may relay to Anthropic."],
      body: ["Unmatched models and fallbacks are refused (403 ", { code: "relay_not_allowed" }, ")."],
      actions: [
        { kind: "create_relay_key", label: "Create a relay key" },
        { kind: "api_keys", label: "API Keys", href: "/tokens" },
      ],
    });
  }

  // 7. Recent requests that did not run locally.
  const recent = decisions ? recentIssue(decisions) : null;
  if (recent) issues.push(recent);

  const note =
    rules !== undefined && enabledRules.length === 0
      ? settings.passthrough_unmatched
        ? "On with no rules: every Claude model goes to Anthropic."
        : "On with no rules: every Claude model gets 404."
      : undefined;

  if (issues.length) {
    const head = issues[0];
    const kind: RouterStateKind = head.code === "stats_error" ? "error" : head.tone;
    return { kind, issues, strip: { ...issueStrip(issues), ...(note ? { note } : {}) } };
  }

  // Nothing wrong found, but without stats there is nothing true to say yet.
  if (!stats) return { kind: "loading", issues, strip: null };

  // 8. On, waiting for the first request.
  if (stats.since === null) {
    return {
      kind: "waiting",
      issues,
      strip: {
        tone: "neutral",
        role: "status",
        title: ["Routing is on — waiting for the first request."],
        body: ["Point Claude Code here (Connect below)."],
        actions: [{ kind: "connect", label: "Connect Claude Code", href: "#connect" }],
        pulse: true,
        ...(note ? { note } : {}),
        more: [],
      },
    };
  }

  // 9. Healthy.
  const parts: string[] = [];
  if (decisions?.[0]) parts.push(`Last request ${fmtAgo(decisions[0].ts, now)}`);
  const targets = new Set(enabledRules.map((r) => r.target_model_id)).size;
  if (targets > 0) parts.push(`${plural(targets, "local target")} healthy`);
  const split = trafficSplit(stats.totals);
  if (split.total > 0) {
    parts.push(`${fmtPct(split.segments[0].pct)} answered locally since ${fmtClock(stats.since)}`);
  } else {
    parts.push(`no requests since ${fmtClock(stats.since)}`);
  }
  return {
    kind: "ok",
    issues,
    strip: {
      tone: "ok",
      role: "status",
      title: ["Routing normally."],
      body: [`${parts.join(" · ")}.`],
      actions: [],
      ...(note ? { note } : {}),
      more: [],
    },
  };
}
