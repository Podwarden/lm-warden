// Every user-visible string of the sessions pages (English only; one module, for a later i18n pass).

import { formatCompact } from "@/lib/live-stats";
import type { TurnDeadWeight } from "@/lib/sessions/types";

/** "40 s", "19 min", "1 h 48 min", "31 h" (a day or more: hours only). */
function spanText(s: number): string {
  if (s < 60) return `${Math.max(0, Math.round(s))} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s >= 86_400) return `${Math.round(s / 3600).toLocaleString()} h`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s - h * 3600) / 60);
  if (m === 60) return `${h + 1} h 00 min`;
  return m === 0 ? `${h} h` : `${h} h ${String(m).padStart(2, "0")} min`;
}

/** "≈ 40 s", "≈ 19 min", "≈ 1 h 48 min", "≈ 31 h" (shared by strings.ts and format.ts). */
export function fmtApproxText(s: number): string {
  return `≈ ${spanText(s)}`;
}

export const S = {
  nav: "Sessions",
  links: { open: "open", title: "Open this request's session" },
  crumbList: "Sessions",
  crumbDetail: "Session",
  list: {
    title: "Sessions",
    intro:
      "Start from the overview: pick a time on the chart or a ready view, then open a key or harness to see its sessions.",
    rangeLabel: "Range",
    rangeText: {
      "1h": "the last hour",
      "6h": "the last 6 hours",
      "24h": "the last 24 hours",
      "7d": "the last 7 days",
      "30d": "the last 30 days",
    } as Record<string, string>,
    chartHead: (n: number, range: string) => `${n.toLocaleString()} session${n === 1 ? "" : "s"} in ${range}`,
    chartHint:
      "Drag on the chart to narrow the time. Bar height: sessions started per hour; light top: share with dead weight over 10%.",
    chartAria: "Sessions started per hour",
    clearSelection: "Clear selection",
    views: {
      all: { label: "All", small: (n: number) => n.toLocaleString() },
      running: { label: "Running now", small: (n: number) => n.toLocaleString() },
      biggest: { label: "Biggest", small: (_n: number) => "top 50 by prompt tokens" },
      dead_weight: { label: "Most dead weight", small: (n: number) => `over 20%: ${n.toLocaleString()}` },
      cache_trouble: { label: "Cache trouble", small: (n: number) => `under 60% from cache: ${n.toLocaleString()}` },
      long_running: { label: "Long-running", small: (n: number) => `over 1 h: ${n.toLocaleString()}` },
      evicted: { label: "Evicted after pause", small: (n: number) => n.toLocaleString() },
    },
    viewsAria: "Ready views",
    groupByLabel: "Group by",
    groupBy: { key: "Key", harness: "Harness", day: "Day", none: "None" },
    whereLabel: "Where",
    where: { all: "All", local: "Local", relayed: "Relayed", hybrid: "Hybrid" },
    searchLabel: "Search",
    searchPlaceholder: "Find a session id or key name",
    sortLabel: "Sort",
    sort: {
      newest: "Newest first",
      prompt_tokens: "Most prompt tokens",
      dead_weight: "Most dead weight",
      duration: "Longest",
      evicted_lost: "Most time lost to evictions",
    },
    firstCol: {
      key: "Key and its sessions",
      harness: "Harness and its sessions",
      day: "Day and its sessions",
      none: "Sessions",
    },
    cols: {
      sessions: "Sessions",
      duration: "Duration",
      prompt: "Prompt tokens",
      anatomy: "Prompt anatomy",
      saved: "Saved vs plain vLLM",
      score: "Prompt Lens score",
    },
    openInLens: "Open in Prompt Lens",
    notScored: "not scored yet",
    durationTotal: (d: string) => `${d} total`,
    whereCounts: (w: { relayed: number; hybrid: number; local: number }) =>
      [
        w.relayed ? `${w.relayed.toLocaleString()} relayed` : null,
        w.hybrid ? `${w.hybrid.toLocaleString()} hybrid` : null,
        w.local ? `${w.local.toLocaleString()} local` : null,
      ]
        .filter(Boolean)
        .join(", "),
    groupEvicted: (n: number, s: number | null) =>
      `${n.toLocaleString()} evicted after a pause` + (s === null ? "" : `, ${fmtApproxText(s)} lost`),
    running: "running",
    agents: (n: number) => `${n} agent${n === 1 ? "" : "s"}`,
    compactions: (n: number) => `${n} compaction${n === 1 ? "" : "s"}`,
    relayed: "relayed",
    hybrid: "hybrid",
    active: "running now",
    savedRelayed: "— relayed",
    savedUnknown:
      "Unknown: this model has no cold-rate measurement yet, or it was deleted. Token figures are still shown.",
    savedNote:
      "≈ prefill time LM Warden's cache-aware placement spared, against plain vLLM routing the same turns by load.",
    showMore: (n: number, label: string) => `Show ${n.toLocaleString()} more of ${label}'s sessions`,
    showMoreFlat: (n: number) => `Show ${n.toLocaleString()} more sessions`,
    expand: (label: string) => `${label}: show sessions`,
    loading: "Loading sessions…",
    empty: "No sessions in this range.",
    emptyFiltered: "No sessions match this selection.",
    oldest: (date: string) => `The oldest recorded request is from ${date}.`,
    truncated: "Very busy range: only the newest requests were scanned. Pick a shorter range to see everything.",
    unavailable: "Sessions are unavailable right now. The rest of the dashboard keeps working.",
  },
  harness: {
    claude_code: "Claude Code",
    codex: "Codex",
    opencode: "OpenCode",
    openai_sdk: "OpenAI SDK",
    other: "Other",
  } as Record<string, string>,
  anatomy: {
    cached: "from cache",
    dead: "dead weight",
    recomputed: "recomputed",
    new: "new this turn",
    unknown: "cache unknown",
    aria: (parts: string) => `Prompt anatomy: ${parts}`,
    part: (label: string, pct: number) => `${label} ${pct}%`,
    title: (label: string, tokens: string) => `${label}: ≈ ${tokens} tokens`,
  } as const,
  pauses: {
    title: "Pauses and evictions",
    surviveTitle: "How long caches survive a pause",
    summary: (resumes: number, evicted: number, lostS: number | null) =>
      `${resumes.toLocaleString()} resume${resumes === 1 ? "" : "s"} after a pause of a minute or more; ` +
      (evicted === 0
        ? "none found its cache evicted"
        : `${evicted.toLocaleString()} found their cache evicted` +
          (lostS === null ? "" : `, ${fmtApproxText(lostS)} of prefill lost`)),
    cols: {
      pause: "Pause",
      resumes: "Resumes",
      evicted: "Evicted",
      share: "Share evicted",
      lost: "≈ Time lost",
    },
    bucket: {
      "1-5m": "1–5 min",
      "5-15m": "5–15 min",
      "15-60m": "15–60 min",
      "1-4h": "1–4 h",
      "4h+": "over 4 h",
    } as Record<string, string>,
    hint: (keepS: number, recoveredS: number) =>
      `A disk tier that keeps prefixes for ${spanText(keepS)} would have recovered ${fmtApproxText(recoveredS)}. ` +
      "Estimate; disk load time not included.",
    noEvictions: "No resume after a pause had to prefill again in this selection.",
    none: "No pauses of a minute or more in this session.",
    unassessed: (n: number) =>
      `${n.toLocaleString()} resume${n === 1 ? "" : "s"} could not be assessed: the engine reported no cached tokens.`,
    explain:
      "A resume is a turn that starts more than a minute after its conversation's previous turn ended. It counts as evicted when at least half of the prompt it could have reused had to be prefilled again.",
  },

  page: {
    title: "Session",
    loading: "Loading session…",
    notFound: "Session not found.",
    notRecordedYet:
      "This request is still running. Its session appears here once its first turn is recorded.",
    unavailable: "This session is unavailable right now. Try again in a moment.",
  },
  header: {
    title: (key: string | null, harness: string) => (key ? `${key} · ${harness} session` : `${harness} session`),
    meta: (start: string, running: string, models: string, replicas: number | null) =>
      `Started ${start}, ${running}` +
      (models ? ` · ${models}` + (replicas !== null && replicas > 1 ? ` on ${replicas} replicas` : "") : ""),
    today: (hm: string) => `${hm} today`,
    runningFor: (d: string) => `running for ${d}`,
    ranFor: (d: string) => `ran for ${d}`,
    labels: {
      requests: "Requests",
      prompt: "Prompt tokens",
      generated: "Generated",
      served: "Served from cache",
      saved: "Saved vs plain vLLM",
      recomputed: "Recomputed",
    },
    subs: {
      requests: (convs: number, comps: number) =>
        `${convs} conversation${convs === 1 ? "" : "s"}, ${comps} compaction${comps === 1 ? "" : "s"}`,
      prompt: (pct: number) => `${pct}% served from cache`,
      promptUnknown: "cache use unknown",
      generated: "tokens",
      served: "of prefill not spent, at this model's cold rate",
      saved: "turns kept on their replica vs load-based routing",
      recomputed: "prefill spent on prompt the cache didn't have",
    },
    unknown: {
      relayed: "Relayed: the upstream served these turns, so its cache and timing are not ours to measure.",
      cache: "Cache unknown: this engine reports no cached tokens.",
      rate: "No cold-rate measurement for this model yet, so prefill time cannot be estimated.",
      model: "This model is no longer configured, so its replicas and cold rate are unknown.",
    },
    relayed: (n: number, pct: number | null) =>
      `relayed: ${n} request${n === 1 ? "" : "s"}` +
      (pct === null ? "" : `, ${pct}% from the upstream's prompt cache`),
    truncated: "Showing the newest 5,000 turns of this session.",
    oldest: (date: string) => `History is kept from ${date}.`,
  },
  deadWeight: {
    title: "Dead weight in this session's prompts",
    of: (dead: string, prompt: string) => `≈ ${dead} of ${prompt} prompt tokens`,
    measured: (m: number, n: number) => `measured on ${m.toLocaleString()} of ${n.toLocaleString()} turns`,
    notMeasured: "Not measured: these turns were recorded before dead weight was tracked.",
    kinds: {
      unused_tools: {
        label: "Unused tools",
        sub: (unused: number, offered: number) => `${unused} of ${offered} never called · strippable`,
      },
      dup: { label: "Repeated content", sub: () => "the same message sent again in one prompt" },
      boil: { label: "Harness boilerplate", sub: () => "reminders and date lines re-sent each turn" },
      reason: { label: "Carried reasoning", sub: () => "old thinking sent back" },
    },
    legendAria: "Legend",
  },
  engines: {
    title: "The same traffic on other engines",
    note: "A replay of this session's turns through each engine's cache rules. Prefill recomputed, at this machine's measured rate.",
    scope: "This session alone; disk load time is not included.",
    names: {
      lm_warden: "LM Warden",
      vllm: "Plain vLLM",
      llamacpp: "llama.cpp server",
      mlx_lm: "mlx-lm server",
      omlx: "oMLX",
    } as Record<string, string>,
    measured: "measured",
    slots: (n: number) => `${n} slots`,
    omlxSmall: "excl. disk loads",
    unknown: "unknown",
    skipped: (n: number) =>
      `${n.toLocaleString()} turn${n === 1 ? "" : "s"} had no cold rate, so ${n === 1 ? "its" : "their"} time is left out.`,
  },
  timeline: {
    title: "Timeline",
    inferred: "inferred",
    moreLanes: (n: number) => `… ${n} more lane${n === 1 ? "" : "s"} ▸`,
    fewerLanes: "Show fewer lanes",
    relayed: "relayed",
    legend: "Bar colour: share of the prompt served from cache. Stripe: why the cache broke.",
  },
  tree: {
    title: "Conversations",
    inferred: "inferred",
    known: "known",
    compaction: "after compaction",
    turns: (n: number) => `${n} turn${n === 1 ? "" : "s"}`,
    sideCalls: (n: number) => `side calls (${n})`,
    all: "Whole session",
  },
  turn: {
    title: "Turns",
    cols: {
      turn: "Turn",
      edge: "Edge",
      prompt: "Prompt",
      cache: "Cache",
      gen: "Gen",
      time: "Prefill + gen",
      replica: "Replica",
      tier: "Tier",
      anatomy: "Prompt anatomy",
    },
    upstream: "upstream",
    expand: "Show message structure",
    lineageUnknown: "Lineage unknown before this turn (a restart, or history older than the in-memory window).",
    sys: "System",
    tools: "Tools",
    msgs: "Messages",
    lastUser: "Last user message",
    replica: "Replica",
    chars: (n: number) => `${n.toLocaleString()} chars`,
    broke: (n: number, cause: string) => `cache broke at msg ${n} · ${cause}`,
    causeShort: { compacted: "compaction", harness_rewrite: "rewrite" } as Record<string, string>,
    evicted: (pauseS: number | null) =>
      pauseS === null ? "evicted after a pause" : `evicted after a ${spanText(pauseS)} pause`,
    deadWeight: (d: TurnDeadWeight) => {
      const kinds = (
        [
          ["unused tools", d.unused_tools],
          ["repeated", d.dup],
          ["boilerplate", d.boil],
          ["reasoning", d.reason],
        ] as const
      )
        .filter(([, v]) => v !== null)
        .map(([k, v]) => `${k} ≈ ${formatCompact(v)}`);
      const share = d.share === null ? "" : ` (${Math.round(d.share * 100)}%)`;
      return `Dead weight ≈ ${formatCompact(d.total)} tokens${share}` + (kinds.length ? `: ${kinds.join(" · ")}` : "");
    },
    deadWeightUnknown: "Dead weight not measured for this turn",
  },
  tools: {
    title: "Tools",
    headline: (t: { offered: number; called: number; unused: number; perTurn: string | null; share: number | null }) =>
      `offered ${t.offered}, called ${t.called}; ${t.unused} unused` +
      (t.perTurn === null ? "" : ` ≈ ${t.perTurn} tokens per turn`) +
      (t.share === null ? "" : ` (${Math.round(t.share * 100)}% of context)`) +
      ", held in KV for every turn",
    none: "No tool data for this selection.",
    costNote:
      "Unused tool definitions ride in the prompt on every turn: they take KV memory and slow decode attention even when the prefix is cached. Token counts are estimates (≈).",
    cols: { tool: "Tool", family: "Family", tokens: "≈ Tokens", chars: "Chars", called: "Called", offered: "Offered turns" },
    truncated: "This harness offered more than 128 tools; only the first 128 are listed.",
    prev: "Previous",
    next: "Next",
  },
} as const;
