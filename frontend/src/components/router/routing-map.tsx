"use client";

// The routing map (plan §4.4, §5, §7, §8): the page's one bold move. An <ol>
// of rules in first-match order — pattern → local model [pill], a meta line
// with the failure path and only the non-default options, live per-rule
// traffic — and the always-present last row "every other Claude model →
// Anthropic / 404", which reflects `passthrough_unmatched`. Replaces the old
// rules table and the Connect panel's "Currently overridden" list.
//
// Editing happens on the row: ↑/↓ (PUT of the full id order), the enable
// switch, and the ⋯ menu (Edit, Delete…, and on phones also Move and
// Pause/Enable). Under 760 px each row becomes a stacked card; ↑/↓ and the
// switch are hidden with CSS and reached through the menu instead.

import { useEffect, useRef, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { Modal } from "@/components/ui/modal";
import { RuleDialog } from "@/components/router/rule-dialog";
import { PassthroughDialog } from "@/components/router/passthrough-dialog";
import { RowMenu } from "@/components/router/row-menu";
import { RouterSwitch } from "@/components/router/switch";
import {
  BAND_LABEL,
  ICON_BTN,
  META,
  PATTERN,
  PILL_DOT,
  ROW,
  ROW_ALARM,
  ROW_OFF,
  ROW_OFF_PATTERN,
  ROW_REST,
  SEC,
  SEC_HEAD,
  SEC_NOTE,
  SEC_TITLE,
  TONE_BAR,
  TONE_FG,
  btn,
  pill,
  type Tone,
} from "@/components/router/styles";
import { cn } from "@/lib/utils";
import {
  fmtClock,
  fmtIn,
  fmtInt,
  ruleTraffic,
  type RouterSettingsOut,
  type RouterStatsOut,
  type RuleOut,
  type RuleTraffic,
  type TargetStats,
} from "@/lib/router";

export interface RoutingMapProps {
  /** undefined while loading (row skeletons). */
  rules: RuleOut[] | undefined;
  /** undefined while loading. */
  settings: RouterSettingsOut | undefined;
  /** Live counters and breakers; undefined → every row shows "—". */
  stats?: RouterStatsOut;
  /** A rule was created, edited, deleted, toggled or reordered: re-fetch rules. */
  onChange: () => void;
  /** `passthrough_unmatched` was changed from the last row: re-fetch settings. */
  onSettingsChange: () => void;
  /** When given, "Add rule" / "Add a rule" call it (the page owns the add
   *  dialog, e.g. to share it with the checklist presets). Otherwise the map
   *  opens its own RuleDialog. */
  onAddRule?: (pattern?: string) => void;
  /** Clock for the breaker countdown (tests); defaults to Date.now(). */
  now?: number;
}

// ── Layout (mockup .route grid) ────────────────────────────────────────────
// desktop: order | pattern | arrow | target | traffic | actions
// < 760px: [n] pattern … ⋯ / → target [pill] / meta / traffic
const GRID_DESKTOP =
  "min-[760px]:grid-cols-[28px_minmax(0,1fr)_28px_minmax(0,1.15fr)_172px_112px] min-[760px]:items-center min-[760px]:gap-x-3 min-[760px]:gap-y-1";
const GRID = `relative grid grid-cols-[24px_minmax(0,1fr)_auto] items-start gap-x-2.5 gap-y-1.5 ${GRID_DESKTOP}`;
const MAP_HEAD = `hidden px-4 pb-1.5 ${BAND_LABEL} min-[760px]:grid min-[760px]:grid-cols-[28px_minmax(0,1fr)_28px_minmax(0,1.15fr)_172px_112px] min-[760px]:gap-3`;

const C_ORD =
  "col-start-1 row-start-1 grid h-6 w-6 place-items-center rounded-md bg-chat-surface-2/70 text-[12px] font-semibold text-chat-muted";
const C_PAT = "col-start-2 col-end-3 row-start-1";
const C_ARROW = "col-start-3 row-start-1 hidden h-3.5 w-7 text-chat-dim min-[760px]:block";
const C_TGT =
  "col-start-2 col-end-3 row-start-2 flex min-w-0 flex-wrap items-center gap-2 min-[760px]:col-start-4 min-[760px]:col-end-5 min-[760px]:row-start-1";
const C_META =
  "col-start-2 col-end-4 row-start-3 flex flex-wrap items-center gap-x-3.5 gap-y-1.5 min-[760px]:col-end-5 min-[760px]:row-start-2";
const C_TRAFFIC =
  "col-start-2 col-end-4 row-start-4 flex min-w-0 flex-col-reverse gap-1.5 text-[12.5px] min-[760px]:col-start-5 min-[760px]:col-end-6 min-[760px]:row-start-1 min-[760px]:row-end-3 min-[760px]:flex-col";
const C_ACTS =
  "col-start-3 col-end-4 row-start-1 row-end-3 flex items-center justify-end gap-0.5 min-[760px]:col-start-6 min-[760px]:col-end-7";
const MINIBAR = "flex h-1.5 overflow-hidden rounded-full bg-chat-surface-2";

const STROKE = {
  info: "rgb(var(--vw-info-fg))",
  danger: "rgb(var(--vw-danger))",
  amber: "rgb(var(--vw-amber-fg))",
};

// ── Derived per-row facts ──────────────────────────────────────────────────

function breakerOf(rule: RuleOut, stats?: RouterStatsOut): TargetStats | undefined {
  return stats?.targets.find((t) => t.model_id === rule.target_model_id);
}

/** Rendered with display:none (itself or an ancestor). */
function isHidden(el: HTMLElement): boolean {
  for (let n: HTMLElement | null = el; n; n = n.parentElement) {
    if (getComputedStyle(n).display === "none") return true;
  }
  return false;
}

/** Failing over right now. half_open is not: open_until has passed and the
 *  next request is the probe (amber pill, no tint). */
function isIncident(rule: RuleOut, target?: TargetStats): boolean {
  return rule.enabled && target?.breaker === "open";
}

/** The target pill: paused / deleted / breaker (overrides) / engine status. */
function targetPill(
  rule: RuleOut,
  target: TargetStats | undefined,
  now: number,
): { tone: Tone; text: string } | null {
  if (!rule.enabled) return { tone: "idle", text: "rule paused" };
  if (rule.target_served_name === null) return { tone: "danger", text: "model deleted" };
  if (target?.breaker === "open") {
    const left = fmtIn(target.open_until, now);
    return { tone: "danger", text: left !== null ? `breaker open · ${left}` : "breaker open" };
  }
  if (target?.breaker === "half_open") return { tone: "amber", text: "breaker half-open · next request probes" };
  const st = rule.target_status;
  if (!st) return null;
  if (st === "loaded") return { tone: "ok", text: "loaded" };
  if (st === "failed") return { tone: "danger", text: "failed" };
  return { tone: "amber", text: st };
}

function trafficWords(t: RuleTraffic): { n: number; word: string; tone: Tone }[] {
  return [
    { n: t.local, word: "local", tone: "ok" as Tone },
    { n: t.fallback, word: "fell back", tone: "amber" as Tone },
    { n: t.refused, word: "refused", tone: "danger" as Tone },
  ].filter((p) => p.n > 0);
}

function sentence(
  i: number,
  rule: RuleOut,
  pillInfo: { text: string } | null,
  incident: boolean,
  traffic: RuleTraffic | null,
): string {
  const head = `Rule ${i + 1}: ${rule.pattern}`;
  if (!rule.enabled) {
    return `${head} is paused${rule.target_served_name ? ` (goes to ${rule.target_served_name} when on)` : ""}.`;
  }
  const to =
    rule.target_served_name === null
      ? "goes to a deleted model."
      : `goes to ${rule.target_served_name}${pillInfo ? `, ${pillInfo.text}` : ""}.`;
  const fail = incident
    ? rule.fallback
      ? "Falling back to Anthropic now."
      : "Refusing with 529 now."
    : rule.fallback
      ? "If it fails, falls back to Anthropic."
      : "If it fails, refuses with 529.";
  const tr = traffic
    ? `${trafficWords(traffic)
        .map((p) => `${fmtInt(p.n)} ${p.word}`)
        .join(", ")}.`
    : "No traffic yet.";
  return `${head} ${to} ${fail} ${tr}`;
}

// ── Small pieces ───────────────────────────────────────────────────────────

function Arrow({ dashed }: { dashed?: boolean }) {
  return (
    <svg viewBox="0 0 28 14" aria-hidden="true" className={C_ARROW}>
      <path
        d="M1 7h23M19 2l5 5-5 5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeDasharray={dashed ? "3 3" : undefined}
      />
    </svg>
  );
}

function Swatch({ color, dashed, width = 1.6 }: { color: string; dashed?: boolean; width?: number }) {
  return (
    <svg viewBox="0 0 22 8" aria-hidden="true" className="h-2 w-[22px] flex-none">
      <path d="M1 4h20" stroke={color} strokeWidth={width} strokeDasharray={dashed ? "3 3" : undefined} />
    </svg>
  );
}

function Traffic({ traffic, incident }: { traffic: RuleTraffic | null; incident: boolean }) {
  if (!traffic) {
    return (
      <span data-testid="rule-traffic" className={C_TRAFFIC}>
        <span className="text-chat-muted">—</span>
      </span>
    );
  }
  const parts = trafficWords(traffic);
  return (
    <span data-testid="rule-traffic" className={C_TRAFFIC}>
      <span className="tabular-nums">
        {parts.map((p, k) => (
          <span key={p.word}>
            {k > 0 && " · "}
            {k === 0 ? (
              <b className="text-[15px] font-semibold">{fmtInt(p.n)}</b>
            ) : p.word === "fell back" && incident ? (
              <b className={cn("text-[15px] font-semibold", TONE_FG.amber)}>{fmtInt(p.n)}</b>
            ) : (
              fmtInt(p.n)
            )}{" "}
            {p.word}
          </span>
        ))}
      </span>
      <span className={MINIBAR}>
        {parts.map((p) => (
          <i
            key={p.word}
            className={cn("block h-full", TONE_BAR[p.tone])}
            style={{ width: `${(p.n / traffic.total) * 100}%` }}
          />
        ))}
      </span>
    </span>
  );
}

function Skeleton() {
  return (
    <div data-testid="router-rules-loading" aria-hidden="true">
      {[0, 1, 2].map((k) => (
        <div key={k} className={cn(ROW, "flex items-center gap-3")}>
          <span className="h-6 w-6 animate-pulse rounded-md bg-chat-surface-2/70 motion-reduce:animate-none" />
          <span className="h-3.5 w-32 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
          <span className="h-3.5 w-40 animate-pulse rounded bg-chat-surface-2/70 motion-reduce:animate-none" />
        </div>
      ))}
    </div>
  );
}

// ── The map ────────────────────────────────────────────────────────────────

type PendingFocus = { id: string | null; target: "menu" | "up" | "down"; from: RuleOut[] };

export function RoutingMap({
  rules,
  settings,
  stats,
  onChange,
  onSettingsChange,
  onAddRule,
  now,
}: RoutingMapProps) {
  const sorted = rules ? [...rules].sort((a, b) => a.position - b.position) : [];
  const [editing, setEditing] = useState<RuleOut | null>(null);
  const [adding, setAdding] = useState(false);
  const [deleting, setDeleting] = useState<RuleOut | null>(null);
  const [passOpen, setPassOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<PendingFocus | null>(null);
  const menuRefs = useRef(new Map<string, HTMLButtonElement>());
  const upRefs = useRef(new Map<string, HTMLButtonElement>());
  const downRefs = useRef(new Map<string, HTMLButtonElement>());
  const addRef = useRef<HTMLButtonElement | null>(null);
  const clock = now ?? Date.now();

  // Focus after a reorder (stays on the moved row's arrow) or a delete (the
  // next row's ⋯, else "Add rule"), once the re-fetched rules arrive.
  useEffect(() => {
    if (!pending || busy || rules === pending.from) return;
    setPending(null);
    if (pending.id === null) {
      addRef.current?.focus();
      return;
    }
    const pick = (m: Map<string, HTMLButtonElement>) => m.get(pending.id as string);
    let el: HTMLButtonElement | undefined;
    if (pending.target === "menu") el = pick(menuRefs.current);
    else {
      const [first, other] =
        pending.target === "up" ? [upRefs.current, downRefs.current] : [downRefs.current, upRefs.current];
      // The arrows sit in a display:none wrapper below 760 px (the viewport
      // may have changed since the click): skip a hidden one.
      const usable = (b?: HTMLButtonElement) => !!b && !b.disabled && !isHidden(b);
      el = pick(first);
      if (!usable(el)) el = pick(other);
      if (!usable(el)) el = pick(menuRefs.current);
    }
    el?.focus();
  }, [pending, busy, rules]);

  async function send(path: string, method: string, body?: unknown): Promise<boolean> {
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(path, {
        method,
        ...(body !== undefined
          ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
          : {}),
      });
      if (!r.ok) {
        setError(await errorDetail(r, `Request failed (HTTP ${r.status})`));
        onChange(); // re-fetch: the list may have moved under us
        return false;
      }
      onChange();
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
      return false;
    } finally {
      setBusy(false);
    }
  }

  /** `fromMenu`: the ⋯ menu's Move items exist only on phones, where the
   *  arrow buttons are hidden — focus returns to the ⋯ button instead. */
  async function move(idx: number, delta: -1 | 1, fromMenu = false) {
    const ids = sorted.map((r) => r.id);
    const id = ids[idx];
    [ids[idx], ids[idx + delta]] = [ids[idx + delta], ids[idx]];
    const from = rules ?? [];
    if (await send("/api/router/rules/order", "PUT", { ids })) {
      setPending({ id, target: fromMenu ? "menu" : delta < 0 ? "up" : "down", from });
    }
  }

  function toggle(rule: RuleOut) {
    void send(`/api/router/rules/${encodeURIComponent(rule.id)}`, "PATCH", { enabled: !rule.enabled });
  }

  function add() {
    if (onAddRule) onAddRule(undefined);
    else setAdding(true);
  }

  async function confirmDelete() {
    if (!deleting) return;
    const idx = sorted.findIndex((r) => r.id === deleting.id);
    const next = sorted[idx + 1] ?? sorted[idx - 1] ?? null;
    const from = rules ?? [];
    const ok = await send(`/api/router/rules/${encodeURIComponent(deleting.id)}`, "DELETE");
    if (ok) {
      setDeleting(null);
      setPending({ id: next?.id ?? null, target: "menu", from });
    }
  }

  const loading = rules === undefined || settings === undefined;
  const passthrough = settings?.passthrough_unmatched ?? true;
  const passCount = stats?.passthrough.requests ?? 0;

  return (
    <section aria-labelledby="router-map-h" className={SEC}>
      <div className={SEC_HEAD}>
        <h2 id="router-map-h" tabIndex={-1} className={cn(SEC_TITLE, "focus:outline-none")}>
          Where each Claude model goes
        </h2>
        <span className={SEC_NOTE}>First match wins · /v1/messages and count_tokens</span>
        {sorted.length > 0 && (
          <div className="ml-auto">
            <button
              ref={addRef}
              type="button"
              className={btn("default", "px-3 py-1")}
              onClick={add}
            >
              Add rule
            </button>
          </div>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="mx-4 mb-3 rounded-[10px] border border-vw-danger/55 bg-vw-danger-bg/30 px-3 py-2 text-[13px] text-vw-danger-fg"
        >
          {error}
        </div>
      )}

      {loading ? (
        <Skeleton />
      ) : (
        <>
          {sorted.length > 0 && (
            <div className={MAP_HEAD} aria-hidden="true">
              <span />
              <span>Claude Code asks for</span>
              <span />
              <span>Answered by</span>
              <span>{stats?.since ? `Since ${fmtClock(stats.since)}` : "Traffic"}</span>
              <span />
            </div>
          )}

          {sorted.length === 0 && (
            <div className="mx-4 mb-3 flex flex-wrap items-center justify-between gap-3.5 rounded-[10px] border border-dashed border-vw-rule-soft p-[18px] text-[13px] text-chat-muted">
              <span>
                <b className="text-chat-fg">No rules yet.</b>{" "}
                {passthrough
                  ? "With routing on, every Claude model would go to Anthropic."
                  : "With routing on, every Claude model would get 404."}
              </span>
              <button
                ref={addRef}
                type="button"
                className={btn("primary", "px-3 py-1")}
                onClick={add}
              >
                Add a rule
              </button>
            </div>
          )}

          <ol aria-label="Rules, first match wins" data-testid="router-rules" className="m-0 list-none p-0">
            {sorted.map((r, i) => {
              const target = breakerOf(r, stats);
              const incident = isIncident(r, target);
              const pillInfo = targetPill(r, target, clock);
              const traffic = ruleTraffic(stats, r.id);
              const first = i === 0;
              const last = i === sorted.length - 1;
              return (
                <li
                  key={r.id}
                  data-rule-id={r.id}
                  data-incident={incident ? "true" : undefined}
                  className={cn(ROW, GRID, incident && ROW_ALARM, !r.enabled && ROW_OFF)}
                >
                  <span data-testid="rule-sentence" className="sr-only">
                    {sentence(i, r, pillInfo, incident, traffic)}
                  </span>
                  <span aria-hidden="true" className={C_ORD}>
                    {i + 1}
                  </span>
                  <span
                    aria-hidden="true"
                    className={cn(C_PAT, PATTERN, "cursor-pointer", !r.enabled && ROW_OFF_PATTERN)}
                    onClick={() => setEditing(r)}
                  >
                    {r.pattern}
                  </span>
                  <Arrow dashed={incident} />
                  <span aria-hidden="true" className={cn(C_TGT, "cursor-pointer")} onClick={() => setEditing(r)}>
                    <span className="text-chat-dim min-[760px]:hidden">→</span>
                    {r.target_served_name !== null && (
                      <span className="font-semibold [overflow-wrap:anywhere]">{r.target_served_name}</span>
                    )}
                    {pillInfo && (
                      <span className={pill(pillInfo.tone)}>
                        {pillInfo.tone !== "idle" && <span className={PILL_DOT} />}
                        {pillInfo.text}
                      </span>
                    )}
                  </span>
                  <span aria-hidden="true" className={cn(C_META, META)}>
                    {!r.enabled ? (
                      <span>Paused — skipped until you turn it back on</span>
                    ) : (
                      <>
                        {incident ? (
                          r.fallback ? (
                            <span className={cn("inline-flex items-center gap-1.5 font-semibold", TONE_FG.amber)}>
                              <Swatch color={STROKE.amber} dashed width={2} />
                              Falling back to Anthropic now
                            </span>
                          ) : (
                            <span className={cn("inline-flex items-center gap-1.5 font-semibold", TONE_FG.danger)}>
                              <Swatch color={STROKE.danger} width={2} />
                              Refusing with 529 now
                            </span>
                          )
                        ) : r.fallback ? (
                          <span className="inline-flex items-center gap-1.5">
                            <Swatch color={STROKE.info} dashed />
                            If it fails: Anthropic
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5">
                            <Swatch color={STROKE.danger} />
                            If it fails: refuse (529, Claude Code retries)
                          </span>
                        )}
                        <span>{r.strip_thinking ? "Thinking off" : "Thinking passed through"}</span>
                        {r.min_max_tokens > 0 && <span>max_tokens ≥ {fmtInt(r.min_max_tokens)}</span>}
                      </>
                    )}
                  </span>
                  <span aria-hidden="true" className="contents">
                    <Traffic traffic={r.enabled ? traffic : null} incident={incident} />
                  </span>
                  <span className={C_ACTS}>
                    <span className="hidden min-[760px]:contents">
                      <button
                        ref={(el) => {
                          if (el) upRefs.current.set(r.id, el);
                          else upRefs.current.delete(r.id);
                        }}
                        type="button"
                        className={ICON_BTN}
                        aria-label={`Move ${r.pattern} up`}
                        disabled={first || busy}
                        onClick={() => void move(i, -1)}
                      >
                        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true" className="h-4 w-4">
                          <path d="M8 13V3M4 7l4-4 4 4" />
                        </svg>
                      </button>
                      <button
                        ref={(el) => {
                          if (el) downRefs.current.set(r.id, el);
                          else downRefs.current.delete(r.id);
                        }}
                        type="button"
                        className={ICON_BTN}
                        aria-label={`Move ${r.pattern} down`}
                        disabled={last || busy}
                        onClick={() => void move(i, 1)}
                      >
                        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true" className="h-4 w-4">
                          <path d="M8 3v10M4 9l4 4 4-4" />
                        </svg>
                      </button>
                      <span className="mx-1 inline-flex items-center">
                        <RouterSwitch
                          size="sm"
                          checked={r.enabled}
                          disabled={busy}
                          label={`Enable rule ${r.pattern}`}
                          onChange={() => toggle(r)}
                        />
                      </span>
                    </span>
                    <RowMenu
                      ref={(el) => {
                        if (el) menuRefs.current.set(r.id, el);
                        else menuRefs.current.delete(r.id);
                      }}
                      label={`Actions for ${r.pattern}`}
                      items={[
                        { key: "edit", label: "Edit rule", ariaLabel: `Edit rule ${r.pattern}`, onSelect: () => setEditing(r) },
                        {
                          key: "up",
                          label: "Move up",
                          disabled: first || busy,
                          className: "min-[760px]:hidden",
                          onSelect: () => void move(i, -1, true),
                        },
                        {
                          key: "down",
                          label: "Move down",
                          disabled: last || busy,
                          className: "min-[760px]:hidden",
                          onSelect: () => void move(i, 1, true),
                        },
                        {
                          key: "toggle",
                          label: r.enabled ? "Pause rule" : "Enable rule",
                          disabled: busy,
                          className: "min-[760px]:hidden",
                          onSelect: () => toggle(r),
                        },
                        {
                          key: "delete",
                          label: "Delete rule…",
                          ariaLabel: `Delete rule ${r.pattern}`,
                          danger: true,
                          onSelect: () => setDeleting(r),
                        },
                      ]}
                    />
                  </span>
                </li>
              );
            })}

            {/* The implicit last rule: what happens to everything else. */}
            <li data-testid="router-rest" className={cn(ROW, GRID, ROW_REST)}>
              <span className="sr-only">
                {passthrough
                  ? "Every other Claude model and /v1 path goes to Anthropic, on the user's own login. Needs a key that may relay to Anthropic."
                  : "Every other Claude model and /v1 path is answered 404 Not found and is not relayed."}
                {passCount > 0 ? ` ${fmtInt(passCount)} passed through.` : ""}
              </span>
              <span aria-hidden="true" className={C_ORD}>
                ∗
              </span>
              <span aria-hidden="true" className={cn(C_PAT, "text-[13.5px] font-medium text-chat-muted")}>
                Every other Claude model and /v1 path
              </span>
              <Arrow />
              <span aria-hidden="true" className={C_TGT}>
                <span className="text-chat-dim min-[760px]:hidden">→</span>
                {passthrough ? (
                  <>
                    <span className="inline-flex items-center gap-[7px] font-semibold">
                      <span className="h-2.5 w-2.5 flex-none rounded-[3px] bg-vw-info-fg" />
                      Anthropic
                    </span>{" "}
                    <span className="text-[12.5px] text-chat-muted">on the user&apos;s own login</span>
                  </>
                ) : (
                  <>
                    <span className="inline-flex items-center gap-[7px] font-semibold">
                      <span className="h-2.5 w-2.5 flex-none rounded-[3px] bg-chat-dim" />
                      404 Not found
                    </span>{" "}
                    <span className="text-[12.5px] text-chat-muted">(not relayed)</span>
                  </>
                )}
              </span>
              <span aria-hidden="true" className={cn(C_META, META)}>
                {passthrough ? (
                  <span>
                    Needs a key with <b className="font-medium text-chat-fg">May relay to Anthropic</b> · or
                    answer 404 instead
                  </span>
                ) : (
                  <span>Claude Code reports these models as not found · or pass them through to Anthropic</span>
                )}
              </span>
              <span aria-hidden="true" className="contents">
                {passthrough && passCount > 0 ? (
                  <span data-testid="rule-traffic" className={C_TRAFFIC}>
                    <span className="tabular-nums">
                      <b className="text-[15px] font-semibold">{fmtInt(passCount)}</b> passed through
                    </span>
                    <span className={MINIBAR}>
                      <i className={cn("block h-full w-full", TONE_BAR.info)} />
                    </span>
                  </span>
                ) : (
                  <span data-testid="rule-traffic" className={C_TRAFFIC}>
                    <span className="text-chat-muted">—</span>
                  </span>
                )}
              </span>
              <span className={C_ACTS}>
                <button
                  type="button"
                  className={btn("default", "px-3 py-1")}
                  aria-label="Change what happens to every other Claude model"
                  onClick={() => setPassOpen(true)}
                >
                  Change
                </button>
              </span>
            </li>
          </ol>
        </>
      )}

      <RuleDialog
        open={adding || editing !== null}
        rule={editing}
        onClose={() => {
          setAdding(false);
          setEditing(null);
        }}
        onSaved={onChange}
      />
      <PassthroughDialog
        open={passOpen}
        value={passthrough}
        onClose={() => setPassOpen(false)}
        onSaved={onSettingsChange}
      />
      <Modal open={deleting !== null} onClose={() => setDeleting(null)} title="Delete rule">
        <p className="mb-4 text-sm">
          Delete the rule <span className="!font-mono">{deleting?.pattern}</span>? Matching
          requests go back to the default handling.
        </p>
        <div className="flex justify-end gap-2">
          <button type="button" className={btn()} onClick={() => setDeleting(null)}>
            Cancel
          </button>
          <button
            type="button"
            className={btn("danger")}
            disabled={busy}
            onClick={() => void confirmDelete()}
          >
            Delete
          </button>
        </div>
      </Modal>
    </section>
  );
}
