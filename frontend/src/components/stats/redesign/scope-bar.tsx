"use client";

// The scope bar (design 2.2, 3.1): one row of fixed height. One model tab per
// model (a fixed 280 px with a fixed-width state chip); switching the tab
// switches the whole page. On one install the install name is plain text.
//
// TODO(B12): multisite. The Install selector ("All installs (N)" plus one button
// per install with a reachability dot) needs the cluster work (design 2.5,
// backend B12): install identity, reachability and per-install feeds tagged with
// install_id. Until then a page shows one install and the selector is absent.
// `INSTALL_SELECTOR_ENABLED` stays false until that lands.

import { FIXED, type ScopeState } from "@/lib/stats-redesign";
import { cn } from "@/lib/utils";

import { Line, StateChip, Swatch, hueOf } from "./primitives";

export const INSTALL_SELECTOR_ENABLED = false;

export interface ScopeTab {
  id: string;
  name: string;
  state: ScopeState;
}

export function ScopeBar({
  tabs,
  selected,
  onSelect,
  installName,
  emptyLabel = "no model loaded",
  emptyState = "No data",
}: {
  tabs: readonly ScopeTab[];
  selected: string | null;
  onSelect: (id: string) => void;
  /** The one tab shown when there is nothing to scope to. */
  emptyLabel?: string;
  emptyState?: ScopeState;
  /** Admin, one install: the install's name as plain text. Absent for key holders. */
  installName?: string | null;
}) {
  const solo = tabs.length <= 1;
  // Fixed per breakpoint: 280 px, 220 px on phones so the chip stays visible.
  const tabW = { "--tw": `${FIXED.tabW}px`, "--twp": `${FIXED.tabWPhone}px` } as React.CSSProperties;
  const tabCls = "w-[var(--twp)] md:w-[var(--tw)]";
  return (
    <div
      data-testid="scope-bar"
      data-fixed-height={FIXED.scopeBar}
      style={{ height: FIXED.scopeBar }}
      className="mt-3 flex items-center gap-3.5 overflow-x-auto overflow-y-hidden whitespace-nowrap rounded-lg border border-chat-rule bg-chat-surface px-2.5"
    >
      <span className="hidden flex-none text-[10px] uppercase tracking-[0.08em] text-chat-dim sm:inline">Model</span>
      <div role="tablist" aria-label="Model" className="flex flex-none gap-1.5">
        {tabs.length === 0 ? (
          <div
            data-testid="scope-tab-empty"
            style={tabW}
            className={cn(tabCls, "flex items-center gap-2 rounded-lg border border-chat-rule px-2 py-[5px] text-xs text-chat-dim")}
          >
            <span className="flex-1 overflow-hidden text-ellipsis">{emptyLabel}</span>
            <StateChip state={emptyState} />
          </div>
        ) : (
          tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={t.id === selected}
              data-testid="scope-tab"
              onClick={() => !solo && onSelect(t.id)}
              style={tabW}
              className={cn(
                tabCls,
                "flex flex-none items-center gap-2 rounded-lg border px-2 py-[5px]",
                t.id === selected ? "border-amber-500 bg-amber-400/[0.06]" : "border-chat-rule",
                solo ? "cursor-default" : "cursor-pointer",
              )}
            >
              <Swatch color={hueOf(t.id)} />
              <span className="flex-1 overflow-hidden text-ellipsis text-left font-mono text-xs" title={t.name}>
                {t.name}
              </span>
              <StateChip state={t.state} />
            </button>
          ))
        )}
      </div>
      {installName !== undefined && (
        <>
          <span className="flex-none text-[10px] uppercase tracking-[0.08em] text-chat-dim">Install</span>
          <span className="font-mono text-xs text-chat-muted">{installName ?? "this install"}</span>
        </>
      )}
    </div>
  );
}

/** The one reserved notice line under the scope bar: the most important notice,
 *  "+N more", or a muted "All feeds OK". Nothing is inserted above the page. */
export function NoticeLine({
  notices,
  ok = "All feeds OK",
  tone = "warn",
  action,
}: {
  notices: readonly { strong: string; rest?: string }[];
  ok?: string;
  /** "info": a statement about what is shown, not a fault. */
  tone?: "warn" | "info";
  /** One action at the end of the line (fixed height: it never wraps). */
  action?: { label: string; onClick: () => void };
}) {
  const first = notices[0];
  return (
    <div
      data-testid="notice-line"
      data-fixed-height={FIXED.notice}
      style={{ height: FIXED.notice }}
      className={cn(
        "mt-2 flex items-center gap-2 overflow-hidden whitespace-nowrap border-l-2 pl-2 text-xs text-chat-dim",
        first ? (tone === "info" ? "border-amber-500" : "border-red-400") : "border-chat-rule",
      )}
      role={first ? "status" : undefined}
    >
      {first ? (
        <Line className="flex-1">
          <b className={cn("font-medium", tone === "info" ? "text-amber-300" : "text-red-300")}>{first.strong}</b>
          {first.rest && <span> · {first.rest}</span>}
          {notices.length > 1 && <span> · +{notices.length - 1} more</span>}
        </Line>
      ) : (
        <Line className="flex-1">
          <b className="font-medium text-chat-muted">{ok}</b>
        </Line>
      )}
      {action && (
        <button
          type="button"
          data-testid="notice-action"
          onClick={action.onClick}
          className="flex-none rounded border border-amber-500 px-2 py-px text-[11px] text-chat-accent hover:bg-amber-400/10"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
