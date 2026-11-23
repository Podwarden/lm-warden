"use client";

// "Cache use": where a key's prompt tokens came from over 1 h / 24 h / 7 d (GPU, RAM,
// disk, computed again), from its request history, so it survives a store outage.
// The admin's card (token details) draws the split as a bar; the holder's card
// (key settings) puts the key's own store figures under the table.

import { useState, type ReactNode } from "react";
import { CARD } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";
import { fmtPct, fmtTokens, WINDOW_KEYS, WINDOW_LABEL, type KvWindows, type WindowKey } from "@/lib/kv";

const SEG = {
  gpu: { label: "GPU", color: "rgb(var(--vw-model-1))" },
  ram: { label: "RAM", color: "rgb(var(--vw-model-5))" },
  nvme: { label: "Disk", color: "rgb(var(--vw-model-6))" },
  recomputed: { label: "Computed again", color: "rgb(var(--vw-strip-chain))" },
} as const;

const TH = "px-1.5 py-2 text-[11px] font-medium uppercase tracking-[.06em] text-chat-dim max-sm:px-0.5 max-sm:text-[10px] max-sm:normal-case max-sm:tracking-normal";
const TD = "whitespace-nowrap px-1.5 py-2 tabular-nums max-sm:px-0.5 max-sm:text-[11.5px]";

export function CacheUseCard({
  windows,
  holder = false,
  children,
  testId,
}: {
  windows: KvWindows;
  /** The key holder's wording ("your"), no bar. */
  holder?: boolean;
  children?: ReactNode;
  testId?: string;
}) {
  const [sel, setSel] = useState<WindowKey>("24h");
  const w = windows[sel];
  const total = w.gpu + (w.ram ?? 0) + (w.nvme ?? 0) + w.recomputed;
  return (
    <section className={CARD} aria-label={holder ? "Your cache use" : "Cache use"} data-testid={testId}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="m-0 text-[14px] font-semibold">{holder ? "Your cache use" : "Cache use"}</h2>
        <div role="group" aria-label="Window" className="inline-flex overflow-hidden rounded-md border border-chat-rule">
          {WINDOW_KEYS.map((k) => (
            <button
              key={k}
              type="button"
              aria-pressed={sel === k}
              onClick={() => setSel(k)}
              className={cn(
                "cursor-pointer border-0 px-2.5 py-1 text-[12px]",
                sel === k ? "bg-chat-surface-2 font-semibold text-chat-accent" : "bg-transparent text-chat-muted hover:text-chat-fg",
              )}
            >
              {k}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-baseline gap-x-3">
        <span data-testid="kv-hit-headline" className="text-[30px] font-semibold tabular-nums leading-tight">
          {fmtPct(w.hit_rate)}
        </span>
        <span className="text-[13px] text-chat-muted">
          of {holder ? "your " : ""}prompt tokens came from a cache, last {WINDOW_LABEL[sel]}
        </span>
      </div>

      {!holder && (
        <>
          <div className="mt-3 flex h-2.5 overflow-hidden rounded-full bg-chat-surface-2" aria-hidden="true">
            {total > 0 &&
              (["gpu", "ram", "nvme", "recomputed"] as const).map((k) => {
                const v = (k === "ram" || k === "nvme" ? w[k] ?? 0 : w[k]) / total;
                return v > 0 ? (
                  <div key={k} style={{ width: `${(v * 100).toFixed(2)}%`, background: SEG[k].color }} className="h-full border-r border-chat-surface last:border-r-0" />
                ) : null;
              })}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-chat-muted">
            {(["gpu", "ram", "nvme", "recomputed"] as const).map((k) => (
              <span key={k} className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-sm" style={{ background: SEG[k].color }} />
                {SEG[k].label}
              </span>
            ))}
          </div>
        </>
      )}

      <div className="mt-3">
        <table className="w-full border-collapse text-[13px] max-sm:text-[12px]">
          <thead>
            <tr className="border-b border-chat-rule text-left">
              <th className={TH}>Window</th>
              <th className={cn(TH, "text-right")}>GPU</th>
              <th className={cn(TH, "text-right")}>RAM</th>
              <th className={cn(TH, "text-right")}>Disk</th>
              <th className={cn(TH, "text-right")}>Computed</th>
              <th className={cn(TH, "text-right")}>Hit rate</th>
            </tr>
          </thead>
          <tbody>
            {WINDOW_KEYS.map((k) => {
              const r = windows[k];
              return (
                <tr key={k} className={cn("border-b border-chat-rule/60", sel === k && "bg-chat-surface-2/60")}>
                  <td className={TD}>{WINDOW_LABEL[k]}</td>
                  <td className={cn(TD, "text-right")}>{fmtTokens(r.gpu)}</td>
                  <td className={cn(TD, "text-right")}>{fmtTokens(r.ram)}</td>
                  <td className={cn(TD, "text-right")}>{fmtTokens(r.nvme)}</td>
                  <td className={cn(TD, "text-right")}>{fmtTokens(r.recomputed)}</td>
                  <td className={cn(TD, "text-right")}>{fmtPct(r.hit_rate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!holder && (
        <p className="m-0 mt-2 text-[12px] text-chat-dim">
          Prompt tokens served from each place, from this key&apos;s request history. Hit rate is the part not computed again.
        </p>
      )}
      {children}
    </section>
  );
}
