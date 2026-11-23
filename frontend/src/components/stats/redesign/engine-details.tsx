"use client";

// ENGINE DETAILS (design 3.7): one fixed card for the scope's model. Lifetime
// counters since engine start, the engine's block-level prefix cache hit rate
// and the lifetime preemption count (no rate: a 2 s rate of a rare event was
// always 0.00, so the Preemptions panel is gone).

import type { LiveEngineFrame } from "@/lib/live-stats";
import { FIXED, fmtTokens } from "@/lib/stats-redesign";

import { FixedCard } from "./primitives";

function Kv({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
      <span className="block text-[10px] uppercase tracking-[0.06em] text-chat-dim">{k}</span>
      {children}
    </div>
  );
}

export function EngineDetails({ frame }: { frame: LiveEngineFrame | undefined }) {
  const t = frame?.throughput;
  const f = frame?.finished;
  // The backend's lifetime figure (local + external); null = not exported,
  // prefix caching off or no prompt yet: hide it rather than show 0 %.
  const rate = frame?.cache && "prefix_cache_hit_rate" in frame.cache ? frame.cache.prefix_cache_hit_rate : undefined;
  const local = rate === undefined ? frame?.cache?.prefix_hit_rate_cumulative : null;
  const external = rate === undefined ? frame?.cache?.external_prefix_hit_rate_cumulative : null;
  const pre = frame?.engine?.preemptions_total;
  const n = (v: number | null | undefined) => (v == null ? "-" : String(v));
  return (
    <FixedCard height={FIXED.engineCard} phoneHeight={FIXED.engineCardPhone} testid="engine-details">
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs md:[grid-template-columns:repeat(auto-fit,minmax(170px,1fr))]">
        <Kv k="Prompt (includes cached)">{t?.prompt_tokens_total != null ? `${fmtTokens(t.prompt_tokens_total)} tokens` : "-"}</Kv>
        <Kv k="Generated">{t?.generation_tokens_total != null ? `${fmtTokens(t.generation_tokens_total)} tokens` : "-"}</Kv>
        <Kv k="Stop / length / abort">
          {n(f?.stop)} / {n(f?.length)} / {n(f?.abort)}
        </Kv>
        {rate != null && <Kv k="Engine prefix cache, lifetime">{Math.round(rate * 100)} % (local and external)</Kv>}
        {rate === undefined && (local != null || external != null) && (
          <Kv k="Engine prefix cache, lifetime">
            {[
              local != null ? `local ${Math.round(local * 100)} %` : null,
              external != null ? `external ${Math.round(external * 100)} %` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </Kv>
        )}
        <Kv k="Preemptions">{pre != null ? `${pre} (lifetime, no rate)` : "-"}</Kv>
      </div>
    </FixedCard>
  );
}
