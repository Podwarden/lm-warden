"use client";

// The turn strip (spec 12.1 item 5, v1 section 10): one bar per turn of the
// prompt's tokens, split into reused (green), recomputed and lost (red) and new
// this turn, with the layer letter of the findings that name the turn. A bar
// opens that turn in the request view.

import { forScope, layerOf } from "@/lib/prompt-lens/findings";
import type { TurnBar } from "@/lib/prompt-lens/scorecard";
import { fmtTok } from "@/lib/prompt-lens/scorecard";
import type { LensScope } from "@/lib/prompt-lens/stream";
import { cn } from "@/lib/utils";

const MARK: Record<string, string> = { E: "text-vw-ttft", C: "text-vw-dur", R: "text-vw-completion" };
const BAR_PX = 72;

export function TurnStrip({ turns, scope, onShowTurn }: { turns: TurnBar[]; scope: LensScope; onShowTurn: (reqId: string) => void }) {
  const max = Math.max(1, ...turns.map((t) => t.reused + t.recomputed + t.new));
  const h = (v: number) => Math.max(v > 0 ? 1 : 0, Math.round((v / max) * BAR_PX));
  return (
    <div data-testid="turn-strip" className="rounded-lg border border-chat-rule bg-chat-surface p-3">
      <div className="flex items-end gap-[3px] overflow-x-auto pb-1" style={{ minHeight: BAR_PX + 28 }}>
        {turns.map((t) => {
          const layers = [...new Set(t.markers.map(layerOf))];
          return (
            <button
              key={t.n}
              type="button"
              data-testid="turn-bar"
              data-turn={t.n}
              onClick={() => onShowTurn(t.req_id)}
              title={`turn ${t.n}: reused ${fmtTok(t.reused)}, recomputed ${fmtTok(t.recomputed)}, new ${fmtTok(t.new)}`}
              className="flex w-[14px] min-w-[14px] flex-none cursor-pointer flex-col items-center justify-end gap-0.5 bg-transparent p-0"
            >
              <span className="h-3 text-[10px] font-semibold leading-3">
                {layers.map((l) => (
                  <b key={l} className={cn(MARK[l])}>
                    {l}
                  </b>
                ))}
              </span>
              <span className="flex w-full flex-col justify-end" style={{ height: BAR_PX }}>
                <span className="block w-full bg-vw-info-fg/70" style={{ height: h(t.new) }} />
                <span className="block w-full bg-vw-danger-fg" style={{ height: h(t.recomputed) }} />
                <span className="block w-full bg-vw-ok-fg/80" style={{ height: h(t.reused) }} />
              </span>
              <em className="text-[9px] not-italic text-chat-dim">{t.n}</em>
            </button>
          );
        })}
      </div>
      <p className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-chat-dim">
        <span><i aria-hidden className="mr-1 inline-block h-2 w-2 bg-vw-ok-fg/80" />{forScope("reused from cache", scope)}</span>
        <span><i aria-hidden className="mr-1 inline-block h-2 w-2 bg-vw-danger-fg" />recomputed (lost)</span>
        <span><i aria-hidden className="mr-1 inline-block h-2 w-2 bg-vw-info-fg/70" />new this turn</span>
      </p>
    </div>
  );
}
