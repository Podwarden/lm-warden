// The same session replayed through each engine's cache rules: prefill recomputed, as time at this machine's rate.

import { S } from "@/lib/sessions/strings";
import type { EngineReplay } from "@/lib/sessions/types";

import { fmtApprox } from "./format";

export function EngineComparison({ r }: { r: EngineReplay }) {
  const anyTime = r.engines.some((e) => e.recompute_s !== null);
  const val = (e: EngineReplay["engines"][number]) => (anyTime ? e.recompute_s : e.recompute_tokens);
  const max = Math.max(0, ...r.engines.map((e) => val(e) ?? 0));
  return (
    <section data-testid="engine-comparison" className="space-y-2">
      <h2 className="text-sm font-semibold text-chat-fg">{S.engines.title}</h2>
      <p className="max-w-[72ch] text-xs text-chat-muted">
        {S.engines.note} {S.engines.scope}
      </p>
      <div className="space-y-1.5">
        {r.engines.map((e) => {
          const v = val(e);
          const width = v === null || max <= 0 ? 0 : (v / max) * 100;
          const small =
            e.engine === "lm_warden"
              ? S.engines.measured
              : e.engine === "llamacpp"
                ? S.engines.slots(r.slots)
                : e.engine === "omlx"
                  ? S.engines.omlxSmall
                  : null;
          return (
            <div
              key={e.engine}
              data-testid={`engine-${e.engine}`}
              className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_6.5rem] items-center gap-3 text-sm"
            >
              <span className="truncate text-chat-fg">
                {S.engines.names[e.engine] ?? e.engine}
                {small && <small className="ml-1.5 text-chat-dim">{small}</small>}
              </span>
              <span className="h-2 overflow-hidden rounded-sm bg-chat-rule">
                <i
                  data-bar
                  className={`block h-full ${e.engine === "lm_warden" ? "bg-emerald-500" : "bg-stone-500"}`}
                  style={{ width: `${width}%` }}
                />
              </span>
              <span className="text-right tabular-nums text-chat-fg">
                {e.recompute_s !== null ? fmtApprox(e.recompute_s) : S.engines.unknown}
              </span>
            </div>
          );
        })}
      </div>
      {r.skipped_time > 0 && <p className="text-xs text-chat-dim">{S.engines.skipped(r.skipped_time)}</p>}
    </section>
  );
}
