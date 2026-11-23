"use client";

// The admin's view of one key in the cache store (token details): what it keeps in
// RAM and on disk against its share, how often its parts were dropped, and whose new
// entries pushed them out ("dropped by", admin only). Store figures come from the
// store itself, so they cover the time since it started, not the windows. Then the
// isolation cost (isolated keys only, admin only: it would tell a holder whether
// someone else sent the same text).

import { CARD, CARD_TITLE } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";
import { fmtGB, fmtGiB, fmtInt, fmtPct, fmtTokens, sinceAgo, tenantLabel, type KvUsage, type TokenCacheOut } from "@/lib/kv";
import { Notice, ShareBar } from "./parts";

/** The cost headline keeps two decimals in millions (1.24M), the mockup's figure. */
function costText(n: number): string {
  return n >= 1e6 && n < 1e8 ? `${(n / 1e6).toFixed(2)}M` : fmtTokens(n);
}

function Tier({
  label,
  size,
  used,
  share,
  part,
  drops,
  fmt,
}: {
  label: string;
  size: string | null;
  used: number;
  share: number;
  part: number;
  drops: number;
  fmt: (n: number) => string;
}) {
  return (
    <div className="rounded-lg border border-vw-rule-soft/60 bg-chat-page/40 p-3">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="font-semibold uppercase tracking-[.06em] text-chat-muted">{label}</span>
        {size && <span className="text-chat-dim">tier {size}</span>}
      </div>
      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-1.5">
        <span className="text-[20px] font-semibold tabular-nums">{fmt(used)}</span>
        <span className="text-[12.5px] text-chat-muted">of a {fmt(share)} share</span>
      </div>
      <ShareBar used={used} share={share} label={`${label}: ${fmt(used)} of a ${fmt(share)} share`} className="mt-2.5" />
      <dl className="m-0 mt-2.5 grid grid-cols-[1fr_auto] gap-y-1 text-[12.5px]">
        <dt className="text-chat-dim">Share of the tier</dt>
        <dd className="m-0 text-right tabular-nums">{fmtPct(part)}</dd>
        <dt className="text-chat-dim">Dropped to make room</dt>
        <dd className="m-0 text-right tabular-nums">{fmtInt(drops)}</dd>
      </dl>
    </div>
  );
}

export function TokenStoreUse({
  report,
  usage,
  storeOn,
}: {
  report: TokenCacheOut;
  usage: KvUsage | undefined;
  /** Whether any model has the cache store on. */
  storeOn: boolean;
}) {
  const store = report.store;
  const policy = usage?.policy === "fair" ? "fair" : "lru";
  const ago = sinceAgo(store?.since ?? usage?.since);
  const by = store?.evicted_by ?? [];
  const maxBy = Math.max(1, ...by.map((e) => e.count));
  const isolated = report.isolation === "isolated";
  return (
    <>
      <section className={CARD} aria-label="KV cache store use" data-testid="token-store-use">
        <h2 className={cn(CARD_TITLE, "flex flex-wrap items-baseline gap-x-2")}>
          KV cache store use
          {storeOn && store && ago && (
            <span className="text-[12px] font-normal text-chat-dim">since the store started, {ago}</span>
          )}
        </h2>
        {!storeOn ? (
          <Notice tone="info">
            No model has the KV cache store on, so only the GPU cache is used. Turn it on in a model&apos;s settings (vLLM
            models).
          </Notice>
        ) : !store ? (
          <Notice tone="bad">
            The KV cache store is not answering, so its use is unavailable. Requests keep working: their prompts are computed
            on the GPU. The figures above come from request history and are complete.
          </Notice>
        ) : (
          <>
            <div className="grid gap-3 min-[700px]:grid-cols-2">
              <Tier
                label="RAM"
                size={usage?.tiers ? fmtGiB(usage.tiers.ram.capacity).replace(".0 ", " ") : null}
                used={store.ram_bytes}
                share={store.ram_share_bytes}
                part={store.ram_share}
                drops={store.ram_evictions}
                fmt={fmtGiB}
              />
              <Tier
                label="Disk"
                size={usage?.tiers && usage.tiers.nvme.capacity > 0 ? fmtGB(usage.tiers.nvme.capacity) : null}
                used={store.nvme_bytes}
                share={store.nvme_share_bytes}
                part={store.nvme_share}
                drops={store.nvme_evictions}
                fmt={fmtGB}
              />
            </div>
            {policy === "lru" && (
              <p className="m-0 mt-2 text-[12px] text-chat-dim">
                Shares are shown for planning. With the current policy (oldest first) the store drops the oldest part,
                whoever wrote it.
              </p>
            )}
            <div className="mt-4 border-t border-chat-rule pt-3" data-testid="kv-evicted-by">
              <h3 className="m-0 mb-2 text-[13px] font-semibold">
                Dropped by{" "}
                <span className="text-[12px] font-normal text-chat-dim">whose new cache entries pushed this key&apos;s parts out</span>
              </h3>
              {by.length === 0 ? (
                <p className="m-0 text-[12.5px] text-chat-dim">Nothing of this key&apos;s was dropped.</p>
              ) : (
                <ul className="m-0 list-none space-y-1.5 p-0">
                  {by.map((e) => (
                    <li key={e.token_id} className="grid grid-cols-[1fr_120px_48px] items-center gap-3 text-[13px] max-sm:grid-cols-[1fr_64px_40px]">
                      <span className="truncate">{tenantLabel(e.token_id, e.name, report.token_id)}</span>
                      <span className="h-1.5 overflow-hidden rounded-full bg-chat-surface-2" aria-hidden="true">
                        <span className="block h-full rounded-full bg-vw-danger" style={{ width: `${(e.count / maxBy) * 100}%` }} />
                      </span>
                      <span className="text-right tabular-nums">{fmtInt(e.count)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </>
        )}
      </section>

      {isolated && (
        <section className={CARD} aria-label="Isolation cost" data-testid="kv-isolation-cost">
          <h2 className={cn(CARD_TITLE, "flex items-baseline gap-2")}>
            Isolation cost <span className="text-[12px] font-normal text-chat-dim">admins only</span>
          </h2>
          <div className="flex flex-wrap items-baseline gap-x-3">
            <span className="text-[28px] font-semibold tabular-nums">{costText(report.isolation_cost_tokens["24h"])}</span>
            <span className="text-[13px] text-chat-muted">
              prompt tokens computed again in the last 24 h that another key had cached
            </span>
          </div>
          <p className="m-0 mt-1 text-[12px] text-chat-dim">
            What isolation cost this key. Never shown to the key holder: it would tell them whether someone else sent the
            same text.
          </p>
        </section>
      )}
    </>
  );
}
