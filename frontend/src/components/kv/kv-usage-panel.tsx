"use client";

// The cache store on the Models page (GET /api/kv/usage, admin only): its status and
// policy, the two tiers (in use, waiting to be cleared, free), then one row per model
// with the store on (weight, part, use) that opens onto its keys (use, share, drops and
// the key that caused most of them). Figures cover the time since the store started.

import { Fragment, useState } from "react";
import { CARD } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";
import {
  fmtGB,
  fmtGiB,
  fmtInt,
  fmtPct,
  sinceAgo,
  tenantLabel,
  type KvModelUsage,
  type KvStoreMode,
  type KvUsage,
} from "@/lib/kv";
import { Mark, Notice, ShareBar } from "./parts";
import { LiveNumber } from "@/components/ui/live-number";

export interface KvModelInfo {
  name: string;
  kv_store?: KvStoreMode;
}

function TierCard({
  label,
  cap,
  used,
  dead,
  capacity,
  fmt,
  color,
}: {
  label: string;
  cap: string;
  used: number;
  dead: number;
  capacity: number;
  fmt: (n: number) => string;
  color: string;
}) {
  const live = Math.max(0, used - dead);
  const free = Math.max(0, capacity - used);
  const pct = (n: number) => (capacity > 0 ? `${((n / capacity) * 100).toFixed(2)}%` : "0%");
  return (
    <div className="rounded-lg border border-vw-rule-soft/60 bg-chat-page/40 p-3">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="font-semibold uppercase tracking-[.06em] text-chat-muted">{label}</span>
        <span className="text-chat-dim">{cap}</span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <LiveNumber reserve={9} className="text-[22px] font-semibold">
          {fmt(used)}
        </LiveNumber>
        <span className="text-[13px] text-chat-muted">used</span>
      </div>
      <div className="mt-2.5 flex h-2.5 overflow-hidden rounded-full bg-chat-surface-2" aria-hidden="true">
        <div style={{ width: pct(live), background: color }} />
        <div className="bg-vw-danger" style={{ width: pct(dead) }} />
      </div>
      <dl className="m-0 mt-2.5 grid grid-cols-[1fr_auto] gap-y-1 text-[12.5px]">
        <dt className="text-chat-dim">In use</dt>
        <dd className="m-0 text-right tabular-nums">{fmt(live)}</dd>
        <dt className="text-chat-dim">Waiting to be cleared</dt>
        <dd className="m-0 text-right tabular-nums">{fmt(dead)}</dd>
        <dt className="text-chat-dim">Free</dt>
        <dd className="m-0 text-right tabular-nums">{fmt(free)}</dd>
      </dl>
    </div>
  );
}

/** RAM and disk together, in whichever unit reads best. */
function fmtShort(bytes: number): string {
  return bytes >= 100e9 ? fmtGB(bytes) : fmtGiB(bytes);
}

const ROW =
  "grid items-center gap-x-3 gap-y-1 border-t border-chat-rule px-3 py-2 text-[13px] min-[760px]:grid-cols-[minmax(0,1.6fr)_60px_60px_80px_80px_minmax(0,1.6fr)] grid-cols-2";
const NUM = "tabular-nums min-[760px]:text-right";
const SMALL_LABEL = "text-[11px] text-chat-dim min-[760px]:hidden";

export function KvUsagePanel({
  usage,
  models,
  storeOn,
  isolated,
}: {
  usage: KvUsage | undefined;
  /** Model row id -> its served name and store tier. */
  models: Record<string, KvModelInfo>;
  /** Whether any model has the cache store on. */
  storeOn: boolean;
  /** Token ids of isolated keys, marked in the key rows. */
  isolated?: Set<string>;
}) {
  const first = usage?.models[0]?.model_id;
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const isOpen = (id: string) => open[id] ?? id === first;
  const available = !!usage?.available && !!usage.tiers;
  const policy = usage?.policy === "fair" ? "fair" : "lru";
  const ago = sinceAgo(usage?.since);
  const status = !storeOn ? "Off" : available ? "Running" : "Not answering";
  const tone = !storeOn ? "idle" : available ? "ok" : "bad";

  return (
    <section className={CARD} aria-label="KV cache store" data-testid="kv-usage-panel">
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <h2 className="m-0 text-[14px] font-semibold">KV cache store</h2>
        <Mark tone={tone}>{status}</Mark>
        {available && (
          <span className="text-[12px] text-chat-dim">
            Eviction: <b className="font-semibold text-chat-muted">{policy === "fair" ? "fair share" : "oldest first"}</b>
            {ago && <> · since the store started, {ago}</>}
          </span>
        )}
      </div>

      {!storeOn ? (
        <Notice tone="info">No model has the KV cache store on. Turn it on in a vLLM model&apos;s settings, under KV cache store.</Notice>
      ) : !available || !usage?.tiers ? (
        <Notice tone="bad">
          The KV cache store is not answering. Usage is unavailable. Requests keep working; their prompts are computed on the
          GPU. The api restarts the store after three missed checks.
        </Notice>
      ) : (
        <>
          <div className="grid gap-3 min-[760px]:grid-cols-2">
            <TierCard
              label="RAM"
              cap={fmtGiB(usage.tiers.ram.capacity).replace(".0 ", " ")}
              used={usage.tiers.ram.used}
              dead={usage.tiers.ram.dead}
              capacity={usage.tiers.ram.capacity}
              fmt={fmtGiB}
              color="rgb(var(--vw-model-5))"
            />
            {usage.tiers.nvme.capacity > 0 ? (
              <TierCard
                label="Disk"
                cap={`cap ${fmtGB(usage.tiers.nvme.capacity)}`}
                used={usage.tiers.nvme.used}
                dead={usage.tiers.nvme.dead}
                capacity={usage.tiers.nvme.capacity}
                fmt={fmtGB}
                color="rgb(var(--vw-model-6))"
              />
            ) : (
              <Notice tone="info">No disk tier: no disk folder is set (Settings, Prompt cache).</Notice>
            )}
          </div>
          <p className="m-0 mt-2 text-[12px] text-chat-dim">
            &quot;Waiting to be cleared&quot;: parts nobody can reach any more (a reset or switched key, a removed model). They
            are dropped first.
          </p>

          {usage.models.length > 0 && (
            <div className="mt-3 overflow-hidden rounded-lg border border-vw-rule-soft/60" data-testid="kv-usage-table">
              <div className={cn(ROW, "border-t-0 text-[11px] font-medium uppercase tracking-[.06em] text-chat-dim max-[759px]:hidden")}>
                <span>Model / key</span>
                <span className="text-right">Weight</span>
                <span className="text-right">Part</span>
                <span className="text-right">RAM</span>
                <span className="text-right">Disk</span>
                <span>Use against share · dropped</span>
              </div>
              {usage.models.map((m) => (
                <ModelRows
                  key={m.model_id}
                  m={m}
                  info={models[m.model_id]}
                  open={isOpen(m.model_id)}
                  onToggle={() => setOpen((o) => ({ ...o, [m.model_id]: !isOpen(m.model_id) }))}
                  tiers={usage.tiers!}
                  isolated={isolated}
                />
              ))}
            </div>
          )}
          <p className="m-0 mt-2 text-[12px] leading-[1.45] text-chat-dim">
            Part: the model&apos;s slice of each tier, from its weight. Share: the key&apos;s slice of its model&apos;s part, from
            priority + 1. The bar is use against share; the white tick is the share.
            {policy === "lru" && (
              <b className="font-semibold text-chat-muted">
                {" "}
                With oldest first, parts and shares do not decide what is dropped; they show what fair share would do.
              </b>
            )}
          </p>
        </>
      )}
    </section>
  );
}

function ModelRows({
  m,
  info,
  open,
  onToggle,
  tiers,
  isolated,
}: {
  m: KvModelUsage;
  info: KvModelInfo | undefined;
  open: boolean;
  onToggle: () => void;
  tiers: NonNullable<KvUsage["tiers"]>;
  isolated?: Set<string>;
}) {
  const name = info?.name ?? m.model_id;
  const ramOnly = info?.kv_store === "ram";
  const keys = m.tenants.length;
  return (
    <Fragment>
      <div className={cn(ROW, "bg-chat-surface-2/40")}>
        <button
          type="button"
          aria-expanded={open}
          onClick={onToggle}
          className="col-span-2 flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-left text-chat-fg min-[760px]:col-span-1"
        >
          <span aria-hidden="true" className="text-[10px] text-chat-dim">
            {open ? "▾" : "▸"}
          </span>
          <span className="!font-mono font-semibold">{name}</span>
        </button>
        <span className={NUM}>
          <span className={SMALL_LABEL}>Weight </span>
          {m.weight}
        </span>
        <span className={NUM}>
          <span className={SMALL_LABEL}>Part </span>
          {fmtPct(m.part.ram)}
        </span>
        <span className={NUM}>
          <span className={SMALL_LABEL}>RAM </span>
          {fmtGiB(m.used.ram)}
        </span>
        <span className={NUM}>
          <span className={SMALL_LABEL}>Disk </span>
          {ramOnly ? <b className="font-semibold">RAM only</b> : fmtGB(m.used.nvme)}
        </span>
        <span className="text-[12px] text-chat-dim">
          {keys} key{keys === 1 ? "" : "s"}
        </span>
      </div>
      {open &&
        m.tenants.map((t) => {
          const shareBytes = t.share.ram * m.part.ram * tiers.ram.capacity + t.share.nvme * m.part.nvme * tiers.nvme.capacity;
          const drops = t.evictions.ram + t.evictions.nvme;
          const by = t.evicted_by[0];
          return (
            <div key={t.token_id} className={ROW}>
              <span className="col-span-2 flex items-center gap-2 pl-5 min-[760px]:col-span-1">
                <span className="truncate">{tenantLabel(t.token_id, t.name)}</span>
                {isolated?.has(t.token_id) && <Mark tone="info">isolated</Mark>}
              </span>
              <span className="max-[759px]:hidden" />
              <span className={NUM}>
                <span className={SMALL_LABEL}>Share </span>
                {fmtPct(t.share.ram)}
              </span>
              <span className={NUM}>
                <span className={SMALL_LABEL}>RAM </span>
                {fmtGiB(t.used.ram)}
              </span>
              <span className={NUM}>
                <span className={SMALL_LABEL}>Disk </span>
                {fmtGB(t.used.nvme)}
              </span>
              <span className="col-span-2 flex items-center gap-2 min-[760px]:col-span-1">
                <ShareBar
                  used={t.used.total}
                  share={shareBytes}
                  label={`${tenantLabel(t.token_id, t.name)}: ${fmtShort(t.used.total)} of a ${fmtShort(shareBytes)} share`}
                  className="min-w-[60px] flex-1"
                />
                <span className="whitespace-nowrap text-[12px] tabular-nums text-chat-muted">
                  {drops > 0 && by ? `${fmtInt(drops)} · by ${tenantLabel(by.token_id, by.name)}` : fmtInt(drops)}
                </span>
              </span>
            </div>
          );
        })}
    </Fragment>
  );
}
