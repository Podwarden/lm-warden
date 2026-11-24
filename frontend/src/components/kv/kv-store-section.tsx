"use client";

// A model's cache store settings (model settings page):
// - "Cache share": the weight (kv_store_weight, 1 to 100), a LIVE field that goes to
//   the store at once. vLLM only. Like key caps and the idle times it acts only under
//   the fair eviction policy, so under oldest first it is disabled and says why.
// - "Cache store": off / RAM / RAM + disk (kv_store). It changes the engine's flags, so
//   it sits in the Engine band and is locked while the model is loaded. RAM + disk is
//   refused while no disk folder is set (owner, 2026-11-22), pointing to the server
//   settings. Then, in words, whether isolated keys can use the model.

import { useId, useState } from "react";
import { cn } from "@/lib/utils";
import { fmtPct, type CacheIsolation, type EvictPolicy, type KvStoreMode } from "@/lib/kv";

const CARD = "rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55 px-4 py-3.5";
const HELP = "m-0 text-[12.5px] leading-[1.45] text-chat-muted";

export function isVllm(backend: string | null | undefined): boolean {
  return (backend ?? "vllm") === "vllm";
}

export function KvWeightCard({
  weight,
  part,
  kvStore,
  policy,
  disabled,
  dirty,
  onWeight,
}: {
  weight: number;
  /** The model's part of each tier (from /api/kv/usage), when the store reports it. */
  part: number | null;
  kvStore: KvStoreMode;
  policy: EvictPolicy;
  disabled?: boolean;
  dirty?: boolean;
  onWeight: (w: number) => void;
}) {
  const id = useId();
  const off = policy !== "fair";
  // The text as typed; a whole number from 1 to 100 goes out at once, anything else is
  // clamped when the field loses focus (so clearing it to type "5" does not jump to 1).
  const [text, setText] = useState(String(weight));
  const [seen, setSeen] = useState(weight);
  if (weight !== seen) {
    setSeen(weight);
    setText(String(weight));
  }
  function commit() {
    const n = Math.round(Number(text));
    const v = text.trim() === "" || !Number.isFinite(n) ? 1 : Math.min(100, Math.max(1, n));
    setText(String(v));
    if (v !== weight || String(v) !== text) onWeight(v);
  }
  return (
    <section className={CARD} aria-label="Cache share" data-testid="section-kv-weight">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="m-0 flex items-center gap-2 text-[14px] font-semibold">
          Cache share
          {dirty && <span aria-label="changed" className="h-[7px] w-[7px] rounded-full bg-chat-accent" />}
        </h3>
        <span className="text-[12px] text-chat-dim">
          weight {weight}
          {part != null && <> · {fmtPct(part)} of each tier</>}
        </span>
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <label htmlFor={id} className="text-[13px]">
          Weight
        </label>
        <span className="inline-flex items-center gap-2">
          <input
            id={id}
            type="number"
            min={1}
            max={100}
            step={1}
            value={text}
            disabled={disabled || off}
            onChange={(e) => {
              const t = e.target.value;
              setText(t);
              if (/^\d+$/.test(t.trim())) {
                const n = Number(t);
                if (n >= 1 && n <= 100) onWeight(n);
              }
            }}
            onBlur={commit}
            className="w-[84px] rounded-md border border-chat-rule bg-chat-page px-2 py-1 text-right text-[13px] tabular-nums text-chat-fg disabled:opacity-45"
          />
          <span className="text-[12px] text-chat-dim">1 to 100</span>
        </span>
      </div>
      <p className={cn(HELP, "mt-2")}>
        This model&apos;s part of the RAM and disk tiers, against the other models that use the store. Two models with
        weights 3 and 1 get 75 % and 25 %. With one model it changes nothing.
        {kvStore === "off" && <> Turn the KV cache store on below to use it.</>}
      </p>
      {off && (
        <p className={cn(HELP, "mt-1")}>
          <b className="font-semibold text-chat-fg">Has no effect now:</b> weights work only with the fair eviction policy
          (Settings, Prompt cache).
        </p>
      )}
    </section>
  );
}

const OPTIONS: { v: KvStoreMode; title: string; help: string }[] = [
  { v: "off", title: "Off", help: "Prompts are cached on the GPU only." },
  { v: "ram", title: "RAM", help: "Also keeps them in host RAM. Lost when the server restarts." },
  { v: "ram+nvme", title: "RAM + disk", help: "Also writes them to disk. Survives a restart; reloads a long prompt in seconds." },
  {
    v: "nvme",
    title: "Disk only",
    help: "Writes them to disk and keeps about 1 GiB of RAM to move them. Survives a restart. For hosts with little spare RAM.",
  },
];

const DISK_MODES: KvStoreMode[] = ["ram+nvme", "nvme"];

export function KvStoreCard({
  backend,
  loaded,
  kvStore,
  nvmeDir,
  cacheIsolation,
  disabled,
  dirty,
  onStore,
}: {
  backend: string | null | undefined;
  loaded: boolean;
  kvStore: KvStoreMode;
  /** The server's disk folder (runtime kv_store_nvme_dir); "" or null: no disk tier. */
  nvmeDir: string | null | undefined;
  cacheIsolation: CacheIsolation | undefined;
  disabled?: boolean;
  dirty?: boolean;
  onStore: (v: KvStoreMode) => void;
}) {
  const name = useId();
  const vllm = isVllm(backend);
  const engine = backend === "llamacpp" ? "llama.cpp" : backend === "mlx" ? "MLX" : !backend || backend === "vllm" ? "vLLM" : backend;
  const noDisk = nvmeDir !== undefined && !(nvmeDir ?? "").trim();
  const summary = vllm ? (kvStore === "off" ? "off" : kvStore === "ram" ? "RAM" : kvStore === "nvme" ? "disk only" : "RAM + disk") : "not available";
  return (
    <section className={CARD} aria-label="KV cache store" data-testid="section-kv-store">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="m-0 flex items-center gap-2 text-[14px] font-semibold">
          KV cache store
          {dirty && <span aria-label="changed" className="h-[7px] w-[7px] rounded-full bg-chat-accent" />}
        </h3>
        <span className="text-[12px] text-chat-dim">{summary}</span>
      </div>
      <div role="radiogroup" aria-label="KV cache store" className="mt-3 grid gap-2 grid-cols-2 min-[700px]:grid-cols-4">
        {OPTIONS.map((o) => {
          const refused = (!vllm && o.v !== "off") || (DISK_MODES.includes(o.v) && noDisk);
          const off = disabled || loaded || refused;
          const on = kvStore === o.v;
          return (
            <label
              key={o.v}
              className={cn(
                "flex cursor-pointer gap-2.5 rounded-lg border px-3 py-2.5",
                on ? "border-chat-accent/60 bg-vw-amber-bg/25" : "border-vw-rule-soft/60",
                off && "cursor-not-allowed opacity-60",
              )}
            >
              <input
                type="radio"
                name={name}
                value={o.v}
                checked={on}
                disabled={off}
                onChange={() => onStore(o.v)}
                className="mt-1 accent-[rgb(var(--vw-btn))]"
              />
              <span>
                <span className="block text-[13.5px] font-semibold">{o.title}</span>
                <span className="block text-[12.5px] leading-[1.4] text-chat-muted">
                  {!vllm && o.v !== "off" ? "vLLM models only." : !vllm ? "Prompts are cached by the engine only." : o.help}
                </span>
              </span>
            </label>
          );
        })}
      </div>
      {vllm && noDisk && (
        <p className={cn(HELP, "mt-2")}>
          No disk folder is set, so RAM + disk and Disk only are not offered. Set one in Settings, Prompt cache, then restart the warden.
        </p>
      )}
      {vllm && (
        <p className={cn(HELP, "mt-2")}>
          {loaded && <>🔒 Unload the model to change this. </>}Sizes are set in Settings, Prompt cache. Prompts under 2,048
          tokens are never stored.
        </p>
      )}
      {cacheIsolation && (
        <p className="m-0 mt-3 rounded-lg border border-vw-rule-soft/60 px-3 py-2 text-[12.5px] leading-[1.45]">
          {cacheIsolation === "isolated" ? (
            <>
              Isolated keys: <b className="font-semibold">served.</b> {engine} keeps each isolated key&apos;s cache apart on
              the GPU, in RAM and on disk.
            </>
          ) : (
            <>
              Isolated keys: <b className="font-semibold">refused.</b> {engine} cannot keep one key&apos;s cache apart from
              another&apos;s, so requests with an isolated key get 403.
            </>
          )}
        </p>
      )}
    </section>
  );
}

/** Both cards together, the shape the model settings page shows them in. */
export function KvStoreSection({
  backend,
  loaded,
  kvStore,
  weight,
  nvmeDir,
  cacheIsolation,
  part,
  policy,
  onStore,
  onWeight,
}: {
  backend: string | null | undefined;
  loaded: boolean;
  kvStore: KvStoreMode;
  weight: number;
  nvmeDir: string | null | undefined;
  cacheIsolation: CacheIsolation | undefined;
  part: number | null;
  policy: EvictPolicy;
  onStore: (v: KvStoreMode) => void;
  onWeight: (w: number) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {isVllm(backend) && <KvWeightCard weight={weight} part={part} kvStore={kvStore} policy={policy} onWeight={onWeight} />}
      <KvStoreCard
        backend={backend}
        loaded={loaded}
        kvStore={kvStore}
        nvmeDir={nvmeDir}
        cacheIsolation={cacheIsolation}
        onStore={onStore}
      />
    </div>
  );
}
