"use client";

// Settings, "Prompt cache" tab (owner, 2026-11-22: its own tab). Three sections:
// - Store size (kv_store_ram_gib, kv_store_nvme_dir, kv_store_nvme_cap_gb): read when the
//   warden starts, so the section carries the restart chip.
// - What gets dropped first (kv_evict_policy lru | fair, kv_idle_ram_min, kv_idle_nvme_h):
//   applies now, pushed to the store. The idle times act only under fair share, so under
//   oldest first they are disabled and say why.
// - New keys (kv_isolation_default sharing | isolated): applies now, to new keys only.
// Edited in place; Save sends only what changed, Discard drops the edits.

import { useEffect, useState, type ReactNode } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";
import { Skeleton } from "@/components/ui/skeleton";
import { btn } from "@/components/tokens/detail/styles";
import { Mark } from "@/components/kv/parts";
import { cn } from "@/lib/utils";
import { fmtGiB, fmtGB, sinceAgo, type KvUsage } from "@/lib/kv";

const RUNTIME = "/api/settings/runtime";

type Key =
  | "kv_store_ram_gib"
  | "kv_store_nvme_dir"
  | "kv_store_nvme_cap_gb"
  | "kv_evict_policy"
  | "kv_idle_ram_min"
  | "kv_idle_nvme_h"
  | "kv_isolation_default";

type Draft = Record<Key, string>;

const DEFAULTS: Draft = {
  kv_store_ram_gib: "16",
  kv_store_nvme_dir: "",
  kv_store_nvme_cap_gb: "250",
  kv_evict_policy: "lru",
  kv_idle_ram_min: "120",
  kv_idle_nvme_h: "168",
  kv_isolation_default: "sharing",
};
const KEYS = Object.keys(DEFAULTS) as Key[];
const NUMERIC: Key[] = ["kv_store_ram_gib", "kv_store_nvme_cap_gb", "kv_idle_ram_min", "kv_idle_nvme_h"];

function fromRuntime(r: Record<string, string | null | undefined>): Draft {
  const out = { ...DEFAULTS };
  for (const k of KEYS) {
    const v = r[k];
    if (v != null && v !== "") out[k] = String(v);
  }
  return out;
}

interface KvStoreStatus {
  daemon: { healthy: boolean; generation: number; ram_requested_bytes: number | null; ram_granted_bytes: number | null };
}

function Section({ title, sub, chip, children }: { title: string; sub?: string; chip: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-vw-rule-soft/70 bg-chat-surface/55 px-4 py-3.5">
      <div className="flex flex-wrap items-start justify-between gap-2 border-b border-chat-rule pb-2.5">
        <div>
          <h3 className="m-0 text-[15px] font-semibold">{title}</h3>
          {sub && <p className="m-0 mt-0.5 text-[12.5px] text-chat-muted">{sub}</p>}
        </div>
        {chip}
      </div>
      <div className="divide-y divide-chat-rule">{children}</div>
    </section>
  );
}

function Field({ id, label, help, children }: { id?: string; label: string; help?: ReactNode; children: ReactNode }) {
  return (
    <div className="grid gap-2 py-3 min-[760px]:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] min-[760px]:gap-5">
      <div>
        {id ? (
          <label htmlFor={id} className="text-[13.5px] font-semibold">
            {label}
          </label>
        ) : (
          <span className="text-[13.5px] font-semibold">{label}</span>
        )}
        {help && <p className="m-0 mt-1 text-[12.5px] leading-[1.45] text-chat-muted">{help}</p>}
      </div>
      <div>{children}</div>
    </div>
  );
}

function NumberInput({ id, value, unit, disabled, onChange }: { id: string; value: string; unit: string; disabled?: boolean; onChange: (v: string) => void }) {
  return (
    <span className="inline-flex items-center gap-2">
      <input
        id={id}
        inputMode="numeric"
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-[84px] rounded-md border border-chat-rule bg-chat-page px-2 py-1.5 text-right text-[13px] tabular-nums text-chat-fg disabled:opacity-45"
      />
      <span className="text-[12.5px] text-chat-muted">{unit}</span>
    </span>
  );
}

function Choice<T extends string>({
  name,
  label,
  value,
  options,
  onChange,
}: {
  name: string;
  /** The group's accessible name. */
  label: string;
  value: T;
  options: { v: T; title: ReactNode; help: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid gap-2 min-[700px]:grid-cols-2">
      {options.map((o) => {
        const on = value === o.v;
        return (
          <label
            key={o.v}
            className={cn(
              "flex cursor-pointer gap-2.5 rounded-lg border px-3 py-2.5",
              on ? "border-chat-accent/60 bg-vw-amber-bg/25" : "border-vw-rule-soft/60",
            )}
          >
            <input
              type="radio"
              name={name}
              value={o.v}
              checked={on}
              onChange={() => onChange(o.v)}
              className="mt-1 accent-[rgb(var(--vw-btn))]"
            />
            <span>
              <span className="block text-[13.5px] font-semibold">{o.title}</span>
              <span className="block text-[12.5px] leading-[1.4] text-chat-muted">{o.help}</span>
            </span>
          </label>
        );
      })}
    </div>
  );
}

export function PromptCacheTab() {
  const { data, error, isLoading, mutate } = useSWR<Record<string, string | null>>(RUNTIME, authFetchJSON, {
    revalidateOnFocus: false,
  });
  const { mutate: globalMutate } = useSWRConfig();
  const { data: status } = useSWR<KvStoreStatus>("/api/system/kv-store", authFetchJSON, {
    refreshInterval: 15_000,
    shouldRetryOnError: false,
  });
  const { data: usage } = useSWR<KvUsage>("/api/kv/usage", authFetchJSON, {
    refreshInterval: 30_000,
    shouldRetryOnError: false,
  });
  const { data: models } = useSWR<{ models: { kv_store?: string }[] }>("/api/models", authFetchJSON, {
    shouldRetryOnError: false,
  });

  const [draft, setDraft] = useState<Draft | null>(null);
  const [seeded, setSeeded] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!data) return;
    const sig = JSON.stringify(fromRuntime(data));
    if (sig === seeded) return;
    setSeeded(sig);
    setDraft(fromRuntime(data));
  }, [data, seeded]);

  if (isLoading || (!data && !error)) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }
  if (error && !data) {
    return (
      <div role="alert" className="rounded-md border border-red-700 bg-red-900/30 p-4 text-sm text-red-200">
        Failed to load settings{error instanceof Error ? `: ${error.message}` : "."}
      </div>
    );
  }
  if (!data || !draft) return null;

  const snap = fromRuntime(data);
  const dirty = KEYS.filter((k) => draft[k].trim() !== snap[k]);
  const set = (k: Key) => (v: string) => {
    setDraft((d) => (d ? { ...d, [k]: v } : d));
    setNotice(null);
    setSaveError(null);
  };
  const fair = draft.kv_evict_policy === "fair";

  async function save() {
    if (!draft) return;
    const body: Record<string, unknown> = {};
    for (const k of dirty) {
      const v = draft[k].trim();
      if (NUMERIC.includes(k)) {
        if (!/^\d+$/.test(v)) {
          setSaveError("Sizes and times are whole numbers.");
          return;
        }
        body[k] = Number(v);
      } else body[k] = v;
    }
    setSaving(true);
    setSaveError(null);
    try {
      const r = await authFetch(RUNTIME, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          detail = detailOf(await r.json()) ?? detail;
        } catch {
          /* not JSON */
        }
        setSaveError(detail);
        return;
      }
      let kinds: string[] = [];
      try {
        const j = (await r.json()) as { requires_restart?: string[] };
        kinds = j.requires_restart ?? [];
      } catch {
        /* no body */
      }
      setNotice(
        kinds.includes("warden-restart")
          ? "Saved. Restart the warden for the store size to take effect."
          : "Saved. The store uses the new settings now.",
      );
      await mutate();
      void globalMutate("/api/kv/usage");
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const d = status?.daemon;
  const storeOn = models ? models.models.some((m) => m.kv_store && m.kv_store !== "off") : undefined;
  const state = d?.healthy ? "running" : storeOn === false ? "off" : status ? "down" : null;
  const ago = sinceAgo(usage?.since);
  const disk = usage?.tiers?.nvme;

  return (
    <div className="space-y-4">
      {state && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12.5px] text-chat-muted" data-testid="kv-store-status">
          <Mark tone={state === "running" ? "ok" : state === "down" ? "bad" : "idle"}>
            {state === "running"
              ? "KV cache store running"
              : state === "down"
                ? "KV cache store not answering"
                : "No model uses the store, so it is not running"}
          </Mark>
          {state === "running" && d && d.ram_granted_bytes != null && d.ram_requested_bytes != null && (
            <span>
              · RAM granted {fmtGiB(d.ram_granted_bytes)} of {Math.round(d.ram_requested_bytes / 1024 ** 3)} asked
              {disk && disk.capacity > 0 && (
                <>
                  {" "}
                  · disk {fmtGB(disk.used).replace(" GB", "")} of {fmtGB(disk.capacity)} used
                </>
              )}
              {ago && <> · started {ago}</>}
            </span>
          )}
        </div>
      )}

      <Section
        title="Store size"
        sub="How much RAM and disk the store may use. Read when the warden starts."
        chip={<Mark tone="warn">Requires warden restart</Mark>}
      >
        <Field id="kv-ram-gib" label="RAM tier size" help="The store gets less when the host is short of memory; the status line shows what it got.">
          <NumberInput id="kv-ram-gib" value={draft.kv_store_ram_gib} unit="GiB" onChange={set("kv_store_ram_gib")} />
        </Field>
        <Field id="kv-nvme-dir" label="Disk folder" help="Empty means no disk tier. Use a fast local disk.">
          <input
            id="kv-nvme-dir"
            value={draft.kv_store_nvme_dir}
            placeholder="/path/to/fast/disk"
            onChange={(e) => set("kv_store_nvme_dir")(e.target.value)}
            className="w-full max-w-[320px] rounded-md border border-chat-rule bg-chat-page px-2.5 py-1.5 !font-mono text-[13px] text-chat-fg"
          />
        </Field>
        <Field id="kv-nvme-cap" label="Disk tier size" help="Most the store may write to that folder.">
          <NumberInput id="kv-nvme-cap" value={draft.kv_store_nvme_cap_gb} unit="GB" onChange={set("kv_store_nvme_cap_gb")} />
        </Field>
      </Section>

      <Section
        title="What gets dropped first"
        sub="When a tier is full, the store drops a cached part to make room."
        chip={<Mark tone="ok">Applies now</Mark>}
      >
        <Field label="Eviction policy">
          <Choice
            name="kv-evict-policy"
            label="Eviction policy"
            value={draft.kv_evict_policy as "lru" | "fair"}
            onChange={set("kv_evict_policy")}
            options={[
              { v: "lru", title: "Oldest first", help: "Drops the part used least recently, whoever wrote it. Default." },
              {
                v: "fair",
                title: (
                  <>
                    Fair share{" "}
                    <Mark tone="warn" className="ml-1 align-middle">
                      trial
                    </Mark>
                  </>
                ),
                help: "Splits each tier between models by weight, then between keys by priority + 1, and drops from whoever is most over their share.",
              },
            ]}
          />
        </Field>
        <Field id="kv-idle-ram" label="Keep recent parts in RAM for" help="Under fair share, a part used within this time is dropped only after idle parts. Default 120.">
          <NumberInput id="kv-idle-ram" value={draft.kv_idle_ram_min} unit="minutes" disabled={!fair} onChange={set("kv_idle_ram_min")} />
        </Field>
        <Field id="kv-idle-nvme" label="Keep recent parts on disk for" help="The same on disk. Default 168 (7 days).">
          <NumberInput id="kv-idle-nvme" value={draft.kv_idle_nvme_h} unit="hours" disabled={!fair} onChange={set("kv_idle_nvme_h")} />
        </Field>
        {!fair && (
          <p className="m-0 py-3 text-[12.5px] text-chat-muted">The two times, key caps and model weights only work with fair share.</p>
        )}
      </Section>

      <Section title="New keys" chip={<Mark tone="ok">Applies now</Mark>}>
        <Field label="Cache for new keys" help="Existing keys keep their setting. The web chat always shares.">
          <Choice
            name="kv-isolation-default"
            label="Cache for new keys"
            value={draft.kv_isolation_default as "sharing" | "isolated"}
            onChange={set("kv_isolation_default")}
            options={[
              { v: "sharing", title: "Shared", help: "New keys reuse each other's cached prompts. Fastest. Default." },
              { v: "isolated", title: "Isolated", help: "Each new key has its own cache. Slower first turns; no key can tell what another sent." },
            ]}
          />
        </Field>
      </Section>

      <div className="flex flex-wrap items-center justify-end gap-3">
        {saveError && (
          <span role="alert" className="text-[12.5px] text-vw-danger-fg">
            {saveError}
          </span>
        )}
        {!saveError && notice && (
          <span role="status" className="text-[12.5px] text-chat-muted">
            {notice}
          </span>
        )}
        {dirty.length > 0 && <span className="text-[12.5px] text-chat-accent">Unsaved</span>}
        <button
          type="button"
          className={btn()}
          disabled={dirty.length === 0 || saving}
          onClick={() => {
            setDraft(snap);
            setSaveError(null);
          }}
        >
          Discard
        </button>
        <button type="button" className={btn("primary")} disabled={dirty.length === 0 || saving} onClick={() => void save()}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </div>
  );
}
