"use client";

// Settings, "Prices" tab (#347). What turns a Prompt Lens saving into money.
// - One section per provider the install uses (the router's upstream, every upstream seen
//   in the request history, every provider priced by hand), then the local models.
// - Prices are fetched daily from public lists (OpenRouter, else LiteLLM) and ship with the
//   release as a snapshot for offline installs. Every price shows its source and date.
// - An admin can override one price; the override is audited and wins. A model with no
//   known price shows tokens in Prompt Lens.
// Prices are per million tokens in the card's currency; the API carries micro-units.

import { useState } from "react";
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";
import { Skeleton } from "@/components/ui/skeleton";
import { btn } from "@/components/tokens/detail/styles";

const PRICES = "/api/admin/prices";
const AUDIT = "/api/admin/prices/audit";
const PER_M = 1_000_000;

export interface Price {
  currency: string;
  input_per_m_micros: number;
  output_per_m_micros: number | null;
  cache_read_per_m_micros: number | null;
  cache_write_per_m_micros: number | null;
}
export interface PriceEntry {
  model: string;
  price: Price | null;
  source: string | null;
  fetched_at: string | null;
  override?: { by: string | null; at: number | null };
  list_price?: Price | null;
  list_source?: string | null;
  requests?: number;
}
interface Provider {
  host: string;
  label: string;
  reasons: string[];
  catalogue_models: number;
  models: PriceEntry[];
}
interface LocalModel {
  id: string;
  name: string;
  reference: { provider: string; model: string | null } | null;
  priced_as: PriceEntry | null;
}
interface Refresh {
  interval_hours: number;
  network_allowed: boolean;
  last_attempt: number | null;
  last_success: number | null;
  last_error: string | null;
  last_source: string | null;
  catalogue_rows: number;
  snapshot: { source: string | null; fetched_at: string | null; rows: number };
  sources: string[];
}
interface PricesAnswer {
  providers: Provider[];
  local: LocalModel[];
  refresh: Refresh;
}
interface AuditEntry {
  ts: number;
  actor: string;
  action: string;
  provider: string | null;
  model: string | null;
  old: Record<string, unknown> | null;
  new: Record<string, unknown> | null;
}

const field =
  "rounded border border-slate-600 bg-slate-900 px-1.5 py-1 text-xs text-slate-200 disabled:opacity-50";

const perM = (micros: number | null | undefined): string => (micros == null ? "-" : String(micros / PER_M));
const toMicros = (text: string): number | null => {
  const t = text.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) && n >= 0 ? Math.round(n * PER_M) : NaN;
};

export function sourceLabel(source: string | null): string {
  switch (source) {
    case "override":
      return "set by you";
    case "openrouter":
      return "OpenRouter";
    case "litellm":
      return "LiteLLM";
    case "snapshot":
      return "bundled snapshot";
    default:
      return source ?? "";
  }
}

function priceText(p: Price): string {
  const parts = [`in ${perM(p.input_per_m_micros)}`, `out ${perM(p.output_per_m_micros)}`];
  if (p.cache_read_per_m_micros != null) parts.push(`cache read ${perM(p.cache_read_per_m_micros)}`);
  if (p.cache_write_per_m_micros != null) parts.push(`cache write ${perM(p.cache_write_per_m_micros)}`);
  return `${parts.join(" · ")} ${p.currency}/M`;
}

const day = (iso: string | null) => (iso ? iso.slice(0, 10) : "");
const when = (ts: number | null) => (ts ? new Date(ts * 1000).toLocaleString() : "never");

async function call(url: string, method: string, body?: unknown): Promise<string | null> {
  try {
    const r = await authFetch(url, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (r.ok) return null;
    let detail = `HTTP ${r.status}`;
    try {
      detail = detailOf(await r.json()) ?? detail;
    } catch {
      /* non-JSON body */
    }
    return detail;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
}

function PriceLine({ e }: { e: PriceEntry }) {
  if (!e.price) {
    return <span className="text-amber-400">no price known: Prompt Lens shows tokens</span>;
  }
  return (
    <span>
      {priceText(e.price)}
      <span className="text-slate-500">
        {" "}
        · {sourceLabel(e.source)}, {day(e.fetched_at)}
        {e.source === "override" && e.override?.by ? `, by ${e.override.by}` : ""}
      </span>
    </span>
  );
}

function OverrideEditor({
  provider,
  entry,
  onDone,
  onError,
}: {
  provider: string;
  entry: PriceEntry;
  onDone: () => void;
  onError: (m: string | null) => void;
}) {
  const p = entry.price;
  const [d, setD] = useState({
    input: p ? perM(p.input_per_m_micros) : "",
    output: p?.output_per_m_micros != null ? perM(p.output_per_m_micros) : "",
    read: p?.cache_read_per_m_micros != null ? perM(p.cache_read_per_m_micros) : "",
    write: p?.cache_write_per_m_micros != null ? perM(p.cache_write_per_m_micros) : "",
    currency: p?.currency ?? "USD",
  });
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof d) => (e: React.ChangeEvent<HTMLInputElement>) => setD({ ...d, [k]: e.target.value });

  async function save() {
    const input = toMicros(d.input);
    const output = toMicros(d.output);
    const read = toMicros(d.read);
    const write = toMicros(d.write);
    if (input === null || Number.isNaN(input)) return onError("The input price must be a number.");
    if (Number.isNaN(output) || Number.isNaN(read) || Number.isNaN(write))
      return onError("A price must be a number or empty.");
    setBusy(true);
    const err = await call(`${PRICES}/override`, "PUT", {
      provider,
      model: entry.model,
      currency: d.currency.trim() || "USD",
      input_per_m_micros: input,
      output_per_m_micros: output,
      cache_read_per_m_micros: read,
      cache_write_per_m_micros: write,
    });
    setBusy(false);
    onError(err);
    if (!err) onDone();
  }

  async function remove() {
    setBusy(true);
    const err = await call(`${PRICES}/override?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(entry.model)}`, "DELETE");
    setBusy(false);
    onError(err);
    if (!err) onDone();
  }

  return (
    <div className="mt-1 flex flex-wrap items-center gap-1.5">
      <input aria-label="Input price" placeholder="input / M" className={`${field} w-24`} value={d.input} onChange={set("input")} />
      <input aria-label="Output price" placeholder="output / M" className={`${field} w-24`} value={d.output} onChange={set("output")} />
      <input aria-label="Cache read price" placeholder="cache read / M" className={`${field} w-28`} value={d.read} onChange={set("read")} />
      <input aria-label="Cache write price" placeholder="cache write / M" className={`${field} w-28`} value={d.write} onChange={set("write")} />
      <input aria-label="Currency" className={`${field} w-14`} value={d.currency} onChange={set("currency")} />
      <button type="button" className={btn()} disabled={busy} onClick={() => void save()}>
        Save override
      </button>
      {entry.source === "override" && (
        <button type="button" className={btn()} disabled={busy} onClick={() => void remove()}>
          Remove override
        </button>
      )}
      <button type="button" className={btn()} disabled={busy} onClick={onDone}>
        Cancel
      </button>
    </div>
  );
}

function ModelRow({
  provider,
  entry,
  onChanged,
  onError,
}: {
  provider: string;
  entry: PriceEntry;
  onChanged: () => void;
  onError: (m: string | null) => void;
}) {
  const [editing, setEditing] = useState(false);
  return (
    <li data-testid="price-model" className="text-xs text-slate-300">
      <span className="font-medium text-slate-200">{entry.model}</span>: <PriceLine e={entry} />{" "}
      {!editing && (
        <button type="button" className="text-sky-400 hover:underline" onClick={() => setEditing(true)}>
          {entry.price ? "override" : "set price"}
        </button>
      )}
      {entry.source === "override" && entry.list_price && (
        <span className="text-slate-500"> · list price {priceText(entry.list_price)}</span>
      )}
      {editing && (
        <OverrideEditor
          provider={provider}
          entry={entry}
          onError={onError}
          onDone={() => {
            setEditing(false);
            onChanged();
          }}
        />
      )}
    </li>
  );
}

function ProviderSection({
  p,
  onChanged,
  onError,
}: {
  p: Provider;
  onChanged: () => void;
  onError: (m: string | null) => void;
}) {
  const [q, setQ] = useState("");
  const [found, setFound] = useState<PriceEntry[] | null>(null);
  async function search() {
    try {
      const r = await authFetchJSON<{ models: PriceEntry[] }>(
        `${PRICES}/models?provider=${encodeURIComponent(p.host)}&q=${encodeURIComponent(q)}&limit=20`,
      );
      setFound(r.models);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }
  return (
    <section data-testid={`price-provider-${p.host}`} className="space-y-1.5 border-t border-slate-800 pt-3">
      <h4 className="text-sm font-semibold text-slate-200">
        {p.label} <span className="text-xs font-normal text-slate-500">{p.host} · {p.reasons.join(", ")}</span>
      </h4>
      {p.catalogue_models === 0 && (
        <p className="text-xs text-slate-500">No public price list covers this provider: set prices by hand.</p>
      )}
      {p.models.length === 0 ? (
        <p className="text-xs text-slate-500">No requests relayed here yet. Find a model below to see or set its price.</p>
      ) : (
        <ul className="space-y-1">
          {p.models.map((m) => (
            <ModelRow key={m.model} provider={p.host} entry={m} onChanged={onChanged} onError={onError} />
          ))}
        </ul>
      )}
      {p.catalogue_models > 0 && (
        <div className="flex items-center gap-1.5">
          <input
            aria-label={`Find a ${p.label} model`}
            placeholder={`find a model (${p.catalogue_models} priced)`}
            className={`${field} w-56`}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void search()}
          />
          <button type="button" className={btn()} onClick={() => void search()}>
            Find
          </button>
        </div>
      )}
      {found && (
        <ul data-testid="price-found" className="space-y-1">
          {found.length === 0 && <li className="text-xs text-slate-500">No model matches.</li>}
          {found.map((m) => (
            <ModelRow
              key={m.model}
              provider={p.host}
              entry={m}
              onChanged={() => {
                onChanged();
                void search();
              }}
              onError={onError}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function LocalSection({ data, onChanged, onError }: { data: PricesAnswer; onChanged: () => void; onError: (m: string | null) => void }) {
  const [draft, setDraft] = useState<Record<string, { provider: string; model: string }>>({});
  if (data.local.length === 0) {
    return <p className="text-xs text-slate-500">No local models yet.</p>;
  }
  const hosts = data.providers.map((p) => p.host);
  async function save(id: string, ref: { provider: string; model: string } | null) {
    const err = await call(`/api/admin/lens/reference-card/${encodeURIComponent(id)}`, "PUT", ref ?? {});
    onError(err);
    if (!err) onChanged();
  }
  return (
    <ul className="space-y-2">
      {data.local.map((m) => {
        const d = draft[m.id] ?? { provider: m.reference?.provider ?? hosts[0] ?? "", model: m.reference?.model ?? "" };
        const set = (v: Partial<typeof d>) => setDraft({ ...draft, [m.id]: { ...d, ...v } });
        return (
          <li key={m.id} data-testid="price-local" className="text-xs text-slate-300">
            <span className="font-medium text-slate-200">{m.name}</span>:{" "}
            {m.priced_as && m.reference ? (
              <>
                priced as {m.reference.provider} · {m.reference.model}, <PriceLine e={m.priced_as} />
              </>
            ) : (
              <span className="text-slate-500">no reference price: Prompt Lens shows tokens and seconds</span>
            )}
            <div className="mt-1 flex flex-wrap items-center gap-1.5">
              <select
                aria-label={`Provider to price ${m.name} as`}
                className={field}
                value={d.provider}
                onChange={(e) => set({ provider: e.target.value })}
              >
                {hosts.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
              <input
                aria-label={`Model to price ${m.name} as`}
                placeholder="model name"
                className={`${field} w-52`}
                value={d.model}
                onChange={(e) => set({ model: e.target.value })}
              />
              <button type="button" className={btn()} disabled={!d.provider || !d.model.trim()} onClick={() => void save(m.id, { provider: d.provider, model: d.model.trim() })}>
                Price as this
              </button>
              {m.reference && (
                <button type="button" className={btn()} onClick={() => void save(m.id, null)}>
                  Clear
                </button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function RefreshBlock({ r, onChanged, onError }: { r: Refresh; onChanged: () => void; onError: (m: string | null) => void }) {
  const [hours, setHours] = useState(String(r.interval_hours));
  const [busy, setBusy] = useState(false);
  async function saveInterval() {
    const h = Number(hours);
    if (!Number.isInteger(h) || h < 0 || h > 720) return onError("The interval is a whole number of hours, 0 to 720.");
    const err = await call(`${PRICES}/settings`, "PUT", { refresh_interval_hours: h });
    onError(err);
    if (!err) onChanged();
  }
  async function refreshNow() {
    setBusy(true);
    const err = await call(`${PRICES}/refresh`, "POST");
    setBusy(false);
    onError(err);
    onChanged();
  }
  return (
    <div data-testid="price-refresh" className="space-y-1.5 text-xs text-slate-400">
      <p>
        Prices are read from public lists: {r.sources.join("; ")}. Nothing about your install is sent.{" "}
        {r.network_allowed
          ? r.interval_hours > 0
            ? `Refreshed every ${r.interval_hours} h.`
            : "Automatic refresh is off."
          : "This install is offline, so only the bundled snapshot is used."}
      </p>
      <p>
        Last refreshed: {when(r.last_success)}
        {r.last_source ? ` from ${sourceLabel(r.last_source)}` : ""}. {r.catalogue_rows} prices held; bundled snapshot of{" "}
        {day(r.snapshot.fetched_at)} ({r.snapshot.rows} prices
        {r.snapshot.source ? `, ${r.snapshot.source}` : ""}).
      </p>
      {r.last_error && (
        <p data-testid="price-refresh-error" className="text-amber-400">
          Last attempt failed ({when(r.last_attempt)}), the previous prices are kept: {r.last_error}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <label className="flex items-center gap-1.5">
          Refresh every
          <input aria-label="Refresh interval in hours" className={`${field} w-16`} value={hours} onChange={(e) => setHours(e.target.value)} />
          hours (0 = off)
        </label>
        <button type="button" className={btn()} onClick={() => void saveInterval()}>
          Save interval
        </button>
        <button type="button" className={btn()} disabled={busy} onClick={() => void refreshNow()}>
          Refresh now
        </button>
      </div>
    </div>
  );
}

function AuditList() {
  const { data } = useSWR<{ entries: AuditEntry[] }>(AUDIT, authFetchJSON, { revalidateOnFocus: false });
  if (!data || data.entries.length === 0) return <p className="text-xs text-slate-500">No changes yet.</p>;
  return (
    <ul data-testid="price-audit" className="space-y-0.5 text-xs text-slate-400">
      {data.entries.slice(0, 20).map((a, i) => (
        <li key={`${a.ts}-${i}`}>
          {new Date(a.ts * 1000).toLocaleString()} · {a.actor} · {a.action.replace("_", " ")}
          {a.provider ? ` · ${a.provider}` : ""}
          {a.model ? ` · ${a.model}` : ""}
        </li>
      ))}
    </ul>
  );
}

export function PricesTab() {
  const { data, mutate, error } = useSWR<PricesAnswer>(PRICES, authFetchJSON, { revalidateOnFocus: false });
  const [err, setErr] = useState<string | null>(null);
  if (error) return <p role="alert" className="text-sm text-red-400">Could not load prices.</p>;
  if (!data) return <Skeleton className="h-40 w-full" />;
  const changed = () => void mutate();
  return (
    <div data-testid="prices-tab" className="space-y-4">
      <header className="space-y-1">
        <h3 className="text-base font-semibold text-slate-100">Prices</h3>
        <p className="text-xs text-slate-500">
          Prompt Lens shows a saving in money from these prices: a relayed request uses its provider&apos;s price, a
          local model uses the reference price you pick. Prices are per million tokens. A model with no known price
          shows tokens instead.
        </p>
      </header>
      <RefreshBlock r={data.refresh} onChanged={changed} onError={setErr} />
      {err && (
        <p role="alert" className="text-xs text-red-400">
          {err}
        </p>
      )}
      {data.providers.length === 0 ? (
        <p className="text-xs text-slate-500">No provider is configured or in use yet.</p>
      ) : (
        data.providers.map((p) => <ProviderSection key={p.host} p={p} onChanged={changed} onError={setErr} />)
      )}
      <section data-testid="price-local-section" className="space-y-1.5 border-t border-slate-800 pt-3">
        <h4 className="text-sm font-semibold text-slate-200">Local models: reference prices</h4>
        <p className="text-xs text-slate-500">Price a local model as a provider&apos;s model to see what its requests would have cost.</p>
        <LocalSection data={data} onChanged={changed} onError={setErr} />
      </section>
      <section className="space-y-1 border-t border-slate-800 pt-3">
        <h4 className="text-sm font-semibold text-slate-200">Recent changes</h4>
        <AuditList />
      </section>
    </div>
  );
}
