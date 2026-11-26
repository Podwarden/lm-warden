"use client";

import { useEffect, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";

// ---------------------------------------------------------------------------
// Where the Prompt Lens value in effect comes from (GET /api/admin/prompt-lens/
// status): the switch on this page once it has been saved, else the env
// variable (VW_PROMPT_LENS_ENABLED),
// else the default (off). Refetched whenever the saved switch changes.
// "Use server default" clears the saved switch (PATCH null), so the env value
// applies again; the last save (who, when, before and after) is shown.
//
// While the env decides, the checkbox already shows the env's value, so
// ticking it changes nothing and the tab's Save (dirty fields only) sends
// nothing: "Save as a setting here" stores the value in effect instead (owner
// report 2026-11-23). It follows the switch's rules: edit mode only, and the
// server takes it from a signed-in session only (403 session_only otherwise).
// Outside edit mode a hint says how to change the greyed switch.
// ---------------------------------------------------------------------------

export interface PromptLensChange {
  ts: number;
  actor: string;
  via: string;
  old_enabled: boolean;
  new_enabled: boolean;
  old_source: string;
  new_source: string;
}

export interface PromptLensStatus {
  enabled: boolean;
  source: "setting" | "env" | "default";
  setting?: boolean | null;
  env_var?: string | null;
  env_value?: boolean | null;
  last_change?: PromptLensChange | null;
}

const NEW_ENV = "VW_PROMPT_LENS_ENABLED";
const STATUS_KEY = "/api/admin/prompt-lens/status";
const RUNTIME_KEY = "/api/settings/runtime";

export function promptLensSourceText(s: PromptLensStatus): string {
  const state = s.enabled ? "On" : "Off";
  const envName = s.env_var;
  if (s.source === "setting") {
    const over = s.env_var ? `; overrides ${envName} (${s.env_value ? "on" : "off"})` : "";
    return `${state}, set here${over}.`;
  }
  if (s.source === "env") {
    return `${state}, from ${envName} in the server's environment, until you save a value here.`;
  }
  return `${state} by default (no ${NEW_ENV}). Saving here turns it on or off without a restart.`;
}

function changeText(c: PromptLensChange): string {
  const when = new Date(c.ts * 1000).toLocaleString();
  const onOff = (b: boolean) => (b ? "on" : "off");
  return `Last changed by ${c.actor} (${c.via}), ${onOff(c.old_enabled)} to ${onOff(c.new_enabled)}, ${when}.`;
}

export function PromptLensSource({ saved, editing = true }: { saved: boolean | undefined; editing?: boolean }) {
  const { data, mutate } = useSWR<PromptLensStatus>(STATUS_KEY, authFetchJSON, {
    revalidateOnFocus: false,
  });
  const { mutate: globalMutate } = useSWRConfig();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    void mutate();
  }, [saved, mutate]);
  if (!data) return null;

  async function patch(value: boolean | null) {
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(RUNTIME_KEY, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_lens_enabled: value }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          detail = detailOf(await r.json()) ?? detail;
        } catch {
          /* non-JSON body */
        }
        setError(detail);
        return;
      }
      await Promise.all([mutate(), globalMutate(RUNTIME_KEY)]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const btn =
    "rounded border border-slate-600 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";
  return (
    <div className="space-y-1.5">
      {!editing && (
        <p data-testid="prompt-lens-edit-hint" className="text-xs text-slate-500">
          Press Edit to change.
        </p>
      )}
      <p data-testid="prompt-lens-source" className="text-xs text-slate-400">
        In effect: {promptLensSourceText(data)}
        {data.last_change ? ` ${changeText(data.last_change)}` : ""}
      </p>
      {data.source === "setting" && (
        <button type="button" className={btn} disabled={busy} onClick={() => void patch(null)}>
          Use server default
        </button>
      )}
      {data.source === "env" && (
        <button
          type="button"
          className={btn}
          disabled={busy || !editing}
          title={editing ? undefined : "Press Edit to change"}
          onClick={() => void patch(data.enabled)}
        >
          Save as a setting here ({data.enabled ? "on" : "off"})
        </button>
      )}
      {error && (
        <p role="alert" className="text-xs text-red-400">
          {error}
        </p>
      )}
      <PriceCards editing={editing} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Price cards (analysis spec 5.4, Task 2.2): what turns a saving into money on
// the scorecard. A relayed route uses the provider's card; an own model shows
// money only when you pick a reference card for it ("priced as ..."); the
// card's effective date is shown next to every money figure. The shipped list
// prices are a starting point you confirm (edit them first: prices change).
// Prices are entered per million tokens in the card's currency.
// ---------------------------------------------------------------------------

const CARDS_KEY = "/api/admin/lens/rate-cards";
const MODELS_KEY = "/api/models";
const PER_M = 1_000_000;

export interface PriceCard {
  id?: string;
  provider: string;
  model: string;
  currency: string;
  input_per_m_micros: number;
  output_per_m_micros: number | null;
  cache_read_per_m_micros: number | null;
  cache_write_per_m_micros: number | null;
  effective_from: string;
  source?: string;
}

interface CardsAnswer {
  cards: PriceCard[];
  defaults: PriceCard[];
  reference: Record<string, { provider: string | null; model: string | null }>;
}

const price = (micros: number | null | undefined): string => (micros == null ? "" : String(micros / PER_M));
const micros = (text: string): number | null => {
  const t = text.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) && n >= 0 ? Math.round(n * PER_M) : NaN;
};
const cardKey = (c: { provider: string | null; model: string | null }) => `${c.provider ?? ""}\u0000${c.model ?? ""}`;

interface Draft {
  provider: string;
  model: string;
  currency: string;
  input: string;
  output: string;
  read: string;
  write: string;
  effective_from: string;
}

const EMPTY: Draft = { provider: "", model: "", currency: "USD", input: "", output: "", read: "", write: "", effective_from: "" };

function draftOf(c: PriceCard): Draft {
  return {
    provider: c.provider,
    model: c.model,
    currency: c.currency,
    input: price(c.input_per_m_micros),
    output: price(c.output_per_m_micros),
    read: price(c.cache_read_per_m_micros),
    write: price(c.cache_write_per_m_micros),
    effective_from: c.effective_from,
  };
}

function PriceCards({ editing }: { editing: boolean }) {
  const { data: answer, mutate } = useSWR<Partial<CardsAnswer>>(CARDS_KEY, authFetchJSON, { revalidateOnFocus: false });
  const models = useSWR<{ models: { id: string; served_model_name: string }[] }>(MODELS_KEY, authFetchJSON, {
    revalidateOnFocus: false,
  });
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // A server without the route, or one answering oddly, shows no block at all.
  if (!answer || !Array.isArray(answer.cards)) return null;
  const data: CardsAnswer = { cards: answer.cards, defaults: answer.defaults ?? [], reference: answer.reference ?? {} };

  async function send(url: string, body: unknown): Promise<boolean> {
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          detail = detailOf(await r.json()) ?? detail;
        } catch {
          /* non-JSON body */
        }
        setError(detail);
        return false;
      }
      await mutate();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    const input = micros(draft.input);
    const output = micros(draft.output);
    const read = micros(draft.read);
    const write = micros(draft.write);
    if (!draft.provider.trim() || !draft.model.trim()) return setError("Give the provider and the model.");
    if (input === null || Number.isNaN(input)) return setError("The input price must be a number.");
    if (output === null || Number.isNaN(output)) return setError("The output price must be a number.");
    if (Number.isNaN(read) || Number.isNaN(write)) return setError("A cache price must be a number or empty.");
    if (!/^\d{4}-\d{2}-\d{2}$/.test(draft.effective_from)) return setError("The effective date is YYYY-MM-DD.");
    const ok = await send(CARDS_KEY, {
      cards: [
        {
          provider: draft.provider.trim(),
          model: draft.model.trim(),
          currency: draft.currency.trim() || "USD",
          input_per_m_micros: input,
          output_per_m_micros: output,
          cache_read_per_m_micros: read,
          cache_write_per_m_micros: write,
          effective_from: draft.effective_from,
        },
      ],
    });
    if (ok) {
      setDraft(EMPTY);
      setNote("Saved.");
    }
  }

  async function reference(modelId: string, value: string) {
    const body = value ? { provider: value.split("\u0000")[0], model: value.split("\u0000")[1] } : {};
    await send(`/api/admin/lens/reference-card/${encodeURIComponent(modelId)}`, body);
  }

  const field = "w-full rounded border border-slate-600 bg-slate-900 px-1.5 py-1 text-xs text-slate-200 disabled:opacity-50";
  const btn =
    "rounded border border-slate-600 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";
  const set = (k: keyof Draft) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setNote(null);
    setDraft((d) => ({ ...d, [k]: e.target.value }));
  };
  const disabled = busy || !editing;
  return (
    <div data-testid="price-cards" className="mt-3 space-y-2 border-t border-slate-800 pt-3">
      <h4 className="text-xs font-semibold text-slate-300">Price cards</h4>
      <p className="text-xs text-slate-500">
        Prompt Lens shows a saving in money only with a price card: a relayed route uses its provider&apos;s card; an
        own model needs a reference card (&ldquo;priced as ...&rdquo;). Prices are per million tokens. Every money
        figure shows the card&apos;s effective date.
      </p>
      {data.cards.length === 0 ? (
        <p data-testid="price-cards-empty" className="text-xs text-slate-500">
          No price cards yet.
        </p>
      ) : (
        <ul data-testid="price-card-list" className="space-y-0.5 text-xs text-slate-300">
          {data.cards.map((c) => (
            <li key={c.id ?? `${c.provider}${c.model}${c.effective_from}`} data-testid="price-card">
              {c.provider} · {c.model}: in {price(c.input_per_m_micros)} · out {price(c.output_per_m_micros)}
              {c.cache_read_per_m_micros != null ? ` · read ${price(c.cache_read_per_m_micros)}` : ""}
              {c.cache_write_per_m_micros != null ? ` · write ${price(c.cache_write_per_m_micros)}` : ""} {c.currency},
              effective {c.effective_from}{" "}
              <button type="button" className="text-sky-400 hover:underline disabled:opacity-50" disabled={!editing} onClick={() => setDraft(draftOf(c))}>
                change
              </button>
            </li>
          ))}
        </ul>
      )}
      {data.defaults.length > 0 && (
        <div data-testid="price-defaults" className="text-xs text-slate-400">
          Shipped list prices (not used until you confirm them):{" "}
          {data.defaults.map((c) => (
            <button
              key={`${c.provider}${c.model}`}
              type="button"
              disabled={!editing}
              title={c.source}
              className="mr-2 text-sky-400 hover:underline disabled:opacity-50"
              onClick={() => setDraft(draftOf(c))}
            >
              {c.model} ({c.effective_from})
            </button>
          ))}
        </div>
      )}
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        <input aria-label="Provider" placeholder="provider" className={field} value={draft.provider} onChange={set("provider")} disabled={disabled} />
        <input aria-label="Model" placeholder="model" className={field} value={draft.model} onChange={set("model")} disabled={disabled} />
        <input aria-label="Currency" placeholder="USD" className={field} value={draft.currency} onChange={set("currency")} disabled={disabled} />
        <input aria-label="Effective from" placeholder="YYYY-MM-DD" className={field} value={draft.effective_from} onChange={set("effective_from")} disabled={disabled} />
        <input aria-label="Input price" placeholder="input / M" className={field} value={draft.input} onChange={set("input")} disabled={disabled} />
        <input aria-label="Output price" placeholder="output / M" className={field} value={draft.output} onChange={set("output")} disabled={disabled} />
        <input aria-label="Cache read price" placeholder="cache read / M" className={field} value={draft.read} onChange={set("read")} disabled={disabled} />
        <input aria-label="Cache write price" placeholder="cache write / M" className={field} value={draft.write} onChange={set("write")} disabled={disabled} />
      </div>
      <button type="button" className={btn} disabled={disabled} onClick={() => void save()}>
        Add or replace price card
      </button>
      {note && <span className="ml-2 text-xs text-emerald-400">{note}</span>}
      {(models.data?.models ?? []).length > 0 && data.cards.length > 0 && (
        <div data-testid="reference-cards" className="space-y-1">
          <h5 className="text-xs font-semibold text-slate-300">Reference card per own model</h5>
          {(models.data?.models ?? []).map((m) => {
            const ref = data.reference[m.id];
            return (
              <label key={m.id} className="flex items-center gap-2 text-xs text-slate-300">
                <span className="w-40 truncate">{m.served_model_name}</span>
                <select
                  aria-label={`Reference card for ${m.served_model_name}`}
                  className={field}
                  disabled={disabled}
                  value={ref ? cardKey(ref) : ""}
                  onChange={(e) => void reference(m.id, e.target.value)}
                >
                  <option value="">none: tokens and seconds only</option>
                  {[...new Map(data.cards.map((c) => [cardKey(c), c])).values()].map((c) => (
                    <option key={cardKey(c)} value={cardKey(c)}>
                      priced as {c.provider} · {c.model}
                    </option>
                  ))}
                </select>
              </label>
            );
          })}
        </div>
      )}
      {error && (
        <p role="alert" className="text-xs text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
