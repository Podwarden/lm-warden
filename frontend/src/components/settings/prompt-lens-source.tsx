"use client";

import { useEffect, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";

// ---------------------------------------------------------------------------
// Where the Prompt Lens value in effect comes from (GET /api/admin/prompt-lens/
// status): the switch on this page once it has been saved, else the env
// variable (VW_PROMPT_LENS_ENABLED),
// else the default (on). Refetched whenever the saved switch changes.
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
      <p className="text-xs text-slate-500">
        Prices for the money figures are in the <span className="text-slate-300">Prices</span> tab.
      </p>
    </div>
  );
}
