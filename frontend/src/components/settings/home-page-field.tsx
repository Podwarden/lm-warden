"use client";

import { RUNTIME_HINTS } from "@/lib/settings-hints";
import type {
  Draft,
  RuntimeKey,
} from "@/components/settings/hooks/use-runtime-settings";

// ---------------------------------------------------------------------------
// HomePageField — what the instance's address (`/`) shows (spec 2026-11-04).
// An Off / Built-in radio, plus what Off answers (the console, or a 404).
// A mode this control does not offer (an edition's own) is left untouched:
// neither radio is checked and the off action stays editable only for Off.
// ---------------------------------------------------------------------------

interface Props {
  draft: Draft;
  setField: <K extends RuntimeKey>(k: K, v: Draft[K]) => void;
  disabled: boolean;
}

const MODES = [
  { value: "off", label: "Off" },
  { value: "builtin", label: "Built-in" },
] as const;

const OFF_ACTIONS = [
  { value: "console", label: "Open the console" },
  { value: "404", label: "Not found (404)" },
] as const;

export function HomePageField({ draft, setField, disabled }: Props) {
  const modeHint = RUNTIME_HINTS.home_page_mode;
  const actionHint = RUNTIME_HINTS.home_page_off_action;
  return (
    <div className="space-y-4">
      <fieldset className="space-y-1" disabled={disabled}>
        <legend className="font-medium text-slate-200">{modeHint.label}</legend>
        <p id="home-page-mode-hint" className="text-xs text-slate-500">
          {modeHint.hint}
        </p>
        <div className="flex gap-4 pt-1" role="radiogroup" aria-describedby="home-page-mode-hint">
          {MODES.map((m) => (
            <label key={m.value} className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
              <input
                type="radio"
                name="home_page_mode"
                value={m.value}
                checked={draft.home_page_mode === m.value}
                onChange={() => setField("home_page_mode", m.value)}
              />
              {m.label}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="space-y-1">
        <label htmlFor="home-page-off-action" className="font-medium text-slate-200">
          {actionHint.label}
        </label>
        <p id="home-page-off-action-hint" className="text-xs text-slate-500">
          {actionHint.hint}
        </p>
        <select
          id="home-page-off-action"
          aria-describedby="home-page-off-action-hint"
          className="block rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
          value={draft.home_page_off_action}
          disabled={disabled || draft.home_page_mode !== "off"}
          onChange={(e) => setField("home_page_off_action", e.target.value)}
        >
          {OFF_ACTIONS.map((a) => (
            <option key={a.value} value={a.value}>
              {a.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
