"use client";

// The guard's settings form, shared by the platform default and a key's own settings. Controlled:
// the parent owns the value and saves it. `mode` "off" is offered only where off is a value of
// this form (the platform default); a key switches the guard off with its own choice instead.

import { useId } from "react";

import { G } from "@/lib/guard/strings";
import type { CustomRule, GuardSettings, PacksResponse } from "@/lib/guard/types";
import { cn } from "@/lib/utils";

export type FormMode = "off" | GuardSettings["mode"];

interface Props {
  value: GuardSettings;
  mode: FormMode;
  allowOff?: boolean;
  packs: PacksResponse | undefined;
  onChange: (value: GuardSettings, mode: FormMode) => void;
}

const FIELD = "rounded-md border border-chat-rule bg-chat-page text-chat-fg px-2 py-1 text-[13px]";
const LEGEND = "mb-1.5 text-[13px] font-semibold";
const HINT = "text-[12px] text-chat-dim";

function Radio({ name, label, checked, onChange }: { name: string; label: string; checked: boolean; onChange: () => void }) {
  return (
    <label className={cn("flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1 text-[13px]", checked ? "border-chat-accent" : "border-chat-rule")}>
      <input type="radio" name={name} checked={checked} onChange={onChange} />
      {label}
    </label>
  );
}

function Choice<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly { value: string; label: string }[]; onChange: (v: T) => void }) {
  const id = useId();
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <label htmlFor={id} className="text-[13px]">
        {label}
      </label>
      <select id={id} className={FIELD} value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function RuleRow({ rule, onChange, onRemove }: { rule: CustomRule; onChange: (r: CustomRule) => void; onRemove: () => void }) {
  const idId = useId();
  const valId = useId();
  return (
    <div className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_auto] items-start gap-2 max-[640px]:grid-cols-1">
      <div>
        <label htmlFor={idId} className={HINT}>
          {G.rules.idLabel}
        </label>
        <input id={idId} className={cn(FIELD, "w-full")} value={rule.id} maxLength={48} onChange={(e) => onChange({ ...rule, id: e.target.value })} />
      </div>
      <div>
        <label htmlFor={valId} className={HINT}>
          {rule.type === "regex" ? G.rules.patternLabel : G.rules.wordsLabel}
        </label>
        {rule.type === "regex" ? (
          <input id={valId} className={cn(FIELD, "w-full font-mono")} value={rule.regex ?? ""} maxLength={512} onChange={(e) => onChange({ ...rule, regex: e.target.value })} />
        ) : (
          <textarea
            id={valId}
            className={cn(FIELD, "w-full font-mono")}
            rows={3}
            value={(rule.values ?? []).join("\n")}
            onChange={(e) => onChange({ ...rule, values: e.target.value.split("\n") })}
          />
        )}
      </div>
      <button type="button" className="mt-5 text-[12px] text-chat-dim hover:text-chat-fg max-[640px]:mt-0" onClick={onRemove}>
        {G.rules.remove} <span className="sr-only">{rule.id}</span>
      </button>
    </div>
  );
}

export function GuardForm({ value, mode, allowOff, packs, onChange }: Props) {
  const name = useId();
  const set = (patch: Partial<GuardSettings>) => onChange({ ...value, ...patch }, mode);
  const setMode = (m: FormMode) => onChange(m === "off" ? value : { ...value, mode: m }, m);
  const togglePack = (id: string) => {
    const on = value.packs.includes(id);
    const order = packs?.packs.map((p) => p.id) ?? [];
    const next = on ? value.packs.filter((p) => p !== id) : [...value.packs, id];
    set({ packs: next.sort((a, b) => order.indexOf(a) - order.indexOf(b)) });
  };
  const setRule = (i: number, r: CustomRule) => set({ rules: value.rules.map((x, j) => (j === i ? r : x)) });
  const setOverride = (id: string, action: string) => {
    const next = { ...value.overrides };
    if (action) next[id] = action;
    else delete next[id];
    set({ overrides: next });
  };

  return (
    <div className="space-y-4">
      <fieldset>
        <legend className={LEGEND}>{G.mode.legend}</legend>
        <div className="flex flex-wrap gap-2">
          {allowOff && <Radio name={name} label={G.mode.off} checked={mode === "off"} onChange={() => setMode("off")} />}
          <Radio name={name} label={G.mode.sanitize} checked={mode === "sanitize"} onChange={() => setMode("sanitize")} />
          <Radio name={name} label={G.mode.block} checked={mode === "block"} onChange={() => setMode("block")} />
        </div>
        <p className={cn(HINT, "mt-1.5")}>{G.mode.help}</p>
      </fieldset>

      {mode !== "off" && (
        <>
          <fieldset>
            <legend className={LEGEND}>
              {G.packs.legend} <span className={cn(HINT, "font-normal")}>· {G.packs.hint}</span>
            </legend>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 max-[640px]:grid-cols-1">
              {(packs?.packs ?? []).map((p) => {
                const on = value.packs.includes(p.id);
                return (
                  <div key={p.id} className="text-[13px]">
                    <div className="flex flex-wrap items-center gap-2">
                      <label className="flex cursor-pointer items-start gap-2">
                        <input type="checkbox" className="mt-1" checked={on} onChange={() => togglePack(p.id)} />
                        <span>{p.label}</span>
                      </label>
                      {on && (
                        <select
                          aria-label={G.packs.actionLabel(p.label)}
                          className={cn(FIELD, "py-0.5 text-[12px]")}
                          value={value.overrides[p.id] ?? ""}
                          onChange={(e) => setOverride(p.id, e.target.value)}
                        >
                          {G.packs.actions.map((a) => (
                            <option key={a.value} value={a.value}>
                              {a.label}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                    <details className="ml-6">
                      <summary className={cn(HINT, "cursor-pointer")}>{G.packs.rules(p.rules.length)}</summary>
                      <p className={HINT}>{p.rules.map((r) => r.label).join(" · ")}</p>
                    </details>
                  </div>
                );
              })}
            </div>
          </fieldset>

          <fieldset>
            <legend className={LEGEND}>
              {G.rules.legend} <span className={cn(HINT, "font-normal")}>· {G.rules.hint}</span>
            </legend>
            <div className="space-y-2">
              {value.rules.map((r, i) => (
                <RuleRow key={i} rule={r} onChange={(x) => setRule(i, x)} onRemove={() => set({ rules: value.rules.filter((_, j) => j !== i) })} />
              ))}
            </div>
            <div className="mt-2 flex gap-2">
              <button type="button" className="text-[13px] text-chat-accent hover:underline" onClick={() => set({ rules: [...value.rules, { id: "", type: "regex", regex: "" }] })}>
                {G.rules.addPattern}
              </button>
              <button type="button" className="text-[13px] text-chat-accent hover:underline" onClick={() => set({ rules: [...value.rules, { id: "", type: "fixed", values: [] }] })}>
                {G.rules.addFixed}
              </button>
            </div>
          </fieldset>

          <div className="space-y-2">
            <Choice label={G.restore.label} value={value.restore} options={G.restore.options} onChange={(v) => set({ restore: v })} />
            <Choice label={G.trailer.label} value={value.trailer} options={G.trailer.options} onChange={(v) => set({ trailer: v })} />
            <Choice label={G.fail.label} value={value.fail} options={G.fail.options} onChange={(v) => set({ fail: v })} />
          </div>
        </>
      )}
    </div>
  );
}
