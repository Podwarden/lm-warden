"use client";

/**
 * Create/edit modal for a model alias. Structure follows add-model-modal.tsx:
 * a Modal hosting a form with a Save button.
 *
 * An alias is a name clients call (matched BEFORE served model names), an
 * ordered target list (loaded or not), and optional settings: guard, thinking
 * policy and request-parameter overrides per the D4c table. The server
 * validates everything on save; the 422 message is rendered next to the
 * field it names.
 */

import { useEffect, useState } from "react";
import useSWR from "swr";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { ModelRow } from "@/components/models/model-card";
import {
  AliasApiError,
  createAlias,
  patchAlias,
  type AliasBody,
  type AliasOut,
} from "@/lib/aliases";
import { cn } from "@/lib/utils";

// ---- the D4c parameter table ──────────────────────────────────────────────

type ParamKind = "int" | "float" | "effort" | "json";

interface ParamSpec {
  /** The parameter key sent on the wire (`chat_template_kwargs.<key>` for the json kind). */
  key: string;
  label: string;
  kind: ParamKind;
  /** Numeric params may use mode "cap"; effort/json may not. */
  numeric: boolean;
}

const PARAM_SPECS: ParamSpec[] = [
  { key: "max_tokens", label: "Max tokens", kind: "int", numeric: true },
  { key: "temperature", label: "Temperature", kind: "float", numeric: true },
  { key: "top_p", label: "Top P", kind: "float", numeric: true },
  { key: "top_k", label: "Top K", kind: "int", numeric: true },
  { key: "min_p", label: "Min P", kind: "float", numeric: true },
  { key: "presence_penalty", label: "Presence penalty", kind: "float", numeric: true },
  { key: "frequency_penalty", label: "Frequency penalty", kind: "float", numeric: true },
  { key: "repetition_penalty", label: "Repetition penalty", kind: "float", numeric: true },
  { key: "seed", label: "Seed", kind: "int", numeric: true },
  { key: "reasoning_effort", label: "Reasoning effort", kind: "effort", numeric: false },
  { key: "chat_template_kwargs", label: "chat_template_kwargs.<key>", kind: "json", numeric: false },
];

const REASONING_EFFORTS = ["low", "medium", "high"];

const GUARD_OPTIONS = [
  { value: "inherit", label: "Inherit" },
  { value: "sanitize", label: "Sanitize" },
  { value: "block", label: "Block" },
] as const;

const THINKING_OPTIONS = [
  { value: "inherit", label: "Inherit" },
  { value: "client", label: "Client decides" },
  { value: "on", label: "On" },
  { value: "off", label: "Off" },
] as const;

type ParamMode = "default" | "force" | "cap";
type EditorMode = "create" | "edit";

interface ParamRowState {
  spec: ParamSpec;
  /** chat_template_kwargs only: the <key> suffix. */
  suffix: string;
  raw: string;
  mode: ParamMode;
}

const DEFAULT_ROW: ParamRowState = { spec: PARAM_SPECS[0], suffix: "", raw: "", mode: "default" };
const freshDefaultRow = (): ParamRowState => ({ ...DEFAULT_ROW });

function rowKey(row: ParamRowState): string {
  return row.spec.kind === "json" ? `chat_template_kwargs.${row.suffix.trim()}` : row.spec.key;
}

function rowValue(row: ParamRowState): unknown {
  if (row.spec.kind === "effort") return row.raw;
  const raw = row.raw.trim();
  if (raw === "") return null;
  if (row.spec.kind === "int") {
    const n = Number(raw);
    return Number.isNaN(n) ? null : Math.trunc(n);
  }
  if (row.spec.kind === "float") {
    const n = Number(raw);
    return Number.isNaN(n) ? null : n;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

function paramsToRows(params: AliasOut["params"]): ParamRowState[] {
  const rows: ParamRowState[] = [];
  for (const [key, p] of Object.entries(params ?? {})) {
    const mode: ParamMode = p?.mode === "cap" || p?.mode === "force" ? p.mode : "default";
    if (key.startsWith("chat_template_kwargs.")) {
      const spec = PARAM_SPECS.find((s) => s.kind === "json")!;
      const raw = p?.value === null ? "null" : String(p?.value ?? "");
      rows.push({ spec, suffix: key.slice("chat_template_kwargs.".length), raw, mode });
      continue;
    }
    const spec = PARAM_SPECS.find((s) => s.key === key);
    if (!spec) continue;
    rows.push({ spec, suffix: "", raw: String(p?.value ?? ""), mode });
  }
  return rows;
}

/** The sub-fields a 422 can name; the editor renders the message next to it. */
interface FieldError {
  field: string | null;
  message: string;
}

function extractError(body: unknown): FieldError | null {
  const detail = (body as { detail?: unknown } | null)?.detail;
  const msgs: string[] = [];
  let loc = "";
  if (typeof detail === "string") {
    msgs.push(detail);
  } else if (Array.isArray(detail)) {
    for (const e of detail) {
      if (e && typeof e === "object") {
        const msg = (e as { msg?: unknown }).msg;
        if (typeof msg === "string") msgs.push(msg);
        const locArr = (e as { loc?: unknown }).loc;
        if (Array.isArray(locArr)) {
          const leaf = locArr
            .filter((x) => typeof x === "string" && x !== "body")
            .slice(-1)[0];
          if (typeof leaf === "string") loc = leaf;
        }
      }
    }
  }
  if (msgs.length === 0) return null;
  const message = msgs.join("; ");
  if (message.startsWith("targets:") || loc === "targets") {
    return { field: "targets", message };
  }
  for (const spec of PARAM_SPECS) {
    const needle = spec.kind === "json" ? "'chat_template_kwargs." : `'${spec.key}'`;
    if (message.includes(needle)) return { field: `param:${spec.key}`, message };
  }
  if (loc === "name" || loc === "guard" || loc === "thinking_policy") {
    return { field: loc, message };
  }
  if (loc === "params") return { field: "params", message };
  return { field: null, message };
}

function dotClassFor(status: string | null | undefined): string {
  switch (status) {
    case "loaded":
    case "pulled":
      return "bg-emerald-400";
    case "loading":
    case "pulling":
    case "unloading":
      return "bg-amber-400";
    case "failed":
      return "bg-red-400";
    default:
      return "bg-slate-500";
  }
}

export function AliasEditor({
  open,
  onClose,
  mode,
  alias,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  mode: EditorMode;
  alias?: AliasOut | null;
  onSaved: () => void;
}) {
  const { data: modelsData } = useSWR<{ models: ModelRow[] }>(
    open ? "/api/models" : null,
    authFetchJSON,
  );
  const models = modelsData?.models ?? [];

  const [name, setName] = useState("");
  const [targetIds, setTargetIds] = useState<string[]>([]);
  const [guard, setGuard] = useState("inherit");
  const [thinkingPolicy, setThinkingPolicy] = useState("inherit");
  const [rows, setRows] = useState<ParamRowState[]>([freshDefaultRow()]);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<FieldError | null>(null);

  // Seed the form every time the modal opens.
  useEffect(() => {
    if (!open) return;
    const a = alias ?? null;
    setName(a?.name ?? "");
    setTargetIds((a?.targets ?? []).map((t) => t.id));
    setGuard(a?.guard ?? "inherit");
    setThinkingPolicy(a?.thinking_policy ?? "inherit");
    const seeded = a ? paramsToRows(a.params) : [];
    setRows(seeded.length > 0 ? seeded : [freshDefaultRow()]);
    setErrors(null);
  }, [open, alias]);

  const nameTaken =
    name.trim() !== "" && models.some((m) => m.served_model_name === name.trim());

  function moveTarget(index: number, delta: -1 | 1) {
    const next = [...targetIds];
    const j = index + delta;
    if (j < 0 || j >= next.length) return;
    [next[index], next[j]] = [next[j], next[index]];
    setTargetIds(next);
  }

  function removeTarget(id: string) {
    setTargetIds(targetIds.filter((t) => t !== id));
  }

  function addTarget(id: string) {
    if (!id || targetIds.includes(id)) return;
    setTargetIds([...targetIds, id]);
  }

  function addRow() {
    setRows([...rows, freshDefaultRow()]);
  }

  function updateRow(index: number, patch: Partial<ParamRowState>) {
    setRows(rows.map((r, i) => (i === index ? { ...r, ...patch } : r)));
  }

  function removeRow(index: number) {
    setRows(rows.filter((_, i) => i !== index));
  }

  function updateParamKind(index: number, key: string) {
    const spec = PARAM_SPECS.find((s) => s.key === key) ?? PARAM_SPECS[0];
    setRows(
      rows.map((r, i) => {
        if (i !== index) return r;
        return {
          ...r,
          spec,
          mode: spec.numeric ? r.mode : "default",
          raw: spec.kind === "effort" ? "low" : r.raw === "low" || r.raw === "medium" || r.raw === "high" ? "" : r.raw,
        };
      }),
    );
  }

  async function save() {
    if (saving) return;
    setSaving(true);
    setErrors(null);
    try {
      const params: AliasBody["params"] = {};
      for (const row of rows) {
        const isJson = row.spec.kind === "json";
        const raw = row.raw.trim();
        if (raw === "" || (isJson && row.suffix.trim() === "")) continue; // empty row: not sent
        const value = rowValue(row);
        if (value === null) continue; // unparseable value: leave to nothing
        params[rowKey(row)] = { value, mode: row.mode };
      }
      const body: AliasBody = {
        name: name.trim(),
        targets: targetIds,
        guard,
        thinking_policy: thinkingPolicy,
        params,
        enabled: mode === "edit" ? (alias?.enabled ?? true) : true,
      };
      if (mode === "create") {
        await createAlias(body);
      } else {
        await patchAlias(alias!.id, body);
      }
      onSaved();
    } catch (e) {
      if (e instanceof AliasApiError) {
        const ext = extractError(e.body);
        setErrors(ext ?? { field: null, message: e.message });
      } else {
        setErrors({ field: null, message: e instanceof Error ? e.message : String(e) });
      }
    } finally {
      setSaving(false);
    }
  }

  const fieldError = (field: string) =>
    errors && errors.field === field ? errors.message : null;

  const paramError = (specKey: string) =>
    errors?.field === `param:${specKey}` ? errors.message : null;

  const generalError = errors && !errors.field ? errors.message : null;

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={mode === "create" ? "Add alias" : "Edit alias"}
    >
      <div className="space-y-4 text-sm">
        <div className="space-y-1.5">
          <label htmlFor="alias-name" className="block font-medium text-slate-200">
            Name
          </label>
          <Input
            id="alias-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. lmw-default"
            autoComplete="off"
            aria-label="Name"
            data-testid="alias-name-input"
          />
          {nameTaken && (
            <p data-testid="alias-priority-note" className="text-xs text-amber-400">
              This alias takes priority over the model with the same name.
            </p>
          )}
          {fieldError("name") && (
            <p role="alert" className="text-xs text-red-400">{fieldError("name")}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <span className="block font-medium text-slate-200">Targets</span>
          <div className="space-y-1.5">
            {targetIds.map((id, i) => {
              const m = models.find((x) => x.id === id);
              const label = m?.served_model_name ?? id;
              return (
                <div
                  key={id}
                  className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900/60 px-2 py-1.5"
                >
                  <span aria-hidden className={cn("h-2 w-2 rounded-full", dotClassFor(m?.status))} />
                  <span className="min-w-0 flex-1 truncate">{label}</span>
                  <button
                    type="button"
                    aria-label={`Move ${label} up`}
                    onClick={() => moveTarget(i, -1)}
                    disabled={i === 0}
                    className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100 disabled:opacity-30"
                  >
                    <ArrowUp className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    aria-label={`Move ${label} down`}
                    onClick={() => moveTarget(i, 1)}
                    disabled={i === targetIds.length - 1}
                    className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100 disabled:opacity-30"
                  >
                    <ArrowDown className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    aria-label={`Remove ${label}`}
                    onClick={() => removeTarget(id)}
                    className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-red-400"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
          <select
            aria-label="Add a target model"
            value=""
            onChange={(e) => addTarget(e.target.value)}
            className="mt-1 h-9 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            data-testid="add-target-select"
          >
            <option value="">Add a model…</option>
            {models
              .filter((m) => !targetIds.includes(m.id))
              .map((m) => (
                <option key={m.id} value={m.id}>
                  {m.served_model_name}
                </option>
              ))}
          </select>
          {fieldError("targets") && (
            <p role="alert" className="text-xs text-red-400">{fieldError("targets")}</p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label htmlFor="alias-guard" className="block font-medium text-slate-200">
              Sensitive data
            </label>
            <select
              id="alias-guard"
              aria-label="Sensitive data"
              value={guard}
              onChange={(e) => setGuard(e.target.value)}
              className="h-9 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            >
              {GUARD_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            {fieldError("guard") && (
              <p role="alert" className="text-xs text-red-400">{fieldError("guard")}</p>
            )}
          </div>
          <div className="space-y-1.5">
            <label htmlFor="alias-thinking" className="block font-medium text-slate-200">
              Thinking
            </label>
            <select
              id="alias-thinking"
              aria-label="Thinking"
              value={thinkingPolicy}
              onChange={(e) => setThinkingPolicy(e.target.value)}
              className="h-9 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            >
              {THINKING_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            {fieldError("thinking_policy") && (
              <p role="alert" className="text-xs text-red-400">{fieldError("thinking_policy")}</p>
            )}
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="block font-medium text-slate-200">Parameters</span>
            <button
              type="button"
              onClick={addRow}
              data-testid="add-param-button"
              className="inline-flex items-center gap-1 text-xs text-slate-300 hover:text-slate-100"
            >
              <Plus className="h-3.5 w-3.5" /> Add parameter
            </button>
          </div>
          {fieldError("params") && (
            <p role="alert" className="text-xs text-red-400">{fieldError("params")}</p>
          )}
          <div className="space-y-2">
            {rows.map((row, i) => {
              const err = paramError(row.spec.key);
              const modeOptions: { value: ParamMode; label: string }[] = row.spec.numeric
                ? [
                    { value: "default", label: "Default" },
                    { value: "force", label: "Force" },
                    { value: "cap", label: "Cap" },
                  ]
                : [
                    { value: "default", label: "Default" },
                    { value: "force", label: "Force" },
                  ];
              const normalizedMode: ParamMode = modeOptions.some((o) => o.value === row.mode)
                ? row.mode
                : "default";
              return (
                <div
                  key={i}
                  className="space-y-1.5 rounded-md border border-slate-700 bg-slate-900/40 p-2"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      aria-label="Parameter"
                      value={row.spec.key}
                      onChange={(e) => updateParamKind(i, e.target.value)}
                      className="h-9 min-w-40 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                    >
                      {PARAM_SPECS.map((s) => (
                        <option key={s.key} value={s.key}>{s.label}</option>
                      ))}
                    </select>
                    {row.spec.kind === "json" && (
                      <Input
                        aria-label="chat_template_kwargs key"
                        placeholder="key"
                        value={row.suffix}
                        onChange={(e) => updateRow(i, { suffix: e.target.value })}
                        className="h-9 w-36"
                      />
                    )}
                    {row.spec.kind === "effort" ? (
                      <select
                        aria-label="Value"
                        value={row.raw}
                        onChange={(e) => updateRow(i, { raw: e.target.value })}
                        className="h-9 w-32 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                      >
                        {REASONING_EFFORTS.map((level) => (
                          <option key={level} value={level}>{level}</option>
                        ))}
                      </select>
                    ) : (
                      <Input
                        aria-label="Value"
                        type={row.spec.kind === "json" ? "text" : "number"}
                        step={row.spec.kind === "float" ? "0.1" : "1"}
                        placeholder="Value"
                        value={row.raw}
                        onChange={(e) => updateRow(i, { raw: e.target.value })}
                        className="w-36"
                      />
                    )}
                    <select
                      aria-label="Mode"
                      value={normalizedMode}
                      onChange={(e) => updateRow(i, { mode: e.target.value as ParamMode })}
                      className="h-9 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                    >
                      {modeOptions.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      aria-label="Remove parameter"
                      onClick={() => removeRow(i)}
                      className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-red-400"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  {err && (
                    <p role="alert" className="text-xs text-red-400">{err}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {generalError && (
          <p role="alert" className="rounded-md border border-red-700 bg-red-900/30 p-2 text-xs text-red-200">
            {generalError}
          </p>
        )}

        <div className="flex justify-end gap-2 border-t border-slate-800 pt-3">
          <Button variant="ghost" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={save} disabled={saving} data-testid="save-alias-button">
            {saving ? "Saving…" : mode === "create" ? "Add alias" : "Save"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
