"use client";

import { useEffect, useId, useState } from "react";
import type { FieldHint } from "@/lib/settings-hints";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, type SelectOption } from "@/components/ui/select";
import { GpuChecklist, type GpuInfo } from "@/components/gpu/gpu-checklist";

// ---------------------------------------------------------------------------
// SettingField — uniform render slot for the heterogeneous PATCH payload.
// ---------------------------------------------------------------------------
//
// The PodWarden-style "label + hint + restart badge + input" is the constant.
// What varies is the *kind* of value: the 11 patchable model fields are a
// mix of strings, numbers, booleans, comma-separated int lists, freeform
// string lists, and key/value maps. Rather than push every per-kind quirk
// into the page (which would either stringify everything at the boundary —
// fragile — or repeat the same chrome 11 times — noisy), this component
// fans out internally on a `kind` discriminator and keeps the page's draft
// dict typed.
//
// Pattern note: the union on `Props` ties `kind` to the matching
// value/onChange types so the page can't accidentally pass a number into a
// "boolean" field. The exhaustive switch at render time would be a compile
// error if a kind were added without a case.

type RestartBadgeKind = "model-reload" | "warden-restart";

type CommonProps = {
  field: FieldHint;
  /** Disables the inner control without hiding it. Used for the "loaded"
   *  status guard at the page level — we render the form so the operator
   *  can see what they would change, but every input refuses input until
   *  they go unload the model. */
  disabled?: boolean;
  /** Optional content rendered in the label row, right of the restart badge.
   *  Generic (any ReactNode); historically used to host benchmark-derived
   *  apply chips, but those were removed in epic/overhaul S1. The slot is
   *  retained for the upcoming S4 "Suggest values" affordance. */
  rightSlot?: React.ReactNode;

  // ---- Opt-in (model settings redesign). All default off, so the global
  // ---- settings tabs render exactly as before.

  /** "token": the model settings redesign anatomy — label → control → hint,
   *  token typography (chat-* / vw-*, no slate), `field.short` as the hint
   *  with a "More" toggle beside it that reveals the long text (nothing is
   *  deleted), and no per-field "requires model-reload" badge (the page
   *  states the reload rule once, on its band heading). Default "legacy":
   *  the global settings tabs render exactly as before. */
  chrome?: "legacy" | "token";
  /** The draft differs from the saved value: an amber dot beside the label
   *  (aria-hidden) plus an sr-only "(unsaved)" in the hint — never inside
   *  the `<label>`, so the label's accessible name does not change. */
  dirty?: boolean;
  /** Fixed id for the control (otherwise generated). Lets a page focus the
   *  input from elsewhere, e.g. the save bar's "Fix" focusing `#dp`. */
  id?: string;
  /** Extra element ids prepended to the control's `aria-describedby`, e.g. a
   *  readout rendered through `afterControl`. */
  describedBy?: string;
  /** Rendered between the control and the hint (token chrome only). */
  afterControl?: React.ReactNode;
};

type TextProps = CommonProps & {
  kind: "text";
  value: string;
  onChange: (v: string) => void;
};

type NumberProps = CommonProps & {
  kind: "number";
  /** null distinguishes "cleared / unset" from "0". `max_model_len` accepts
   *  null to mean "let vLLM pick", so we need to round-trip emptiness as a
   *  first-class value, not coerce-to-zero. */
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  /** Shown while the box is empty (null). */
  placeholder?: string;
};

type BooleanProps = CommonProps & {
  kind: "boolean";
  value: boolean;
  onChange: (v: boolean) => void;
  /** "switch": the same native checkbox with `role="switch"` and switch
   *  styling. Its visible label is the field label itself (no separate
   *  label row, no "enabled/disabled" word); `aria-checked` carries state. */
  variant?: "switch";
};

// Three-valued flag: yes / no / "nobody has said". The third state is not a
// missing value — it is the instruction "work it out yourself", which is what
// the model capability columns mean by NULL (the backend then sniffs the
// on-disk HF config; see app/chat2/catalog.py). A checkbox cannot express it,
// so this renders as a select and emits `null` for Auto.
type TristateProps = CommonProps & {
  kind: "tristate";
  value: boolean | null;
  onChange: (v: boolean | null) => void;
};

// Comma-separated integer list, e.g. gpu_indices. The component owns the
// text representation in local state (well, derived from the value prop)
// and only emits a typed number[] when the parse succeeds; intermediate
// "0," states stay text-only so we don't fight the user mid-typing.
type IntListProps = CommonProps & {
  kind: "int-list";
  value: number[];
  onChange: (v: number[]) => void;
};

// Newline-separated string list, e.g. extra_args. Each non-empty line is
// one entry. Trailing empty lines are dropped on emit so save isn't dirty
// just because the user pressed Enter at the end.
type StringListProps = CommonProps & {
  kind: "string-list";
  value: string[];
  onChange: (v: string[]) => void;
};

// "KEY=value" per line. Same shape as docker-compose env: lines without `=`
// are dropped on emit; whitespace around `=` is trimmed. This matches the
// allowlist filter in app.runtime.env_builder which expects clean KEY=v
// entries.
type KvMapProps = CommonProps & {
  kind: "kv-map";
  value: Record<string, string>;
  onChange: (v: Record<string, string>) => void;
};

type SelectProps = CommonProps & {
  kind: "select";
  value: string | null;
  onChange: (v: string | null) => void;
  options: string[];
  /** A blank option (label "default") prepended to `options` so the user
   *  can explicitly clear the field back to null without typing. The
   *  backend treats `dtype: null` the same as the dtype column being
   *  cleared, which maps to "let vLLM autodetect". */
  allowNull?: boolean;
};

// GPU-index selection backed by the shared GpuChecklist. Unlike int-list
// (a freeform comma-separated text box), this kind renders one checkbox per
// present GPU plus removable ghost rows for configured-but-absent indices,
// and always emits a sorted number[]. The live inventory is passed in via
// `gpus` (the page fetches GET /api/system/gpus); this component stays
// presentational like every other kind.
type GpuSetProps = CommonProps & {
  kind: "gpu-set";
  value: number[];
  onChange: (v: number[]) => void;
  gpus: GpuInfo[];
  /** The model being edited, so its own engine is not reported as competing
   *  with itself for a card it is already loaded on. See GpuChecklist. */
  excludeModelId?: string;
};

type Props =
  | TextProps
  | NumberProps
  | BooleanProps
  | TristateProps
  | IntListProps
  | StringListProps
  | KvMapProps
  | SelectProps
  | GpuSetProps;

function isRestartBadge(r: string): r is RestartBadgeKind {
  return r === "model-reload" || r === "warden-restart";
}

function parseIntList(raw: string): number[] {
  // Permissive parse: split on comma, ignore blanks, drop entries that
  // aren't non-negative integers. The page enforces a stricter "must be a
  // valid integer list before save" check via dirty-tracking — i.e. an
  // unparseable entry gets silently dropped from the emitted value, which
  // tells the page "this draft equals the snapshot" and Save stays clean.
  // That's fine for a stop-the-bleed first pass; a future revision could
  // surface inline validation here.
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .filter((s) => /^\d+$/.test(s))
    .map((s) => Number.parseInt(s, 10));
}

function parseStringList(raw: string): string[] {
  return raw.split(/\r?\n/).map((s) => s.trimEnd()).filter((s) => s.length > 0);
}

function parseKvMap(raw: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const k = trimmed.slice(0, eq).trim();
    const v = trimmed.slice(eq + 1);
    if (!k) continue;
    out[k] = v;
  }
  return out;
}

function kvMapToText(map: Record<string, string>): string {
  return Object.entries(map)
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");
}

// ---------------------------------------------------------------------------
// Freeform-typing sub-components.
// ---------------------------------------------------------------------------
//
// int-list / string-list / kv-map all parse a freeform text box into a typed
// value. The original branches derived the *displayed* value from the parsed
// result on every keystroke (`value={props.value.join(",")}`), which wiped
// trailing commas/newlines — typing "0," parses to [0] whose join is "0", so
// you could never start a second value. These three components own their own
// local text instead: they emit the parsed typed value upward (the contract
// consumers depend on) while keeping the raw text exactly as typed. They
// re-adopt an external `value` (model switch, reset button) only when its
// canonical parsed form diverges from what the local text already parses to,
// so in-progress typing is never clobbered.

const freeformTextareaClass =
  "flex min-h-[5rem] w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 font-mono text-sm text-slate-100 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:cursor-not-allowed disabled:opacity-50";

function IntListInput(props: {
  value: number[];
  onChange: (v: number[]) => void;
  disabled: boolean;
  inputId: string;
  hintId: string;
}) {
  const { value, onChange, disabled, inputId, hintId } = props;
  const [text, setText] = useState(() => value.join(","));

  useEffect(() => {
    if (value.join(",") !== parseIntList(text).join(",")) {
      setText(value.join(","));
    }
    // Resync from the prop only; `text` is intentionally excluded so typing
    // doesn't retrigger the effect against its own emitted value.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <Input
      id={inputId}
      aria-describedby={hintId}
      value={text}
      disabled={disabled}
      inputMode="numeric"
      placeholder="0,1"
      onChange={(e) => {
        const raw = e.target.value;
        setText(raw);
        onChange(parseIntList(raw));
      }}
    />
  );
}

function StringListTextarea(props: {
  value: string[];
  onChange: (v: string[]) => void;
  disabled: boolean;
  inputId: string;
  hintId: string;
}) {
  const { value, onChange, disabled, inputId, hintId } = props;
  const [text, setText] = useState(() => value.join("\n"));

  useEffect(() => {
    // Canonical form = parsed entries joined; "\n" is the serializer the old
    // branch used and args can't contain newlines, so it's a safe comparator.
    if (parseStringList(text).join("\n") !== value.join("\n")) {
      setText(value.join("\n"));
    }
    // same intentional exclusion of `text` as IntListInput — resync on prop change only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <textarea
      id={inputId}
      aria-describedby={hintId}
      className={freeformTextareaClass}
      value={text}
      disabled={disabled}
      rows={Math.max(3, value.length + 1)}
      placeholder="--worker-use-ray"
      onChange={(e) => {
        const raw = e.target.value;
        setText(raw);
        onChange(parseStringList(raw));
      }}
    />
  );
}

// Order-insensitive canonical serialization of a kv-map, used ONLY for the
// resync guard in KvMapTextarea: sort keys so {A:"1",B:"2"} and {B:"2",A:"1"}
// compare equal. Not used for display — display follows the prop's order.
const canonicalKv = (r: Record<string, string>) =>
  Object.entries(r)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");

function KvMapTextarea(props: {
  value: Record<string, string>;
  onChange: (v: Record<string, string>) => void;
  disabled: boolean;
  inputId: string;
  hintId: string;
}) {
  const { value, onChange, disabled, inputId, hintId } = props;
  const [text, setText] = useState(() => kvMapToText(value));

  useEffect(() => {
    // Compare logical content order-insensitively: a save→refresh can echo
    // the same map in a different key order than the user typed, and a plain
    // kvMapToText comparison would then fire and snap the cursor to the
    // server's order. canonicalKv sorts keys for the guard only; the
    // displayed text below still follows the prop's natural order.
    if (canonicalKv(parseKvMap(text)) !== canonicalKv(value)) {
      setText(kvMapToText(value));
    }
    // same intentional exclusion of `text` as IntListInput — resync on prop change only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <textarea
      id={inputId}
      aria-describedby={hintId}
      className={freeformTextareaClass}
      value={text}
      disabled={disabled}
      rows={Math.max(3, Object.keys(value).length + 1)}
      placeholder="HF_HUB_OFFLINE=1"
      onChange={(e) => {
        const raw = e.target.value;
        setText(raw);
        onChange(parseKvMap(raw));
      }}
    />
  );
}

const DIRTY_DOT_CLASS = "inline-block h-1.5 w-1.5 flex-none rounded-full bg-vw-amber-fg";

function DirtyDot() {
  return <span aria-hidden="true" data-testid="setting-field-dirty-dot" className={DIRTY_DOT_CLASS} />;
}

export function SettingField(props: Props) {
  const {
    field,
    disabled = false,
    rightSlot,
    chrome = "legacy",
    dirty = false,
    id: fixedId,
    describedBy,
    afterControl,
  } = props;
  // useId is stable across re-renders so the input <-> hint <-> label
  // wiring survives the unavoidable re-renders from controlled inputs.
  const generatedId = useId();
  const inputId = fixedId ?? generatedId;
  const hintId = useId();
  // What the control's aria-describedby carries: any caller-supplied ids,
  // then the hint. The hint <p> itself keeps `hintId`.
  const describedIds = describedBy ? `${describedBy} ${hintId}` : hintId;
  const longHintId = useId();
  const [moreOpen, setMoreOpen] = useState(false);

  // Restart badges: filter out "none" up front so the JSX below doesn't
  // need to repeat the check, and TypeScript-narrow the kind for the
  // <Badge> content.
  const token = chrome === "token";
  const restartBadge: RestartBadgeKind | null =
    !token && isRestartBadge(field.restart) ? field.restart : null;

  const isSwitch = props.kind === "boolean" && props.variant === "switch";
  const tokenChrome = token || isSwitch;

  // Short hint only when there is a distinct one; otherwise the long hint
  // is the hint and there is nothing to reveal.
  const shortHint = token && field.short && field.short !== field.hint ? field.short : null;

  const unsavedSr = dirty ? (
    <>
      {" "}
      <span className="sr-only">(unsaved)</span>
    </>
  ) : null;

  // The described-by element holds the hint text only. The More toggle and
  // the long text sit beside it, so a screen reader does not end every
  // description with "More".
  const hint = tokenChrome ? (
    <div className="m-0 text-[12.5px] leading-[1.45] text-chat-muted">
      <span id={hintId} className="text-chat-muted">
        {shortHint ?? field.hint}
        {unsavedSr}
      </span>
      {shortHint && (
        <>
          {" "}
          <button
            type="button"
            aria-expanded={moreOpen}
            aria-controls={longHintId}
            onClick={() => setMoreOpen((v) => !v)}
            className="rounded-sm text-[12.5px] text-chat-fg underline underline-offset-2 focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent"
          >
            {moreOpen ? "Less" : "More"}
          </button>
          <span id={longHintId} hidden={!moreOpen} className="mt-1 block [&[hidden]]:hidden">
            {field.hint}
          </span>
        </>
      )}
    </div>
  ) : (
    <p id={hintId} className="text-xs text-slate-500">
      {field.hint}
      {unsavedSr}
    </p>
  );

  const badgeAndSlot =
    restartBadge || rightSlot ? (
      <div className="flex items-center gap-2">
        {restartBadge && (
          <Badge variant="info" aria-label={`requires ${restartBadge}`}>
            requires {restartBadge}
          </Badge>
        )}
        {rightSlot}
      </div>
    ) : null;

  // gpu-set and tristate delegate to controls that own their own ids
  // (GpuChecklist, Select) and never consume inputId, so a
  // `htmlFor={inputId}` there would dangle.
  const labelOwnsNoControl = props.kind === "gpu-set" || props.kind === "tristate";

  if (tokenChrome) {
    const labelClass = "text-[13.5px] font-medium text-chat-fg";
    return (
      <div className="flex min-w-0 flex-col gap-1.5" data-dirty={dirty ? "true" : undefined}>
        {isSwitch ? (
          badgeAndSlot && <div className="flex items-center justify-end gap-2">{badgeAndSlot}</div>
        ) : (
          <div className="flex items-center gap-1.5">
            {labelOwnsNoControl ? (
              <span className={labelClass}>{field.label}</span>
            ) : (
              <label htmlFor={inputId} className={labelClass}>
                {field.label}
              </label>
            )}
            {dirty && <DirtyDot />}
            {badgeAndSlot && <div className="ml-auto">{badgeAndSlot}</div>}
          </div>
        )}
        {renderControl(props, inputId, describedIds, disabled)}
        {afterControl}
        {hint}
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        {labelOwnsNoControl ? (
          <span className="font-medium text-slate-200">{field.label}</span>
        ) : (
          <label htmlFor={inputId} className="font-medium text-slate-200">
            {field.label}
          </label>
        )}
        {dirty && <DirtyDot />}
        <div className="flex items-center gap-2">
          {restartBadge && (
            <Badge variant="info" aria-label={`requires ${restartBadge}`}>
              requires {restartBadge}
            </Badge>
          )}
          {rightSlot}
        </div>
      </div>
      {hint}
      {renderControl(props, inputId, describedIds, disabled)}
    </div>
  );
}

function renderControl(
  props: Props,
  inputId: string,
  hintId: string,
  disabled: boolean,
): React.ReactNode {
  switch (props.kind) {
    case "text":
      return (
        <Input
          id={inputId}
          aria-describedby={hintId}
          value={props.value}
          disabled={disabled}
          onChange={(e) => props.onChange(e.target.value)}
        />
      );

    case "number": {
      // Show empty string for null so the input renders blank instead of
      // "0". On change, an empty input emits null (clears the field);
      // anything else parses as a Number. NaN guards against typing
      // garbage into a number input on browsers that allow it.
      const display = props.value === null ? "" : String(props.value);
      return (
        <Input
          id={inputId}
          aria-describedby={hintId}
          type="number"
          value={display}
          disabled={disabled}
          min={props.min}
          max={props.max}
          step={props.step}
          placeholder={props.placeholder}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              props.onChange(null);
              return;
            }
            const n = Number(raw);
            if (Number.isFinite(n)) props.onChange(n);
          }}
        />
      );
    }

    case "boolean":
      if (props.variant === "switch") {
        // Same native checkbox (keyboard, form semantics, getByLabelText all
        // intact), restyled as a switch. The knob is a sibling span driven by
        // `peer-checked`, not an ::after on the input, which Firefox ignores.
        return (
          <label
            htmlFor={inputId}
            className="inline-flex cursor-pointer items-center gap-2.5 text-[13.5px]"
          >
            <span className="relative inline-flex h-5 w-9 flex-none">
              <input
                id={inputId}
                type="checkbox"
                role="switch"
                aria-checked={props.value}
                aria-describedby={hintId}
                className="peer m-0 h-5 w-9 cursor-pointer appearance-none rounded-full border border-vw-rule-soft bg-chat-surface-2 transition-colors checked:border-vw-live/70 checked:bg-vw-ok-bg focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transition-none"
                checked={props.value}
                disabled={disabled}
                onChange={(e) => props.onChange(e.target.checked)}
              />
              <span
                aria-hidden="true"
                className="pointer-events-none absolute left-[3px] top-[3px] h-3.5 w-3.5 rounded-full bg-chat-muted transition-transform peer-checked:translate-x-4 peer-checked:bg-vw-live peer-disabled:opacity-50 motion-reduce:transition-none"
              />
            </span>
            <span className="font-medium text-chat-fg">{props.field.label}</span>
            {props.dirty && <DirtyDot />}
          </label>
        );
      }
      return (
        <label
          htmlFor={inputId}
          className="flex cursor-pointer items-center gap-2 text-sm text-slate-300"
        >
          <input
            id={inputId}
            type="checkbox"
            aria-describedby={hintId}
            className="h-4 w-4 cursor-pointer rounded border-slate-600 bg-slate-900 text-emerald-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
            checked={props.value}
            disabled={disabled}
            onChange={(e) => props.onChange(e.target.checked)}
          />
          <span>{props.value ? "enabled" : "disabled"}</span>
        </label>
      );

    case "tristate": {
      // The Select primitive models a single string value, so the three
      // states travel as "auto" | "yes" | "no" and are mapped back at the
      // boundary. "auto" MUST come back as null, not as an omitted key — the
      // backend reads null as "reset to auto-detection" and a missing key as
      // "leave whatever is there alone".
      const TRISTATE_OPTIONS: SelectOption<string>[] = [
        { value: "auto", label: "Auto" },
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" },
      ];
      return (
        <Select
          options={TRISTATE_OPTIONS}
          value={props.value === null ? "auto" : props.value ? "yes" : "no"}
          onChange={(v) => props.onChange(v === "auto" ? null : v === "yes")}
          disabled={disabled}
          ariaLabel={props.field.label}
        />
      );
    }

    case "int-list":
      return (
        <IntListInput
          value={props.value}
          onChange={props.onChange}
          disabled={disabled}
          inputId={inputId}
          hintId={hintId}
        />
      );

    case "string-list":
      return (
        <StringListTextarea
          value={props.value}
          onChange={props.onChange}
          disabled={disabled}
          inputId={inputId}
          hintId={hintId}
        />
      );

    case "kv-map":
      return (
        <KvMapTextarea
          value={props.value}
          onChange={props.onChange}
          disabled={disabled}
          inputId={inputId}
          hintId={hintId}
        />
      );

    case "select": {
      // The Select primitive only models a single string value; null is
      // represented by a synthetic "" option that maps back to null on
      // change. We never emit "" upward.
      const options: SelectOption<string>[] = [
        ...(props.allowNull ? [{ value: "", label: "default" }] : []),
        ...props.options.map((o) => ({ value: o, label: o })),
      ];
      return (
        <Select
          options={options}
          value={props.value ?? ""}
          onChange={(v) => props.onChange(v === "" ? null : v)}
          disabled={disabled}
          ariaLabel={props.field.label}
        />
      );
    }

    case "gpu-set":
      // GpuChecklist owns its own per-GPU checkbox ids and the ghost-row /
      // alert chrome; it doesn't take inputId/hintId, so we just hand it the
      // selection state and the live inventory.
      return (
        <GpuChecklist
          gpus={props.gpus}
          selected={props.value}
          onChange={props.onChange}
          disabled={disabled}
          describedById={hintId}
          excludeModelId={props.excludeModelId}
        />
      );
  }
}
