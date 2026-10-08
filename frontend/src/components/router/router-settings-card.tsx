"use client";

// "Timeouts, breaker and upstream" (plan §4.6): a collapsed section whose head
// summarises the current values. Fields are grouped (local leg / breaker /
// Anthropic) and validated on blur against the documented ranges. The footer
// always says why Save is or isn't available, and turns into the sticky save
// bar while there are unsaved changes. Only changed keys are PATCHed.
// `enabled` and `passthrough_unmatched` are no longer here (master switch,
// routing map's rest row); their saves refetch settings, which rebases (not
// resets) the draft here.

import { useEffect, useId, useRef, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { cn } from "@/lib/utils";
import type { RouterSettingsOut, RouterSettingsPatch } from "@/lib/router";
import { btn, FOCUS, SEC, SEC_TITLE } from "./styles";

type NumKey =
  | "local_header_timeout_s"
  | "local_nonstream_timeout_s"
  | "breaker_threshold"
  | "breaker_open_s"
  | "max_body_mb";
type FieldKey = NumKey | "upstream_url";

interface FieldDef {
  key: FieldKey;
  label: string;
  hint?: string;
  /** Ranges from docs/operating.md. */
  min?: number;
  max?: number;
  /** "Breaker stays open for" — the save bar's plain-language name. */
  say: string;
  unit?: string;
}

const GROUPS: { title: string; fields: FieldDef[] }[] = [
  {
    title: "Local leg",
    fields: [
      {
        key: "local_header_timeout_s",
        label: "First-byte timeout (s)",
        hint: "Streamed requests, admission queue included. Cold 30k-token prompts can take 14–40 s.",
        min: 1,
        max: 600,
        say: "First-byte timeout",
        unit: "s",
      },
      {
        key: "local_nonstream_timeout_s",
        label: "Non-streaming timeout (s)",
        hint: "Same, for non-stream requests.",
        min: 1,
        max: 3600,
        say: "Non-streaming timeout",
        unit: "s",
      },
    ],
  },
  {
    title: "Breaker (one per local model)",
    fields: [
      { key: "breaker_threshold", label: "Failures before it opens", min: 1, max: 100, say: "Breaker opens after", unit: "failures" },
      {
        key: "breaker_open_s",
        label: "Stays open for (s)",
        hint: "Then one probe request decides whether it closes.",
        min: 1,
        max: 3600,
        say: "Breaker stays open for",
        unit: "s",
      },
    ],
  },
  {
    title: "Anthropic",
    fields: [
      { key: "upstream_url", label: "Upstream URL", hint: "https only (http for localhost stubs).", say: "Upstream URL" },
      { key: "max_body_mb", label: "Max request body (MB)", min: 1, max: 256, say: "Max request body", unit: "MB" },
    ],
  },
];

const FIELDS = GROUPS.flatMap((g) => g.fields);

function hostOf(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}

/** "first byte 60 s · non-stream 120 s · breaker 3 failures → 60 s · body 32 MB · api.anthropic.com" */
export function settingsSummary(s: RouterSettingsOut): string {
  return [
    `first byte ${s.local_header_timeout_s} s`,
    `non-stream ${s.local_nonstream_timeout_s} s`,
    `breaker ${s.breaker_threshold} failures → ${s.breaker_open_s} s`,
    `body ${s.max_body_mb} MB`,
    hostOf(s.upstream_url),
  ].join(" · ");
}

function validate(f: FieldDef, v: string): string | null {
  if (f.key === "upstream_url") {
    return /^https?:\/\/\S+$/.test(v.trim()) ? null : "Enter an http(s) URL.";
  }
  const n = Number(v);
  if (v.trim() === "" || !Number.isInteger(n) || n < (f.min ?? -Infinity) || n > (f.max ?? Infinity)) {
    return `Between ${f.min} and ${f.max}.`;
  }
  return null;
}

interface Props {
  settings: RouterSettingsOut;
  onSaved: () => void;
  className?: string;
}

export function RouterSettingsCard({ settings, onSaved, className }: Props) {
  const [open, setOpen] = useState(false);
  const [patch, setPatch] = useState<Partial<Record<FieldKey, string>>>({});
  const [invalid, setInvalid] = useState<Partial<Record<FieldKey, string>>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputs = useRef<Partial<Record<FieldKey, HTMLInputElement | null>>>({});
  const bodyId = useId();
  const uid = useId();
  const focusAfterError = useRef<FieldKey | null>(null);

  // A fresh server copy rebases the draft: the master switch and the
  // pass-through row PATCH settings too, and must not discard unsaved edits
  // here. Keep each drafted key whose value still differs from the server's;
  // drop the ones the server now holds (our own save, or someone else's).
  const patchRef = useRef(patch);
  useEffect(() => {
    patchRef.current = patch;
  }, [patch]);
  useEffect(() => {
    const kept: Partial<Record<FieldKey, string>> = {};
    for (const [k, v] of Object.entries(patchRef.current) as [FieldKey, string][]) {
      if (v !== String(settings[k])) kept[k] = v;
    }
    patchRef.current = kept;
    setPatch(kept);
    setInvalid((m) => {
      const next: Partial<Record<FieldKey, string>> = {};
      for (const [k, e] of Object.entries(m) as [FieldKey, string][]) if (k in kept) next[k] = e;
      return next;
    });
  }, [settings]);

  useEffect(() => {
    if (error && focusAfterError.current) {
      inputs.current[focusAfterError.current]?.focus();
      focusAfterError.current = null;
    }
  }, [error]);

  const value = (k: FieldKey) => patch[k] ?? String(settings[k]);
  const dirtyKeys = FIELDS.filter((f) => f.key in patch).map((f) => f.key);
  const dirty = dirtyKeys.length;
  const invalidCount = Object.keys(invalid).length;

  function set(f: FieldDef, v: string) {
    setPatch((p) => {
      const next = { ...p };
      if (v === String(settings[f.key])) delete next[f.key];
      else next[f.key] = v;
      return next;
    });
    // Re-validate a field already flagged so the error clears as it is fixed.
    if (f.key in invalid) {
      setInvalid((m) => {
        const next = { ...m };
        const e = validate(f, v);
        if (e) next[f.key] = e;
        else delete next[f.key];
        return next;
      });
    }
  }

  function blur(f: FieldDef) {
    const e = validate(f, value(f.key));
    setInvalid((m) => {
      const next = { ...m };
      if (e) next[f.key] = e;
      else delete next[f.key];
      return next;
    });
  }

  function discard() {
    setPatch({});
    setInvalid({});
    setError(null);
  }

  async function save() {
    // Validate everything once more: a field may not have been blurred.
    const errs: Partial<Record<FieldKey, string>> = {};
    for (const f of FIELDS) {
      const e = f.key in patch ? validate(f, value(f.key)) : null;
      if (e) errs[f.key] = e;
    }
    if (Object.keys(errs).length) {
      setInvalid(errs);
      inputs.current[Object.keys(errs)[0] as FieldKey]?.focus();
      return;
    }
    const body: Record<string, unknown> = {};
    for (const k of dirtyKeys) body[k] = k === "upstream_url" ? patch[k] : Number(patch[k]);
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch("/api/router/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body as RouterSettingsPatch),
      });
      if (!r.ok) {
        const detail = await errorDetail(r, `Failed to save (HTTP ${r.status})`);
        // Focus the field the server named, else the first changed one.
        focusAfterError.current = dirtyKeys.find((k) => detail.includes(k)) ?? dirtyKeys[0] ?? null;
        setError(detail);
        return;
      }
      setPatch({});
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  function changeSummary(): React.ReactNode {
    if (dirty !== 1) return null;
    const f = FIELDS.find((x) => x.key === dirtyKeys[0])!;
    const unit = f.unit ? ` ${f.unit}` : "";
    return (
      <>
        {" · "}
        {f.say} <s className="opacity-75">{String(settings[f.key])}</s> →{" "}
        <b className="font-medium text-vw-amber-fg [overflow-wrap:anywhere]">
          {patch[f.key]}
          {unit}
        </b>
      </>
    );
  }

  return (
    <section
      aria-labelledby={`${uid}-h`}
      data-testid="router-settings-section"
      className={cn(SEC, className)}
    >
      <h2 className={cn(SEC_TITLE, "px-4 py-3.5")}>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen((o) => !o)}
          className={cn(
            "flex w-full min-w-0 items-center gap-2.5 rounded-md bg-transparent p-0 text-left max-[759px]:flex-wrap",
            FOCUS,
          )}
        >
          <svg
            viewBox="0 0 16 16"
            className={cn("h-4 w-4 flex-none text-chat-muted motion-safe:transition-transform", open && "rotate-90")}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <path d="M6 3l5 5-5 5" />
          </svg>
          <span id={`${uid}-h`} className="whitespace-nowrap">
            Timeouts, breaker and upstream
          </span>
          {dirty > 0 && (
            <span className="whitespace-nowrap rounded-full bg-vw-amber-bg/60 px-[9px] py-px text-[12px] font-semibold text-vw-amber-fg">
              {dirty} unsaved
            </span>
          )}
          {!open && (
            <span className="min-w-0 truncate text-[12.5px] font-normal text-chat-muted max-[759px]:basis-full max-[759px]:whitespace-normal max-[759px]:pl-[26px]">
              {settingsSummary(settings)}
            </span>
          )}
        </button>
      </h2>
      {open && (
        <div id={bodyId} data-testid="router-settings">
          <div className="px-4 pb-4">
            {GROUPS.map((g) => (
              <fieldset key={g.title} className="m-0 mt-[18px] min-w-0 border-0 p-0 first:mt-0">
                <legend className="mb-2.5 p-0 text-[12px] font-semibold uppercase tracking-[0.06em] text-chat-muted">
                  {g.title}
                </legend>
                <div className="grid gap-x-5 gap-y-4 [grid-template-columns:repeat(auto-fit,minmax(220px,1fr))]">
                  {g.fields.map((f) => {
                    const id = `${uid}-${f.key}`;
                    const err = invalid[f.key];
                    const isDirty = f.key in patch;
                    const desc = [err ? `${id}-err` : null, f.hint ? `${id}-hint` : null].filter(Boolean).join(" ");
                    return (
                      <div key={f.key} className="flex min-w-0 flex-col gap-1.5">
                        <label htmlFor={id} className="flex items-center gap-1.5 text-[13.5px] font-medium">
                          {f.label}
                          {isDirty && (
                            <>
                              <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-vw-amber-fg" />
                              <span className="sr-only">(unsaved)</span>
                            </>
                          )}
                        </label>
                        <input
                          id={id}
                          ref={(el) => {
                            inputs.current[f.key] = el;
                          }}
                          type={f.key === "upstream_url" ? "url" : "number"}
                          inputMode={f.key === "upstream_url" ? "url" : "numeric"}
                          min={f.min}
                          max={f.max}
                          value={value(f.key)}
                          aria-invalid={err ? true : undefined}
                          aria-describedby={desc || undefined}
                          onChange={(e) => set(f, e.target.value)}
                          onBlur={() => blur(f)}
                          className={cn(
                            "min-h-9 w-full rounded-lg border bg-chat-page/70 px-2.5 py-[7px] text-[14px] text-chat-fg",
                            err ? "border-vw-danger" : "border-vw-rule-soft",
                            FOCUS,
                          )}
                        />
                        {err && (
                          <p id={`${id}-err`} className="m-0 text-[12.5px] font-medium text-vw-danger-fg">
                            {err}
                          </p>
                        )}
                        {f.hint && (
                          <p id={`${id}-hint`} className="m-0 text-[12.5px] text-chat-muted">
                            {f.hint}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </fieldset>
            ))}
          </div>
          <div
            className={cn(
              "z-[5] flex flex-wrap items-center gap-3 rounded-b-xl border-t border-vw-rule-soft/70 px-4 py-2.5",
              dirty > 0 &&
                "sticky bottom-0 rounded-t-xl border border-vw-rule-soft bg-chat-surface shadow-[0_-8px_24px_rgb(0_0_0/0.25)] max-[759px]:rounded-none",
            )}
          >
            <p className="m-0 min-w-0 flex-1 text-[13.5px] text-chat-muted">
              {dirty === 0 ? (
                "No unsaved changes"
              ) : (
                <>
                  <strong className="font-semibold text-chat-fg">
                    {dirty === 1 ? "1 unsaved change" : `${dirty} unsaved changes`}
                  </strong>
                  {changeSummary()}
                  {" · "}
                  {invalidCount > 0
                    ? `Fix ${invalidCount} field${invalidCount === 1 ? "" : "s"} to save`
                    : "applies to the next request"}
                </>
              )}
            </p>
            {error && (
              <p role="alert" className="m-0 basis-full text-[13px] font-medium text-vw-danger-fg max-[759px]:order-first">
                {error}
              </p>
            )}
            <span className="ml-auto flex gap-2 max-[759px]:w-full">
              <button
                type="button"
                onClick={discard}
                disabled={dirty === 0 || busy}
                className={btn("default", `max-[759px]:min-h-10 max-[759px]:flex-1 ${FOCUS}`)}
              >
                Discard
              </button>
              <button
                type="button"
                onClick={save}
                disabled={dirty === 0 || invalidCount > 0 || busy}
                className={btn("primary", `max-[759px]:min-h-10 max-[759px]:flex-1 ${FOCUS}`)}
              >
                {busy ? "Saving…" : "Save"}
              </button>
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
