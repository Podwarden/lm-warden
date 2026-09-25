"use client";

import { useEffect, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";
import { newPasswordProblem } from "@/lib/password-policy";

// ---------------------------------------------------------------------------
// useRuntimeSettings — extracted from the now-deleted runtime-tab.tsx so
// each of the four runtime sub-tabs (General / Networking / Sessions &
// Tokens / Maintenance) gets its own Edit/Save lifecycle without each
// re-implementing the dirty-tracker, secret-sentinel handling, and
// restart-banner classification.
// ---------------------------------------------------------------------------
//
// Per the #154 spec (`docs/superpowers/specs/2026-05-24-settings-redesign-design.md`):
//
//   The hook holds the full `Draft` and `snapshot` internally (one SWR
//   key, one fetch shared across all four tabs via SWR cache dedupe);
//   the `keys` argument scopes which subset participates in
//   `dirty`/`setField`/`save`. This lets per-tab Save buttons PATCH only
//   their own slice without forcing a global draft.
//
// Invariants the hook preserves verbatim from runtime-tab.tsx:
//
//   * Backend KV-table returns every value as a string-or-null. Per-key
//     coercion (parseNumber / parseBool / parseGpuIndices) seeds a typed
//     Draft. Save emits typed values; the backend's coercers accept
//     both JSON-encoded strings and live arrays/bools/numbers.
//   * Secret keys (admin_password, hf_token) come back as the `"***"`
//     sentinel. The UI renders an empty input; an empty draft means
//     "leave the existing password/token alone" — the dirty-tracker
//     classifies a secret key as dirty iff the user typed a non-empty
//     replacement. This guards against the regression caught by the
//     `does NOT include unchanged secret fields in PATCH body` test.
//   * `seededFor.current` re-seeds the draft from the snapshot exactly
//     once per snapshot identity; after a successful PATCH we clear
//     `seededFor.current` so the next revalidate re-seeds the form to
//     the freshly-saved state.
//   * The PATCH body is the *dirty subset* — keys this tab cares about
//     AND have changed. Per-tab Save buttons don't accidentally
//     overwrite fields owned by other tabs even though all four hooks
//     share a Draft.
//
// New for #154:
//
//   * `public_url` — string, restart 'none'. Same shape as `vllm_version`
//     (free-form text). Backend canonicalises (strips trailing slash);
//     the FE round-trips whatever the backend echoes.
//
// ---------------------------------------------------------------------------

// Iteration order matters — RUNTIME_KEYS drives default snapshot/Draft
// construction. Mirroring the union below keeps TypeScript exhaustive
// across the coercion functions.
export type RuntimeKey =
  | "admin_username"
  | "admin_password"
  | "hf_token"
  | "default_gpu_indices"
  | "default_token_expiration_days"
  | "rotation_grace_hours"
  | "session_access_ttl_minutes"
  | "session_refresh_ttl_days"
  | "sse_ticket_ttl_seconds"
  | "vllm_version"
  | "log_retention_lines"
  | "landing_page_enabled"
  | "public_url"
  | "watchdog_enabled"
  | "watchdog_restore_on_boot"
  | "watchdog_interval_s"
  | "watchdog_failure_threshold"
  | "watchdog_max_restarts";

// Sentinel the backend echoes for secret fields when a value exists.
// We never round-trip this to the server (PATCH body strips keys whose
// draft still equals "").
const SECRET_KEYS: RuntimeKey[] = ["admin_password", "hf_token"];

// Backend KV-table response: every value is a string or null. We coerce
// per-key when seeding the draft below.
export type RuntimeResponse = Partial<Record<RuntimeKey, string | null>>;

// Typed draft after coercion. Split per-key kind so the SettingField
// dispatcher gets the right shape without runtime casts.
export interface Draft {
  admin_username: string;
  admin_password: string;
  hf_token: string;
  default_gpu_indices: number[];
  default_token_expiration_days: number | null;
  rotation_grace_hours: number | null;
  session_access_ttl_minutes: number | null;
  session_refresh_ttl_days: number | null;
  sse_ticket_ttl_seconds: number | null;
  vllm_version: string;
  log_retention_lines: number | null;
  landing_page_enabled: boolean;
  public_url: string;
  watchdog_enabled: boolean;
  watchdog_restore_on_boot: boolean;
  watchdog_interval_s: number | null;
  watchdog_failure_threshold: number | null;
  watchdog_max_restarts: number | null;
}

function parseNumber(raw: string | null | undefined): number | null {
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// #155 unified-port: tolerate the same canonical truthy/falsy spellings the
// backend's _bool coercer accepts. Default to true if the row is missing,
// matching the backend's default-on behavior so the toggle reflects the
// "no opt-out applied" state correctly.
function parseBool(raw: string | null | undefined, defaultValue: boolean): boolean {
  if (raw == null || raw === "") return defaultValue;
  const lowered = raw.trim().toLowerCase();
  if (["true", "1", "yes", "on"].includes(lowered)) return true;
  if (["false", "0", "no", "off"].includes(lowered)) return false;
  return defaultValue;
}

function parseGpuIndices(raw: string | null | undefined): number[] {
  // Backend persists as JSON-encoded text (e.g. `"[0,1]"`). Tolerate
  // legacy comma-separated values too in case the column was hand-edited.
  if (raw == null || raw === "") return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed
        .filter((v) => typeof v === "number" && Number.isInteger(v) && v >= 0)
        .map((v) => v as number);
    }
  } catch {
    /* fall through to comma-split */
  }
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => /^\d+$/.test(s))
    .map((s) => Number.parseInt(s, 10));
}

function snapshotToDraft(s: RuntimeResponse): Draft {
  return {
    admin_username: s.admin_username ?? "",
    // Secret fields: render empty so the operator can type a fresh value.
    admin_password: "",
    hf_token: "",
    default_gpu_indices: parseGpuIndices(s.default_gpu_indices),
    default_token_expiration_days: parseNumber(s.default_token_expiration_days),
    rotation_grace_hours: parseNumber(s.rotation_grace_hours),
    session_access_ttl_minutes: parseNumber(s.session_access_ttl_minutes),
    session_refresh_ttl_days: parseNumber(s.session_refresh_ttl_days),
    sse_ticket_ttl_seconds: parseNumber(s.sse_ticket_ttl_seconds),
    vllm_version: s.vllm_version ?? "",
    log_retention_lines: parseNumber(s.log_retention_lines),
    landing_page_enabled: parseBool(s.landing_page_enabled, true),
    public_url: s.public_url ?? "",
    // Absent rows fall back to the backend defaults so the toggles reflect
    // actual behaviour on a warden that has never had these written.
    watchdog_enabled: parseBool(s.watchdog_enabled, true),
    watchdog_restore_on_boot: parseBool(s.watchdog_restore_on_boot, true),
    watchdog_interval_s: parseNumber(s.watchdog_interval_s) ?? 30,
    watchdog_failure_threshold: parseNumber(s.watchdog_failure_threshold) ?? 3,
    watchdog_max_restarts: parseNumber(s.watchdog_max_restarts) ?? 3,
  };
}

// Per-key deep equality — arrays compare elementwise. We deliberately do
// NOT compare secret keys here: the snapshot always carries "" for them
// (we never display the sentinel), so dirty-tracking for secrets is
// "is the draft non-empty?" — handled in dirtyKeys() below.
function eqValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!eqValue(a[i], b[i])) return false;
    return true;
  }
  return false;
}

function dirtyKeysFor(
  draft: Draft,
  snapshot: Draft,
  scope: readonly RuntimeKey[],
): RuntimeKey[] {
  const out: RuntimeKey[] = [];
  for (const k of scope) {
    if (SECRET_KEYS.includes(k)) {
      // Secret keys are dirty iff the user typed a non-empty replacement.
      // Empty draft means "leave the existing password/token alone".
      const v = draft[k] as string;
      if (v.length > 0) out.push(k);
      continue;
    }
    if (!eqValue(draft[k], snapshot[k])) out.push(k);
  }
  return out;
}

// Compose the PATCH body from the dirty subset. Coerces typed Draft values
// back to whatever the server's coercer expects — numbers stay numbers,
// gpu_indices ships as a number[] (the backend's coercer accepts both
// JSON-encoded strings and live arrays).
function buildPatchBody(
  draft: Draft,
  dirty: RuntimeKey[],
): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const k of dirty) {
    body[k] = draft[k];
  }
  return body;
}

interface PatchResponse {
  ok: boolean;
  requires_restart?: string[];
  requires_restart_kinds?: string[];
}

function restartBannerLines(kinds: string[]): string[] {
  const out: string[] = [];
  if (kinds.includes("model-reload")) {
    out.push(
      "Settings saved — affected models must be unloaded + reloaded to pick this up.",
    );
  }
  if (kinds.includes("warden-restart")) {
    out.push(
      "Settings saved — warden process must be restarted for this to take effect.",
    );
  }
  return out;
}

const SWR_KEY = "/api/settings/runtime";

export interface UseRuntimeSettingsResult {
  /** null while the initial fetch is in flight or has errored. */
  draft: Draft | null;
  /** null while loading; otherwise the last-fetched snapshot, re-coerced. */
  snapshot: Draft | null;
  /** Subset of `scope` whose draft value differs from the snapshot. */
  dirty: RuntimeKey[];
  /** SWR loading flag — true on the first fetch. */
  isLoading: boolean;
  /** SWR error from the GET; PATCH errors land in `saveError`. */
  error: unknown;
  /** Are we in edit mode (Edit clicked, Cancel/Save not yet pressed)? */
  editing: boolean;
  /** Is a PATCH in flight? */
  saving: boolean;
  /** Inline PATCH error (4xx detail or network failure). */
  saveError: string | null;
  /** Lines to render in the per-tab restart banner. */
  restartBanner: string[];
  /** Per-key typed setter — TypeScript narrows on K. */
  setField: <K extends RuntimeKey>(k: K, v: Draft[K]) => void;
  /** Enter edit mode. Clears any stale banner / error. */
  edit(): void;
  /** Exit edit mode and discard local edits. */
  cancel(): void;
  /** Discard local edits but stay in edit mode. */
  reset(): void;
  /** PATCH only the dirty subset within this tab's scope. */
  save(): Promise<void>;
}

/**
 * Subscribes the calling component to `GET /api/settings/runtime` (deduped
 * via SWR across every tab on the page) and provides a per-tab edit
 * lifecycle scoped to the supplied keys.
 *
 * Each tab passes its own `scope` (e.g. `RUNTIME_GENERAL_KEYS`), and
 * `dirty` / `save` operate only on that slice. Tabs that share the
 * underlying SWR cache see the freshly-saved snapshot on the next
 * revalidate, so editing on one tab and clicking another tab shows the
 * updated value without a manual refresh.
 */
export function useRuntimeSettings(
  scope: readonly RuntimeKey[],
): UseRuntimeSettingsResult {
  const { data, error, isLoading, mutate } = useSWR<RuntimeResponse>(
    SWR_KEY,
    authFetchJSON,
    // Same "no polling on edit forms" rationale as the original runtime
    // tab — a background revalidate mid-keystroke would clobber the
    // operator's in-flight edits.
    { revalidateOnFocus: false },
  );
  const { mutate: globalMutate } = useSWRConfig();

  const [draft, setDraft] = useState<Draft | null>(null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [restartBanner, setRestartBanner] = useState<string[]>([]);

  // Seed the draft from the snapshot exactly once per snapshot identity.
  // After a successful PATCH we clear seededFor.current so the next
  // revalidate re-seeds the form to the freshly-saved state.
  const seededFor = useRef<string | null>(null);
  useEffect(() => {
    if (!data) return;
    const sig = JSON.stringify(snapshotToDraft(data));
    if (seededFor.current === sig) return;
    seededFor.current = sig;
    setDraft(snapshotToDraft(data));
  }, [data]);

  const snapshot = data ? snapshotToDraft(data) : null;
  const dirty = draft && snapshot ? dirtyKeysFor(draft, snapshot, scope) : [];

  function setField<K extends RuntimeKey>(k: K, v: Draft[K]) {
    setDraft((d) => (d === null ? d : { ...d, [k]: v }));
  }

  function edit() {
    setEditing(true);
    setRestartBanner([]);
    setSaveError(null);
  }

  function reset() {
    if (!data) return;
    seededFor.current = null;
    setDraft(snapshotToDraft(data));
    setSaveError(null);
  }

  function cancel() {
    reset();
    setEditing(false);
  }

  async function save() {
    if (!data || !draft) return;
    const dirtyNow = dirtyKeysFor(draft, snapshotToDraft(data), scope);
    if (dirtyNow.length === 0) return;
    // A new admin password must meet the server's rule (>= 12 characters,
    // <= 72 bytes); say so before sending anything. The server refuses it
    // too (422, nothing written), this just spares the round trip.
    if (dirtyNow.includes("admin_password")) {
      const problem = newPasswordProblem(draft.admin_password);
      if (problem) {
        setSaveError(problem);
        return;
      }
    }
    setSaving(true);
    setSaveError(null);
    try {
      const body = buildPatchBody(draft, dirtyNow);
      const r = await authFetch(SWR_KEY, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        // Shared with every other surface that reads a FastAPI refusal: a
        // string detail, a 422's [{loc,msg}] list, or a structured refusal
        // carrying `message` (session_only). Hand-rolling it here is how this
        // hook used to render `HTTP 403` for a session-only key.
        let detail = `HTTP ${r.status}`;
        try {
          detail = detailOf(await r.json()) ?? detail;
        } catch {
          /* non-JSON body */
        }
        setSaveError(detail);
        return;
      }
      // Success — parse the requires_restart echo so we can render the
      // appropriate banner.
      let kinds: string[] = [];
      try {
        const j = (await r.json()) as PatchResponse;
        kinds = j.requires_restart ?? j.requires_restart_kinds ?? [];
      } catch {
        /* PATCH bodies are documented JSON; tolerate missing body */
      }
      setRestartBanner(restartBannerLines(kinds));
      seededFor.current = null;
      setEditing(false);
      await mutate();
      // Surface the new admin_username (or any other key consumed
      // elsewhere) to other pages on next render.
      void globalMutate(SWR_KEY);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return {
    draft,
    snapshot,
    dirty,
    isLoading,
    error,
    editing,
    saving,
    saveError,
    restartBanner,
    setField,
    edit,
    cancel,
    reset,
    save,
  };
}
