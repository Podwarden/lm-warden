"use client";

import { SettingField } from "@/components/settings/setting-field";
import type { FieldHint } from "@/lib/settings-hints";
import type {
  Draft,
  RuntimeKey,
} from "@/components/settings/hooks/use-runtime-settings";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";

// ---------------------------------------------------------------------------
// RuntimeField — per-key dispatcher that maps a RuntimeKey to the right
// SettingField kind + typed setter. Extracted from the now-deleted
// runtime-tab.tsx so each of the four runtime sub-tabs can render its
// own subset of fields without re-implementing the switch.
// ---------------------------------------------------------------------------
//
// Secret keys (admin_password, hf_token) get a "Leave blank to keep
// current value" annotation appended to the hint. The snapshot strips
// the `***` sentinel during seeding, so we can't reliably distinguish
// first-ever setup from "value is set" at this layer — the unconditional
// annotation is the conservative call. Worst case: on first setup the
// operator sees a slightly redundant hint, which is a minor nit, not a
// bug. We do NOT mutate the imported hint copy — clone it so the badge /
// restart kind on the original FieldHint stay untouched.
// ---------------------------------------------------------------------------

interface RuntimeFieldProps {
  fieldKey: RuntimeKey;
  hint: FieldHint;
  draft: Draft;
  setField: <K extends RuntimeKey>(k: K, v: Draft[K]) => void;
  disabled: boolean;
  gpus?: GpuInfo[];
}

export function RuntimeField({
  fieldKey,
  hint,
  draft,
  setField,
  disabled,
  gpus = [],
}: RuntimeFieldProps) {
  if (fieldKey === "admin_password" || fieldKey === "hf_token") {
    const annotatedHint: FieldHint = {
      ...hint,
      hint: `${hint.hint} (Leave blank to keep current value.)`,
    };
    return (
      <SettingField
        kind="text"
        field={annotatedHint}
        value={draft[fieldKey]}
        onChange={(v) => setField(fieldKey, v)}
        disabled={disabled}
      />
    );
  }

  switch (fieldKey) {
    case "admin_username":
    case "vllm_version":
    case "public_url":
    case "analytics_ga4_id":
    case "analytics_google_ads_id":
    case "search_console_verification":
      return (
        <SettingField
          kind="text"
          field={hint}
          value={draft[fieldKey]}
          onChange={(v) => setField(fieldKey, v)}
          disabled={disabled}
        />
      );
    case "default_gpu_indices":
      return (
        <SettingField
          kind="gpu-set"
          field={hint}
          value={draft.default_gpu_indices}
          gpus={gpus}
          onChange={(v) => setField("default_gpu_indices", v)}
          disabled={disabled}
        />
      );
    case "default_token_expiration_days":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.default_token_expiration_days}
          onChange={(v) => setField("default_token_expiration_days", v)}
          min={0}
          max={3650}
          step={1}
          disabled={disabled}
        />
      );
    case "rotation_grace_hours":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.rotation_grace_hours}
          onChange={(v) => setField("rotation_grace_hours", v)}
          min={0}
          max={720}
          step={1}
          disabled={disabled}
        />
      );
    case "session_access_ttl_minutes":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.session_access_ttl_minutes}
          onChange={(v) => setField("session_access_ttl_minutes", v)}
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "session_refresh_ttl_days":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.session_refresh_ttl_days}
          onChange={(v) => setField("session_refresh_ttl_days", v)}
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "sse_ticket_ttl_seconds":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.sse_ticket_ttl_seconds}
          onChange={(v) => setField("sse_ticket_ttl_seconds", v)}
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "log_retention_lines":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.log_retention_lines}
          onChange={(v) => setField("log_retention_lines", v)}
          min={1}
          step={1}
          disabled={disabled}
        />
      );
    case "landing_page_enabled":
      return (
        <SettingField
          kind="boolean"
          field={hint}
          value={draft.landing_page_enabled}
          onChange={(v) => setField("landing_page_enabled", v)}
          disabled={disabled}
        />
      );
    case "watchdog_enabled":
      return (
        <SettingField
          kind="boolean"
          field={hint}
          value={draft.watchdog_enabled}
          onChange={(v) => setField("watchdog_enabled", v)}
          disabled={disabled}
        />
      );
    case "watchdog_restore_on_boot":
      return (
        <SettingField
          kind="boolean"
          field={hint}
          value={draft.watchdog_restore_on_boot}
          onChange={(v) => setField("watchdog_restore_on_boot", v)}
          disabled={disabled}
        />
      );
    case "watchdog_interval_s":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.watchdog_interval_s}
          onChange={(v) => setField("watchdog_interval_s", v)}
          disabled={disabled}
        />
      );
    case "watchdog_failure_threshold":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.watchdog_failure_threshold}
          onChange={(v) => setField("watchdog_failure_threshold", v)}
          disabled={disabled}
        />
      );
    case "watchdog_max_restarts":
      return (
        <SettingField
          kind="number"
          field={hint}
          value={draft.watchdog_max_restarts}
          onChange={(v) => setField("watchdog_max_restarts", v)}
          disabled={disabled}
        />
      );
  }
  // Exhaustive — TypeScript narrows fieldKey to `never` here. Returning
  // null lets a future RUNTIME_HINTS addition fail visibly (missing
  // field on the rendered tab) instead of crashing the whole page.
  return null;
}
