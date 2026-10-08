"use client";

import useSWR from "swr";
import {
  RUNTIME_GENERAL_KEYS,
  RUNTIME_HINTS,
  type FieldHint,
} from "@/lib/settings-hints";
import { useRuntimeSettings, type RuntimeKey } from "@/components/settings/hooks/use-runtime-settings";
import { SettingsTabShell } from "@/components/settings/settings-tab-shell";
import { SettingSection } from "@/components/settings/setting-section";
import { RuntimeField } from "@/components/settings/runtime-field";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";

// ---------------------------------------------------------------------------
// General tab — Identity / Hugging Face / Defaults for new models.
//
// Sections come from the #154 spec wireframe. Field ordering inside each
// section follows RUNTIME_GENERAL_KEYS in settings-hints.ts (the
// contract test pins that this tab covers exactly those keys).
// ---------------------------------------------------------------------------

const IDENTITY_KEYS: RuntimeKey[] = ["admin_username", "admin_password"];
const HUGGINGFACE_KEYS: RuntimeKey[] = ["hf_token"];
const DEFAULTS_KEYS: RuntimeKey[] = ["default_gpu_indices"];

export function GeneralTab() {
  const controls = useRuntimeSettings(RUNTIME_GENERAL_KEYS);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;

  // GPU inventory for the default_gpu_indices checkbox picker. /api/system/gpus
  // has a 2 s server-side cache so this is cheap; an empty list (no NVIDIA /
  // probe error) just yields ghost rows for any configured index.
  const { data: gpuData } = useSWR<{ gpus: GpuInfo[]; probed_at: string; probe_error: string | null }>(
    "/api/system/gpus",
    authFetchJSON,
  );
  const gpus = gpuData?.gpus ?? [];

  return (
    <SettingsTabShell title="General" controls={controls}>
      {draft && (
        <>
          <SettingSection
            title="Identity"
            subtitle="The single admin account used to log in."
          >
            {IDENTITY_KEYS.map((k) => {
              const hint: FieldHint | undefined = RUNTIME_HINTS[k];
              if (!hint) return null;
              return (
                <RuntimeField
                  key={k}
                  fieldKey={k}
                  hint={hint}
                  draft={draft}
                  setField={setField}
                  disabled={disabled}
                />
              );
            })}
          </SettingSection>

          <SettingSection
            title="Hugging Face"
            subtitle="Credentials and cache path used when pulling weights."
            restartChip="model-reload"
          >
            {HUGGINGFACE_KEYS.map((k) => {
              const hint: FieldHint | undefined = RUNTIME_HINTS[k];
              if (!hint) return null;
              return (
                <RuntimeField
                  key={k}
                  fieldKey={k}
                  hint={hint}
                  draft={draft}
                  setField={setField}
                  disabled={disabled}
                />
              );
            })}
          </SettingSection>

          <SettingSection
            title="Defaults for new models"
            subtitle="Pre-fills the Add Model modal. Does not affect existing models."
          >
            {DEFAULTS_KEYS.map((k) => {
              const hint: FieldHint | undefined = RUNTIME_HINTS[k];
              if (!hint) return null;
              return (
                <RuntimeField
                  key={k}
                  fieldKey={k}
                  hint={hint}
                  draft={draft}
                  setField={setField}
                  disabled={disabled}
                  gpus={gpus}
                />
              );
            })}
          </SettingSection>
        </>
      )}
    </SettingsTabShell>
  );
}
