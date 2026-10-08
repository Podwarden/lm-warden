"use client";

import {
  RUNTIME_MAINTENANCE_KEYS,
  RUNTIME_HINTS,
  type FieldHint,
} from "@/lib/settings-hints";
import {
  useRuntimeSettings,
  type RuntimeKey,
} from "@/components/settings/hooks/use-runtime-settings";
import { SettingsTabShell } from "@/components/settings/settings-tab-shell";
import { SettingSection } from "@/components/settings/setting-section";
import { RuntimeField } from "@/components/settings/runtime-field";

// ---------------------------------------------------------------------------
// Maintenance tab — vLLM runtime / Logs.
//
// `vllm_version` requires a warden restart (the image tag is read at boot
// when subprocesses are spawned). `log_retention_lines` only affects the
// next /api/logs read — no restart needed.
// ---------------------------------------------------------------------------

const VLLM_RUNTIME_KEYS: RuntimeKey[] = ["vllm_version"];
const LOGS_KEYS: RuntimeKey[] = ["log_retention_lines"];

export function MaintenanceTab() {
  const controls = useRuntimeSettings(RUNTIME_MAINTENANCE_KEYS);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;

  return (
    <SettingsTabShell title="Maintenance" controls={controls}>
      {draft && (
        <>
          <SettingSection
            title="vLLM runtime"
            subtitle="The vLLM image tag used to spawn model subprocesses."
            restartChip="warden-restart"
          >
            {VLLM_RUNTIME_KEYS.map((k) => {
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
            title="Logs"
            subtitle="How much of each log we surface in the UI tail."
          >
            {LOGS_KEYS.map((k) => {
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
        </>
      )}
    </SettingsTabShell>
  );
}
