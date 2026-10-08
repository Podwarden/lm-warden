"use client";

import {
  RUNTIME_NETWORKING_KEYS,
  RUNTIME_ANALYTICS_KEYS,
  RUNTIME_HINTS,
  type FieldHint,
} from "@/lib/settings-hints";
import { useRuntimeSettings } from "@/components/settings/hooks/use-runtime-settings";
import { SettingsTabShell } from "@/components/settings/settings-tab-shell";
import { SettingSection } from "@/components/settings/setting-section";
import { RuntimeField } from "@/components/settings/runtime-field";

// ---------------------------------------------------------------------------
// Networking tab — two sections:
//
//   * Public access (public_url + landing_page_enabled). `public_url` (#154)
//     powers `getPublicBaseUrl()` so user-facing snippets (curl examples,
//     OpenAI client configs) render the right hostname when the warden is
//     behind a reverse proxy.
//   * Website analytics (spec 2026-10-07 §8): the Google tag IDs for the
//     public website. Nothing loads from Google unless one is set.
//
// One useRuntimeSettings scope covers both, so a single Save writes both.
// ---------------------------------------------------------------------------

const SCOPE = [...RUNTIME_NETWORKING_KEYS, ...RUNTIME_ANALYTICS_KEYS] as const;
type ScopeKey = (typeof SCOPE)[number];

export function NetworkingTab() {
  const controls = useRuntimeSettings(SCOPE);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;

  const fields = (keys: readonly ScopeKey[]) =>
    draft &&
    keys.map((k) => {
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
    });

  return (
    <SettingsTabShell title="Networking" controls={controls}>
      {draft && (
        <>
          <SettingSection
            title="Public access"
            subtitle="How clients outside the host see this warden."
          >
            {fields(RUNTIME_NETWORKING_KEYS)}
          </SettingSection>
          <SettingSection
            title="Website analytics"
            subtitle="Google tag for the public website. Nothing loads unless an ID is set."
          >
            {fields(RUNTIME_ANALYTICS_KEYS)}
          </SettingSection>
        </>
      )}
    </SettingsTabShell>
  );
}
