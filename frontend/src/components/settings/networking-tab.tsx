"use client";

import {
  RUNTIME_NETWORKING_KEYS,
  RUNTIME_ANALYTICS_KEYS,
  RUNTIME_UPDATE_KEYS,
  RUNTIME_HINTS,
  type FieldHint,
} from "@/lib/settings-hints";
import { useRuntimeSettings } from "@/components/settings/hooks/use-runtime-settings";
import { SettingsTabShell } from "@/components/settings/settings-tab-shell";
import { SettingSection } from "@/components/settings/setting-section";
import { RuntimeField } from "@/components/settings/runtime-field";
import { HomePageField } from "@/components/settings/home-page-field";
import { useEdition, hasFeature } from "@/lib/edition";

// ---------------------------------------------------------------------------
// Networking tab — two sections:
//
//   * Public access (public_url, then the home page control). `public_url` (#154)
//     powers `getPublicBaseUrl()` so user-facing snippets (curl examples,
//     OpenAI client configs) render the right hostname when the warden is
//     behind a reverse proxy.
//   * Website analytics (spec 2026-10-07 §8): the Google tag IDs for the
//     public website. Nothing loads from Google unless one is set.
//
// One useRuntimeSettings scope covers both, so a single Save writes both.
// ---------------------------------------------------------------------------

const SCOPE = [
  ...RUNTIME_NETWORKING_KEYS,
  ...RUNTIME_UPDATE_KEYS,
  ...RUNTIME_ANALYTICS_KEYS,
] as const;
const HOME_KEYS: readonly string[] = ["home_page_mode", "home_page_off_action"];
type ScopeKey = (typeof SCOPE)[number];

export function NetworkingTab() {
  const controls = useRuntimeSettings(SCOPE);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;
  const edition = useEdition();
  // Website analytics keys exist only with the site package installed.
  const showAnalytics = hasFeature(edition, "site.analytics");
  // An edition's own home page tab replaces the Off / Built-in control.
  const customHome = hasFeature(edition, "home.custom");

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
            {fields(RUNTIME_NETWORKING_KEYS.filter((k) => !HOME_KEYS.includes(k)))}
            {!customHome && <HomePageField draft={draft} setField={setField} disabled={disabled} />}
          </SettingSection>
          <SettingSection
            title="Updates"
            subtitle="Daily check of lmwarden.com for new releases and security advisories."
          >
            {fields(RUNTIME_UPDATE_KEYS)}
          </SettingSection>
          {showAnalytics && (
            <SettingSection
              title="Website analytics"
              subtitle="Google tag for the public website. Nothing loads unless an ID is set."
            >
              {fields(RUNTIME_ANALYTICS_KEYS)}
            </SettingSection>
          )}
        </>
      )}
    </SettingsTabShell>
  );
}
