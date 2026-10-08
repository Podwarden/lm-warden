"use client";

import {
  RUNTIME_SESSIONS_KEYS,
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
// Sessions & Tokens tab — Browser session / Token defaults / Streaming.
//
// Browser-session fields require a warden restart (the JWT secret + TTLs
// are picked up at boot). Token defaults and streaming TTL are no-restart
// — they only affect new tokens / new SSE connects.
// ---------------------------------------------------------------------------

const BROWSER_SESSION_KEYS: RuntimeKey[] = [
  "session_access_ttl_minutes",
  "session_refresh_ttl_days",
];
const TOKEN_DEFAULTS_KEYS: RuntimeKey[] = [
  "default_token_expiration_days",
  "rotation_grace_hours",
];
const STREAMING_KEYS: RuntimeKey[] = ["sse_ticket_ttl_seconds"];

export function SessionsTab() {
  const controls = useRuntimeSettings(RUNTIME_SESSIONS_KEYS);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;

  return (
    <SettingsTabShell title="Sessions & Tokens" controls={controls}>
      {draft && (
        <>
          <SettingSection
            title="Browser session"
            subtitle="How long a logged-in browser stays authenticated."
            restartChip="warden-restart"
          >
            {BROWSER_SESSION_KEYS.map((k) => {
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
            title="Token defaults"
            subtitle="Pre-fills the Token Create dialog. Existing tokens are unaffected."
          >
            {TOKEN_DEFAULTS_KEYS.map((k) => {
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
            title="Streaming"
            subtitle="One-shot tickets used to authenticate SSE connections."
          >
            {STREAMING_KEYS.map((k) => {
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
