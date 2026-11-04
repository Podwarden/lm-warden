"use client";

// Settings → Sensitive data: the platform's default guard settings (GET/PUT /api/guard/settings).
// Off saves null. Every key uses this unless it is switched off or has its own settings.

import { useEffect, useState } from "react";
import useSWR from "swr";

import { GuardForm, type FormMode } from "@/components/guard/guard-form";
import { GuardLog } from "@/components/guard/guard-log";
import { TryIt } from "@/components/guard/try-it";
import { authFetchJSON } from "@/lib/auth-fetch";
import { sendJSON } from "@/lib/guard/api";
import { G } from "@/lib/guard/strings";
import { type GuardSettings, normalise, type PacksResponse, payload } from "@/lib/guard/types";

export function GuardPlatformTab() {
  const packs = useSWR<PacksResponse>("/api/guard/packs", authFetchJSON);
  const saved = useSWR<{ guard: Partial<GuardSettings> | null }>("/api/guard/settings", authFetchJSON);
  const [value, setValue] = useState<GuardSettings | null>(null);
  const [mode, setMode] = useState<FormMode>("off");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  // The form starts from what is stored once both answers are in; later polls never overwrite an edit.
  useEffect(() => {
    if (value !== null || !saved.data || !packs.data) return;
    const g = saved.data.guard;
    setValue(normalise(g ?? {}, packs.data));
    setMode(g ? (g.mode === "block" ? "block" : "sanitize") : "off");
  }, [saved.data, packs.data, value]);

  if (saved.error || packs.error) return <p className="text-sm text-vw-danger-fg">{G.loadError}</p>;
  if (value === null) return null;

  async function save() {
    if (!value) return;
    setBusy(true);
    setNote(null);
    const r = await sendJSON("/api/guard/settings", "PUT", { guard: mode === "off" ? null : payload({ ...value, mode }) });
    setBusy(false);
    setNote(r.ok ? { ok: true, text: G.saved } : { ok: false, text: r.error ?? "" });
    if (r.ok) void saved.mutate();
  }

  return (
    <div className="max-w-3xl space-y-5">
      <div>
        <h2 className="text-lg font-semibold">{G.title}</h2>
        <p className="text-sm text-chat-dim">{G.intro}</p>
        <p className="mt-1 text-sm text-chat-dim">{G.platformIntro}</p>
      </div>
      <GuardForm
        value={value}
        mode={mode}
        allowOff
        packs={packs.data}
        onChange={(v, m) => {
          setValue(v);
          setMode(m);
        }}
      />
      <div className="flex items-center gap-3">
        <button type="button" className="rounded-md border border-vw-btn bg-vw-btn px-3.5 py-1.5 text-[13px] font-medium text-vw-on-btn disabled:opacity-45" disabled={busy} onClick={() => void save()}>
          {busy ? G.saving : G.save}
        </button>
        {note && (
          <span role={note.ok ? "status" : "alert"} className={note.ok ? "text-[12px] text-chat-dim" : "text-[12px] text-vw-danger-fg"}>
            {note.text}
          </span>
        )}
      </div>
      {mode !== "off" && <TryIt settings={{ ...value, mode }} />}
      <GuardLog />
    </div>
  );
}
