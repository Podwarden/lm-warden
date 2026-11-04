"use client";

// /key/settings — what the admin set on this key, visible and locked; the key's name is the one thing its holder
// may change.

import { useEffect, useState } from "react";

import { KeyShell, useKeyAuthRedirect } from "@/components/key-area/key-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type MeKey, meWrite, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";

function guardText(k: MeKey): { value: string; note: string | null } {
  const g = k.guard_effective;
  const mode = g?.mode ?? "off";
  if (mode === "off") return { value: K.settings.guardOff, note: null };
  const parts = [K.settings.guardModes[mode] ?? mode, ...(g?.packs ?? [])];
  if (g?.rules) parts.push(K.settings.ownRules(g.rules));
  return { value: parts.join(" · "), note: g?.source?.mode === "platform" ? K.settings.fromPlatform : null };
}

function Row({ label, children, action }: { label: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <tr className="border-b border-slate-800/60 align-top">
      <td className="w-1/3 px-2 py-2">
        {label}
        <small className="block text-xs text-slate-500 sm:hidden">{K.settings.byAdmin}</small>
      </td>
      <td className="px-2 py-2">{children}</td>
      <td className="hidden px-2 py-2 text-xs text-slate-500 sm:table-cell">{action ?? K.settings.byAdmin}</td>
    </tr>
  );
}

function Settings() {
  const { data, mutate, error } = useMe<MeKey>("/api/me/key");
  const [name, setName] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [writeError, setWriteError] = useState<unknown>(null);
  useKeyAuthRedirect(error ?? writeError);

  useEffect(() => {
    if (data) setName(data.name);
  }, [data]);

  if (!data) return null;
  const guard = guardText(data);
  const trimmed = name.trim();

  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !trimmed || trimmed === data?.name) return;
    setBusy(true);
    setNote(null);
    try {
      const r = await meWrite("/api/me/key", "PATCH", { name: trimmed });
      if (!r.ok) {
        setNote(K.settings.saveError);
        return;
      }
      await mutate((await r.json()) as MeKey, { revalidate: false });
      setNote(K.settings.saved);
    } catch (err) {
      setWriteError(err);
      setNote(K.settings.saveError);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800 text-left text-xs text-slate-500">
            {K.settings.cols.map((c, i) => (
              <th key={i} className={i === 2 ? "hidden px-2 py-1.5 font-medium sm:table-cell" : "px-2 py-1.5 font-medium"}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <Row label={K.settings.models}>{data.allowed_models?.length ? data.allowed_models.join(", ") : K.settings.allModels}</Row>
          <Row label={K.settings.priority}>{data.priority ?? "—"}</Row>
          <Row label={K.settings.thinking}>
            {K.settings.thinkingValues[data.thinking_policy ?? "client"] ?? data.thinking_policy}
          </Row>
          <Row label={K.settings.relay}>{data.anthropic_relay ? K.settings.yes : K.settings.no}</Row>
          <Row label={K.settings.guard}>
            {guard.value}
            {guard.note && <small className="block text-xs text-slate-400">{guard.note}</small>}
          </Row>
          <tr className="align-top">
            <td className="px-2 py-2">
              <label htmlFor="key-name">{K.settings.nameLabel}</label>
            </td>
            <td className="px-2 py-2" colSpan={2}>
              <form onSubmit={save} className="flex flex-wrap items-center gap-2">
                <Input
                  id="key-name"
                  className="max-w-xs"
                  maxLength={64}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
                <Button type="submit" variant="outline" disabled={busy || !trimmed || trimmed === data.name}>
                  {K.settings.save}
                </Button>
                {note && (
                  <span role="status" className="text-xs text-slate-400">
                    {note}
                  </span>
                )}
              </form>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function KeySettingsPage() {
  return (
    <KeyShell>
      <Settings />
    </KeyShell>
  );
}
