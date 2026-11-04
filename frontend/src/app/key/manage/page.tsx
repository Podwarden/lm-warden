"use client";

// /key/manage — rotate (the new key is shown once and this tab switches its sign-in to it) and revoke (the key's
// name must be typed first; revoking ends every sign-in made with the key, this one included).

import { useState } from "react";
import { useSWRConfig } from "swr";

import { KeyShell, useKeyAuthRedirect, useKeySignOut } from "@/components/key-area/key-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type MeKey, meWrite, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";
import { clearKeyToken, isKeyCacheKey, storeKeyToken } from "@/lib/sessions/key-token";

function Rotate() {
  const { mutate } = useSWRConfig();
  const [grace, setGrace] = useState(24);
  const [busy, setBusy] = useState(false);
  const [fresh, setFresh] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [authError, setAuthError] = useState<unknown>(null);
  useKeyAuthRedirect(authError);

  async function rotate() {
    if (busy) return;
    setBusy(true);
    setNote(null);
    try {
      const r = await meWrite("/api/me/key/rotate", "POST", { grace_hours: grace });
      if (!r.ok) {
        setNote(K.manage.rotateError);
        return;
      }
      const body = (await r.json()) as { plaintext?: unknown; token?: unknown; expires_in?: unknown };
      if (typeof body.plaintext !== "string") {
        setNote(K.manage.rotateError);
        return;
      }
      setFresh(body.plaintext);
      // The rotation ended this key's sign-ins; the answer carries one for the new key.
      if (typeof body.token === "string" && typeof body.expires_in === "number") {
        storeKeyToken(body.token, body.expires_in);
      } else {
        clearKeyToken();
      }
      // Refetch with the new sign-in, keeping what is on screen (dropping it would unmount the new key).
      void mutate(isKeyCacheKey);
    } catch (err) {
      setAuthError(err);
      setNote(K.manage.rotateError);
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    if (!fresh) return;
    try {
      await navigator.clipboard.writeText(fresh);
      setCopied(true);
    } catch {
      /* clipboard blocked: the key stays on screen to copy by hand */
    }
  }

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-3">
      <h2 className="font-semibold">{K.manage.rotateTitle}</h2>
      <p className="mb-3 mt-1 text-sm text-slate-400">{K.manage.rotateBody}</p>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label htmlFor="grace">{K.manage.graceLabel}</label>
        <select
          id="grace"
          value={grace}
          onChange={(e) => setGrace(Number(e.target.value))}
          disabled={fresh !== null}
          className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1"
        >
          {K.manage.grace.map((g) => (
            <option key={g.value} value={g.value}>
              {g.label}
            </option>
          ))}
        </select>
        <Button onClick={() => void rotate()} disabled={busy || fresh !== null}>
          {busy ? K.manage.rotating : K.manage.rotate}
        </Button>
      </div>
      {note && (
        <p role="alert" className="mt-2 text-sm text-red-400">
          {note}
        </p>
      )}
      {fresh && (
        <div className="mt-4">
          <p className="text-xs text-slate-400">{K.manage.newKey}</p>
          <code className="my-2 block break-all rounded-md border border-dashed border-amber-500 bg-slate-950 p-2.5">
            {fresh}
          </code>
          <Button variant="outline" onClick={() => void copy()}>
            {copied ? K.manage.copied : K.manage.copy}
          </Button>
          <span className="ml-2 text-xs text-slate-500">{K.manage.newKeyFine}</span>
        </div>
      )}
    </section>
  );
}

function Revoke({ name }: { name: string }) {
  const signOut = useKeySignOut();
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [authError, setAuthError] = useState<unknown>(null);
  useKeyAuthRedirect(authError);

  async function revoke(e: React.FormEvent) {
    e.preventDefault();
    if (busy || typed !== name) return;
    setBusy(true);
    setNote(null);
    try {
      const r = await meWrite("/api/me/key/revoke", "POST", { confirm_name: typed });
      if (!r.ok) {
        setNote(K.manage.revokeError);
        return;
      }
      // The server has already ended every sign-in with this key; clear this tab's and leave.
      await signOut();
    } catch (err) {
      setAuthError(err);
      setNote(K.manage.revokeError);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-lg border border-red-950 bg-slate-900 px-4 py-3">
      <h2 className="font-semibold">{K.manage.revokeTitle}</h2>
      <p className="mb-3 mt-1 text-sm text-slate-400">{K.manage.revokeBody}</p>
      <form onSubmit={revoke} className="flex flex-wrap items-center gap-2 text-sm">
        <label htmlFor="confirm-name">{K.manage.confirmLabel}</label>
        <Input
          id="confirm-name"
          className="max-w-[14rem]"
          autoComplete="off"
          placeholder={name}
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
        />
        <Button type="submit" variant="outline" className="text-red-400" disabled={busy || typed !== name}>
          {K.manage.revoke}
        </Button>
      </form>
      {note && (
        <p role="alert" className="mt-2 text-sm text-red-400">
          {note}
        </p>
      )}
    </section>
  );
}

function Manage() {
  const { data } = useMe<MeKey>("/api/me/key");
  if (!data) return null;
  return (
    <div className="space-y-4">
      <Rotate />
      <Revoke name={data.name} />
    </div>
  );
}

export default function KeyManagePage() {
  return (
    <KeyShell>
      <Manage />
    </KeyShell>
  );
}
