"use client";

// /forest/login — a key holder opens their own forest with their inference API key (Task 8).
// The key goes once, as `Authorization: Bearer`, to the same-origin `POST /api/forest/login`, which answers with a
// forest-only token scoped to that key. Only the token and its expiry are kept (this tab's sessionStorage); the key is
// never stored, logged, put in a URL or echoed in an error. Public: no admin session is needed (session-gate.tsx).

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useSWRConfig } from "swr";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { isForestTokenCacheKey, storeForestToken } from "@/lib/forest/token";

const MESSAGES: Record<number, string> = {
  401: "Key not accepted",
  403: "Key is paused",
  429: "Too many attempts — wait and try again",
};
const GENERIC = "Could not sign in. Try again in a moment.";
const STORAGE = "Signed in, but this browser blocked tab storage, so the forest cannot be opened. Allow site storage and try again.";

export default function ForestLoginPage() {
  const router = useRouter();
  const { mutate } = useSWRConfig();
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !key) return;
    setError(null);
    setBusy(true);
    try {
      const r = await fetch("/api/forest/login", {
        method: "POST",
        credentials: "omit",
        headers: { Authorization: `Bearer ${key}` },
      });
      if (!r.ok) {
        setError(MESSAGES[r.status] ?? GENERIC);
        return;
      }
      const body = (await r.json()) as { token?: unknown; expires_in?: unknown };
      if (typeof body.token !== "string" || !body.token || typeof body.expires_in !== "number") {
        setError(GENERIC);
        return;
      }
      if (!storeForestToken(body.token, body.expires_in)) {
        setError(STORAGE);
        return;
      }
      setKey("");
      // SWR keeps the last view's ForestAuthError (and data) across remounts: drop the forest-token entries first, so
      // the fresh view never bounces straight back here off the stale error.
      await mutate(isForestTokenCacheKey, undefined, { revalidate: false });
      router.replace("/forest?key=1");
    } catch {
      // never the error text: a network error message could carry the request (and so the key)
      setError(GENERIC);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="max-w-sm mx-auto mt-20 space-y-4" aria-labelledby="forest-login-title">
      <h1 id="forest-login-title" className="text-xl font-semibold">
        Your session forest
      </h1>
      <p className="text-sm text-slate-400">
        Sign in with your API key to see the forest of your own sessions. The key is used once to sign in and is never
        stored.
      </p>
      <label className="block">
        API key
        <Input
          name="forest-api-key"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={key}
          onChange={(e) => setKey(e.target.value)}
        />
      </label>
      {error && (
        <p role="alert" className="text-red-500 text-sm">
          {error}
        </p>
      )}
      <Button type="submit" disabled={busy || !key}>
        Open forest
      </Button>
    </form>
  );
}
