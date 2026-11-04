'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSWRConfig } from 'swr';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { setAccessToken } from '@/lib/auth-fetch';
import { pathForStep } from '@/lib/setup-steps';
import { K } from '@/lib/key-area/strings';
import { isKeyCacheKey, keyRequested, storeKeyToken } from '@/lib/sessions/key-token';
import { cn } from '@/lib/utils';

type Mode = 'login' | 'key';

// Admins sign in with a username and password (Login). Someone who was given an
// API key signs in with it (API key): the key goes once, as `Authorization:
// Bearer`, to `POST /api/forest/login`, which answers with a short-lived token
// scoped to that key. Only that token is kept (this tab's sessionStorage); the
// key is never stored, logged, put in a URL or echoed in an error.

function AdminForm() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const r = await fetch('/api/auth/login', {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', 'Origin': window.location.origin },
      body: JSON.stringify({ username, password }),
    });
    if (r.status === 429) {
      // Brute-force throttle (app/auth/throttle.py): too many failures from
      // this address or for this username. Retry-After is in seconds.
      const wait = Number(r.headers.get('Retry-After'));
      setError(
        Number.isFinite(wait) && wait > 0
          ? `Too many failed attempts. Try again in ${wait} seconds.`
          : 'Too many failed attempts. Try again later.',
      );
      return;
    }
    if (!r.ok) { setError('Invalid credentials'); return; }
    const { access_token, expires_in } = await r.json();
    // Pass expires_in (#97) so auth-fetch schedules the proactive refresh
    // timer at 80% of the access TTL. Without this the timer would only
    // start firing after the first user-driven refresh, leaving the
    // initial 15-minute window vulnerable to backend bounces.
    setAccessToken(
      access_token,
      typeof expires_in === 'number' ? expires_in : undefined,
    );
    router.replace('/stats');
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block">{K.login.username}<Input name="username" value={username} onChange={(e) => setUsername(e.target.value)} /></label>
      <label className="block">{K.login.password}<Input name="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit" className="w-full">{K.login.submit}</Button>
    </form>
  );
}

function KeyForm() {
  const router = useRouter();
  const { mutate } = useSWRConfig();
  const [key, setKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !key) return;
    setError(null);
    setBusy(true);
    try {
      const r = await fetch('/api/forest/login', {
        method: 'POST',
        credentials: 'omit',
        headers: { Authorization: `Bearer ${key}` },
      });
      if (!r.ok) {
        setError(K.login.keyErrors[r.status] ?? K.login.keyGeneric);
        return;
      }
      const body = (await r.json()) as { token?: unknown; expires_in?: unknown };
      if (typeof body.token !== 'string' || !body.token || typeof body.expires_in !== 'number') {
        setError(K.login.keyGeneric);
        return;
      }
      if (!storeKeyToken(body.token, body.expires_in)) {
        setError(K.login.keyStorage);
        return;
      }
      setKey('');
      // SWR keeps the last view's KeyAuthError (and data) across remounts: drop the key entries first.
      await mutate(isKeyCacheKey, undefined, { revalidate: false });
      router.replace('/key');
    } catch {
      // never the error text: a network error message could carry the request (and so the key)
      setError(K.login.keyGeneric);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block">
        {K.login.keyLabel}
        <Input
          name="api-key"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={key}
          onChange={(e) => setKey(e.target.value)}
        />
      </label>
      {error && <p role="alert" className="text-red-500 text-sm">{error}</p>}
      <Button type="submit" className="w-full" disabled={busy || !key}>{K.login.keySubmit}</Button>
      <p className="text-xs text-slate-500">{K.login.keyFine}</p>
    </form>
  );
}

export default function LoginPage() {
  // `?key=1` opens the API key tab (where key holders are sent back to).
  // Read from the location, not useSearchParams: the static export would need
  // a Suspense boundary for it, and this page renders nothing until the setup
  // probe below answers anyway.
  const [mode, setMode] = useState<Mode>(() => (keyRequested() ? 'key' : 'login'));
  // Entry gate: until we know setup is complete, render nothing so the
  // sign-in form does NOT flash for a fresh install (which must be funneled
  // to the setup wizard instead). On fetch failure we fall back to showing
  // the form so a transient backend hiccup can't trap the user.
  const [setupChecked, setSetupChecked] = useState(false);
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Public, cookie-less probe — omit credentials.
        const r = await fetch('/api/setup/state', { credentials: 'omit' });
        if (!r.ok) throw new Error(`setup state ${r.status}`);
        const { done, step } = await r.json();
        if (cancelled) return;
        // Fail toward the wizard: only a successful probe that explicitly
        // reports done === true shows the sign-in form. A malformed/non-boolean
        // body funnels to the wizard rather than exposing a sign-in form
        // for an account that may not exist. (Probe FAILURE is handled by the
        // catch below, which falls back to the form so a hiccup can't trap.)
        // Funnel to the server's CURRENT step, not hard-coded welcome — a
        // half-finished setup resumes where it left off instead of landing
        // on a welcome page whose only action the server would reject.
        if (done !== true) {
          router.replace(pathForStep(step));
          return;
        }
      } catch {
        // Network/5xx — fall through to the login form rather than trap.
      }
      if (!cancelled) setSetupChecked(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  if (!setupChecked) return null;

  const tab = (m: Mode, label: string) => (
    <button
      type="button"
      role="tab"
      id={`login-tab-${m}`}
      aria-selected={mode === m}
      aria-controls="login-panel"
      onClick={() => setMode(m)}
      className={cn(
        'flex-1 px-3 py-2 text-sm',
        mode === m ? 'bg-slate-800 text-slate-100 shadow-[inset_0_-2px_0] shadow-emerald-400' : 'text-slate-400 hover:text-slate-200',
      )}
    >
      {label}
    </button>
  );

  return (
    <div className="max-w-sm mx-auto mt-20 space-y-4">
      <h1 className="text-xl font-semibold">LM Warden</h1>
      <div role="tablist" aria-label="Sign in with" className="flex overflow-hidden rounded-lg border border-slate-800">
        {tab('login', K.login.tabLogin)}
        {tab('key', K.login.tabKey)}
      </div>
      <div role="tabpanel" id="login-panel" aria-labelledby={`login-tab-${mode}`}>
        {mode === 'login' ? <AdminForm /> : <KeyForm />}
      </div>
    </div>
  );
}
