'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface Whoami {
  username: string;
  account_type: string;
}

export default function HfTokenPage() {
  const router = useRouter();
  const [token, setToken] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [whoami, setWhoami] = useState<Whoami | null>(null);
  const [busy, setBusy] = useState<'continue' | 'skip' | null>(null);

  async function send(payload: { hf_token: string | null }, kind: 'continue' | 'skip') {
    setError(null);
    setWhoami(null);
    setBusy(kind);
    try {
      const r = await fetch('/api/setup/hf_token', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        setError(body.detail ?? 'Failed to save HF token');
        return;
      }
      const data = await r.json();
      if (data.whoami) {
        // Render whoami in the same paint as the navigation kick-off — React
        // commits this state before the route transition completes, so the
        // user still gets a confirmation glimpse without leaking a timer.
        setWhoami(data.whoami as Whoami);
      }
      router.push('/setup/admin');
    } finally {
      setBusy(null);
    }
  }

  function onContinue(e: React.FormEvent) {
    e.preventDefault();
    if (!token) {
      setError('Enter a token or click Skip.');
      return;
    }
    void send({ hf_token: token }, 'continue');
  }

  function onSkip() {
    void send({ hf_token: null }, 'skip');
  }

  return (
    <form onSubmit={onContinue} className="space-y-4">
      <h1 className="text-2xl font-semibold">HuggingFace token</h1>
      <p className="text-sm text-slate-400">
        Optional. Required to download gated/private models from HuggingFace.
      </p>
      <label className="block space-y-1">
        <span className="text-sm">Token</span>
        <Input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="hf_..."
          // Suppresses browser password-manager save prompts; browsers ignore
          // autoComplete="off" on password-typed inputs.
          autoComplete="one-time-code"
        />
      </label>
      {whoami && (
        <p className="text-emerald-500 text-sm">
          Authenticated as {whoami.username} ({whoami.account_type})
        </p>
      )}
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <div className="flex gap-2">
        <Button type="submit" disabled={busy !== null}>Continue</Button>
        <Button type="button" variant="ghost" onClick={onSkip} disabled={busy !== null}>Skip</Button>
      </div>
    </form>
  );
}
