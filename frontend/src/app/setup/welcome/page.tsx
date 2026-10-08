'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { pathForStep } from '@/lib/setup-steps';

export default function WelcomePage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function begin() {
    setError(null);
    setBusy(true);
    try {
      const r = await fetch('/api/setup/welcome', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        setError(body.detail ?? 'Failed to start setup');
        return;
      }
      // The server reports the step to resume at: "gpus" on a fresh start,
      // or the actual current step if setup was already in progress
      // (idempotent re-entry after Back/reload).
      const body = await r.json().catch(() => ({}));
      router.push(pathForStep(body.step));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-semibold">Welcome to LM Warden</h1>
      <p className="text-sm text-slate-400">
        This wizard will configure your GPUs, HuggingFace token, and admin account.
      </p>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button onClick={begin} disabled={busy}>Begin</Button>
    </section>
  );
}
