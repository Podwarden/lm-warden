'use client';
import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { pathForStep } from '@/lib/setup-steps';
import { SetupCodeGate, type CodeSource } from './setup-code';

// keep in sync with STEPS in app/setup/state_machine.py
const steps = ['welcome', 'gpus', 'hf-token', 'admin', 'done'];

export default function SetupLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const path = pathname.split('/').pop() ?? '';
  // Whether this install wants the one-time setup code (a fresh install
  // does, except in the macOS app's own window), and where it is.
  const [lock, setLock] = useState<{ required: boolean; source: CodeSource }>({
    required: false,
    source: null,
  });

  // Sync the URL to the server's canonical wizard step. The wizard routes
  // are plain URLs, so browser Back / reload can render a page for a step
  // the server has already moved past (or not reached) — and the POST
  // guards would then reject the only visible action, stranding first-run
  // setup. On every wizard page load, ask the server where we are and
  // redirect there if the URL disagrees.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Public, cookie-less probe (same as the login entry gate).
        const r = await fetch('/api/setup/state', { credentials: 'omit' });
        if (!r.ok) return;
        const { step, code_required, code_source } = await r.json();
        if (cancelled) return;
        setLock({ required: code_required === true, source: code_source ?? null });
        const canonical = pathForStep(step);
        if (pathname !== canonical) router.replace(canonical);
      } catch {
        // Network/5xx: leave the rendered page. The server-side step guards
        // still enforce ordering; this sync is a convenience, not a gate.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pathname, router]);

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <ol className="flex gap-4 text-sm">
        {steps.map((s, i) => (
          <li key={s} className={path === s ? 'font-bold' : 'text-slate-500'}>{i + 1}. {s}</li>
        ))}
      </ol>
      <SetupCodeGate required={lock.required} source={lock.source}>
        {children}
      </SetupCodeGate>
    </div>
  );
}
