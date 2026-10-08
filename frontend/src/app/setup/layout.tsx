'use client';
import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { pathForStep } from '@/lib/setup-steps';

// keep in sync with STEPS in app/setup/state_machine.py
const steps = ['welcome', 'gpus', 'hf-token', 'admin', 'done'];

export default function SetupLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const path = pathname.split('/').pop() ?? '';

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
        const { step } = await r.json();
        if (cancelled) return;
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
      {children}
    </div>
  );
}
