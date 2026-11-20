'use client';

// Update banner (spec 2026-11-20). Reads GET /api/system/update-status,
// which the daily update check fills. A security advisory that applies to
// the running version is red and stays until the upgrade; a newer release
// is one dismissible line (dismissed per version, server-side).

import { usePathname } from 'next/navigation';
import { useState } from 'react';
import useSWR from 'swr';
import { authFetch, authFetchJSON } from '@/lib/auth-fetch';
import { isUnauthRoute } from '@/components/nav-bar';

type Advisory = {
  id: string;
  severity?: string | null;
  title?: string | null;
  fixed_in?: string | null;
  url?: string | null;
};
type UpdateStatus = {
  current: string;
  latest: { version: string; url?: string } | null;
  update_available: boolean;
  advisories: Advisory[];
};

const vtag = (v: string) => (v.startsWith('v') ? v : `v${v}`);

export function UpdateBanner() {
  const path = usePathname() ?? '';
  const off = isUnauthRoute(path);
  const { data, mutate } = useSWR<UpdateStatus>(off ? null : '/api/system/update-status', authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
    refreshInterval: 6 * 3600 * 1000,
  });
  const [hidden, setHidden] = useState<string | null>(null);
  if (off || !data) return null;

  const advisories = Array.isArray(data.advisories) ? data.advisories : [];
  if (advisories.length > 0) {
    return (
      <div role="alert" className="border-b border-red-800 bg-red-950/70 text-red-100 text-sm">
        <ul className="container mx-auto px-4 py-2 space-y-1">
          {advisories.map((a) => (
            <li key={a.id}>
              <strong className="font-semibold">Security advisory {a.id}</strong>
              {a.severity ? ` (${a.severity})` : ''}: {a.title ?? 'see the advisory'}
              {a.fixed_in ? ` — fixed in ${vtag(a.fixed_in)}` : ''}.{' '}
              {a.url && (
                <a href={a.url} target="_blank" rel="noopener noreferrer" className="underline hover:text-white">
                  Details
                </a>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const latest = data.latest;
  if (!data.update_available || !latest || hidden === latest.version) return null;

  async function dismiss() {
    if (!latest) return;
    setHidden(latest.version);
    try {
      await authFetch('/api/system/update-status/dismiss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version: latest.version }),
      });
    } finally {
      void mutate();
    }
  }

  return (
    <div role="status" className="border-b border-emerald-900 bg-emerald-950/60 text-emerald-100 text-sm">
      <div className="container mx-auto px-4 py-2 flex items-center justify-between gap-4">
        <span>
          LM Warden {vtag(latest.version)} is available (you run {data.current}).{' '}
          {latest.url && (
            <a href={latest.url} target="_blank" rel="noopener noreferrer" className="underline hover:text-white">
              Release notes
            </a>
          )}
        </span>
        <button
          type="button"
          onClick={() => void dismiss()}
          className="text-emerald-300 hover:text-white text-xs uppercase tracking-wide"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}
