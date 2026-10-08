'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import useSWR from 'swr';
import { Button } from '@/components/ui/button';

interface GpuInfo {
  index: number;
  name: string;
  memory_total_mib: number;
  memory_used_mib: number;
  utilization_pct: number;
}

const fetcher = (url: string) =>
  fetch(url, { credentials: 'include' }).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json() as Promise<GpuInfo[]>;
  });

export default function GpusPage() {
  const router = useRouter();
  const { data, error: loadError, isLoading } = useSWR<GpuInfo[]>('/api/setup/gpus', fetcher);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Initialize selection to all GPUs once data arrives.
  useEffect(() => {
    if (data) setSelected(new Set(data.map((g) => g.index)));
  }, [data]);

  function toggle(idx: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  async function submit() {
    setError(null);
    setBusy(true);
    try {
      const indices = Array.from(selected).sort((a, b) => a - b);
      const r = await fetch('/api/setup/gpus', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ allowed_gpu_indices: indices }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        setError(body.detail ?? 'Failed to save GPU selection');
        return;
      }
      router.push('/setup/hf-token');
    } finally {
      setBusy(false);
    }
  }

  if (isLoading) return <p className="text-sm text-slate-400">Detecting GPUs…</p>;
  if (loadError) return <p className="text-red-500 text-sm">Failed to load GPU list.</p>;
  const gpus = data ?? [];
  if (gpus.length === 0)
    return <p className="text-red-500 text-sm">No GPUs detected by nvidia-smi.</p>;

  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-semibold">Select GPUs</h1>
      <p className="text-sm text-slate-400">
        Choose which GPUs LM Warden may schedule models on.
      </p>
      <ul className="space-y-2">
        {gpus.map((g) => {
          const totalGib = (g.memory_total_mib / 1024).toFixed(1);
          return (
            <li key={g.index} className="flex items-center gap-3 rounded-md border border-slate-700 bg-slate-900 p-3">
              <input
                id={`gpu-${g.index}`}
                type="checkbox"
                checked={selected.has(g.index)}
                onChange={() => toggle(g.index)}
                className="h-4 w-4"
              />
              <label htmlFor={`gpu-${g.index}`} className="flex-1 cursor-pointer text-sm">
                <span className="font-mono text-slate-300">#{g.index}</span>
                <span className="ml-2">{g.name}</span>
                <span className="ml-2 text-slate-500">{totalGib} GiB</span>
              </label>
            </li>
          );
        })}
      </ul>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button onClick={submit} disabled={busy || selected.size === 0}>Continue</Button>
    </section>
  );
}
