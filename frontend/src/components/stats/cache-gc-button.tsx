"use client";

/**
 * "Garbage-collect" button for the cache surface on /cache.
 *
 * Click → preview: POSTs ``/api/cache/models/gc?dry_run=true`` and renders
 * the candidate list (orphans + stale-failed) in a modal with the total
 * bytes that would be freed. Operator clicks "Run GC" → second POST with
 * ``dry_run=false`` actually deletes; on success the modal closes and the
 * parent revalidates the SWR list via ``onMutate()``.
 *
 * The dry-run / confirm split is deliberate: GC removes data from disk
 * with no undo, and the spec calls it out as the riskiest action on the
 * page. Showing the exact list before the destructive call is what makes
 * it safe enough to expose on a stats page rather than a separate admin
 * route.
 *
 * See vllm-warden#114 + ``docs/superpowers/specs/2026-05-20-hf-cache-management-design.md``.
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { Badge } from "@/components/ui/badge";
import { authFetch } from "@/lib/auth-fetch";
import { formatBytes } from "@/lib/fit";
import type { MatchedModelRef } from "./cache-table";

interface GcCandidate {
  repo: string;
  reason: "orphan" | "failed_stale";
  size_bytes: number;
  matched_rows: MatchedModelRef[];
}

interface GcResult {
  dry_run: boolean;
  total_bytes_freed: number;
  candidates: GcCandidate[];
  deleted_paths: string[];
}

interface CacheGcButtonProps {
  /** Called after a successful (non-dry-run) GC so the parent revalidates
   *  the cache list. */
  onMutate: () => void;
}

export function CacheGcButton({ onMutate }: CacheGcButtonProps) {
  const [open, setOpen] = useState(false);
  // Phase: "idle" before the preview call, "preview" while the modal shows
  // candidates, "running" during the real GC, "done" after a successful
  // wipe (shows the deleted-paths summary briefly before the user closes).
  const [phase, setPhase] = useState<"idle" | "preview" | "running" | "done">("idle");
  const [preview, setPreview] = useState<GcResult | null>(null);
  const [result, setResult] = useState<GcResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadPreview(): Promise<void> {
    setError(null);
    setPhase("idle");
    setOpen(true);
    try {
      const r = await authFetch("/api/cache/models/gc?dry_run=true", {
        method: "POST",
      });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (typeof body?.detail === "string") msg = body.detail;
        } catch {
          /* keep HTTP fallback */
        }
        setError(msg);
        return;
      }
      const body = (await r.json()) as GcResult;
      setPreview(body);
      setPhase("preview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "preview failed");
    }
  }

  async function runGc(): Promise<void> {
    setPhase("running");
    setError(null);
    try {
      const r = await authFetch("/api/cache/models/gc?dry_run=false", {
        method: "POST",
      });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (typeof body?.detail === "string") msg = body.detail;
        } catch {
          /* keep HTTP fallback */
        }
        setError(msg);
        setPhase("preview");
        return;
      }
      const body = (await r.json()) as GcResult;
      setResult(body);
      setPhase("done");
      onMutate();
    } catch (err) {
      setError(err instanceof Error ? err.message : "GC failed");
      setPhase("preview");
    }
  }

  function closeModal(): void {
    setOpen(false);
    // Reset on close so the next click runs a fresh dry-run rather than
    // re-displaying a possibly-stale candidate list.
    setPhase("idle");
    setPreview(null);
    setResult(null);
    setError(null);
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={loadPreview}
        data-testid="cache-gc-button"
      >
        Garbage-collect
      </Button>
      <Modal
        open={open}
        onClose={closeModal}
        title="HF cache garbage collection"
        size="lg"
      >
        {error && (
          <p className="mb-3 rounded-md border border-red-500/40 bg-red-500/10 p-2 text-sm text-red-300"
             data-testid="cache-gc-error">
            {error}
          </p>
        )}

        {phase === "idle" && !error && (
          <p className="text-sm text-slate-300">Loading preview…</p>
        )}

        {phase === "preview" && preview && (
          <GcPreviewBody
            preview={preview}
            onCancel={closeModal}
            onRun={runGc}
          />
        )}

        {phase === "running" && (
          <p className="text-sm text-slate-300" data-testid="cache-gc-running">
            Deleting {preview?.candidates.length ?? 0} repo(s)…
          </p>
        )}

        {phase === "done" && result && (
          <GcResultBody result={result} onClose={closeModal} />
        )}
      </Modal>
    </>
  );
}

function GcPreviewBody({
  preview,
  onCancel,
  onRun,
}: {
  preview: GcResult;
  onCancel: () => void;
  onRun: () => void;
}) {
  if (preview.candidates.length === 0) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-slate-300" data-testid="cache-gc-empty">
          Nothing to collect — no orphaned or stale-failed cache entries.
        </p>
        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={onCancel}>
            Close
          </Button>
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-300">
        About to free{" "}
        <span className="font-semibold tabular-nums text-slate-100">
          {formatBytes(preview.total_bytes_freed)}
        </span>{" "}
        across <span className="font-semibold">{preview.candidates.length}</span>{" "}
        repo(s).
      </p>
      <div className="max-h-72 overflow-y-auto rounded-md border border-slate-700">
        <table className="w-full text-xs">
          <thead className="bg-slate-800 text-slate-400">
            <tr>
              <th scope="col" className="px-2 py-1 text-left">Repo</th>
              <th scope="col" className="px-2 py-1 text-left">Reason</th>
              <th scope="col" className="px-2 py-1 text-right">Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {preview.candidates.map((c) => (
              <tr key={c.repo} data-testid={`cache-gc-cand-${c.repo}`}>
                <td className="px-2 py-1 font-mono text-slate-100">{c.repo}</td>
                <td className="px-2 py-1">
                  <Badge variant={c.reason === "orphan" ? "default" : "error"}>
                    {c.reason === "orphan" ? "orphan" : "stale failed"}
                  </Badge>
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-200">
                  {formatBytes(c.size_bytes)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          variant="destructive"
          size="sm"
          onClick={onRun}
          data-testid="cache-gc-confirm"
        >
          Run GC
        </Button>
      </div>
    </div>
  );
}

function GcResultBody({
  result,
  onClose,
}: {
  result: GcResult;
  onClose: () => void;
}) {
  return (
    <div className="space-y-3" data-testid="cache-gc-done">
      <p className="text-sm text-slate-300">
        Freed{" "}
        <span className="font-semibold tabular-nums text-emerald-300">
          {formatBytes(result.total_bytes_freed)}
        </span>{" "}
        across {result.candidates.length} repo(s).
      </p>
      {result.deleted_paths.length > 0 && (
        <details className="rounded-md border border-slate-700 p-2 text-xs">
          <summary className="cursor-pointer text-slate-400">
            {result.deleted_paths.length} path(s) removed
          </summary>
          <ul className="mt-2 space-y-0.5 font-mono text-slate-300">
            {result.deleted_paths.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </details>
      )}
      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
}
