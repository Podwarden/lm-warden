"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LoadProgressCompact } from "@/components/models/load-progress/load-progress-compact";
import type { LoadProgressOut } from "@/lib/load-progress";
import { modelHref } from "@/lib/routes";
import { Mark } from "@/components/kv/parts";
import { KV_STORE_LABEL, MARK } from "@/lib/kv";

export type ModelStatus =
  | "registered"
  | "pulling"
  | "pulled"
  | "loading"
  | "loaded"
  | "unloading"
  | "failed";

export interface ModelRow {
  id: string;
  served_model_name: string;
  hf_repo: string;
  hf_revision: string;
  gpu_indices: number[];
  tensor_parallel_size: number | null;
  status: ModelStatus;
  pulled_bytes: number;
  pulled_total: number | null;
  last_error: string | null;
  /**
   * Which engine serves this row (sub-project C). NULL / absent means vLLM --
   * decision D6, no backfill -- and renders NO badge at all. A badge on every
   * card is noise; a badge that appears exactly when something is unusual is
   * information.
   */
  backend?: string | null;
  /** The KV cache store tier (plan 2); vLLM only. */
  kv_store?: "off" | "ram" | "ram+nvme";
  /** Whether the engine keeps an isolated key's cache apart (admin models list, O4). */
  cache_isolation?: "isolated" | "unsupported";
}

/** Display name for a backend, or null when there is nothing worth showing. */
export function backendBadgeLabel(
  backend: string | null | undefined,
): string | null {
  if (!backend || backend === "vllm") return null;
  return backend === "llamacpp" ? "llama.cpp" : backend;
}

function badgeVariantForStatus(
  status: ModelStatus,
): "default" | "success" | "warning" | "error" | "info" {
  switch (status) {
    case "loaded":
    case "pulled":
      return "success";
    case "loading":
    case "pulling":
    case "unloading":
      return "info";
    case "failed":
      return "error";
    case "registered":
    default:
      return "default";
  }
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}

export function ModelCard({
  model,
  loadProgress,
}: {
  model: ModelRow;
  /** Live load-progress snapshot for the compact bar; only meaningful while
   *  ``status === "loading"``.  The list page passes the matching entry from
   *  ``useActiveLoadProgress()`` by model id, or ``undefined`` if none. */
  loadProgress?: LoadProgressOut;
}) {
  const pct =
    model.status === "pulling" && model.pulled_total && model.pulled_total > 0
      ? Math.min(100, Math.round((model.pulled_bytes / model.pulled_total) * 100))
      : null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-2 space-y-0">
        <div className="min-w-0 flex-1">
          <CardTitle className="truncate">
            <Link
              href={modelHref(model.id)}
              className="hover:underline focus-visible:underline focus-visible:outline-none"
            >
              {model.served_model_name}
            </Link>
          </CardTitle>
          <p className="mt-1 truncate text-xs text-slate-400">
            <span className="font-mono">{model.hf_repo}</span>
            <span className="mx-1 text-slate-500">@</span>
            <span className="font-mono">{model.hf_revision}</span>
          </p>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5">
          {backendBadgeLabel(model.backend) !== null && (
            <Badge variant="info" data-testid="backend-badge">
              {backendBadgeLabel(model.backend)}
            </Badge>
          )}
          <Badge variant={badgeVariantForStatus(model.status)}>{model.status}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {/* These are GPU INDICES, not a count -- "GPUs: 1" on a model pinned
            to index 1 reads as "one GPU". Same wording as the detail page. */}
        <p className="text-slate-300">
          {model.gpu_indices.length === 1 ? "GPU index:" : "GPU indices:"}{" "}
          {model.gpu_indices.length > 0 ? (
            <span className="font-mono" data-testid="model-card-gpu-indices">
              {model.gpu_indices.join(", ")}
            </span>
          ) : (
            <span className="text-slate-500">none</span>
          )}
        </p>
        {(model.kv_store !== undefined || model.cache_isolation !== undefined) && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span data-testid="model-kv-store">
              {(model.backend ?? "vllm") === "vllm" ? (
                <>
                  KV cache store: <b className="font-semibold text-slate-200">{KV_STORE_LABEL[model.kv_store ?? "off"]}</b>
                </>
              ) : (
                "KV cache store not available"
              )}
            </span>
            {model.cache_isolation === "unsupported" && (
              <Mark tone="warn" testId="model-isolation-refused" title="Requests with an isolated key get 403 on this model: its engine cannot keep one key's cache apart from another's.">
                {MARK.refused}
              </Mark>
            )}
          </div>
        )}
        {model.status === "pulling" && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>
                {formatBytes(model.pulled_bytes)}
                {model.pulled_total ? ` / ${formatBytes(model.pulled_total)}` : ""}
              </span>
              {pct !== null && <span>{pct}%</span>}
            </div>
            <div
              className="h-1.5 w-full overflow-hidden rounded bg-slate-700"
              role="progressbar"
              aria-valuenow={pct ?? undefined}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Pull progress"
            >
              <div
                className="h-full bg-emerald-500 transition-[width]"
                style={{ width: pct !== null ? `${pct}%` : "20%" }}
              />
            </div>
          </div>
        )}
        {/* Compact load-progress bar while loading (Task 8): status pill,
            overall bar, current step name — the brief's compact form. */}
        {model.status === "loading" && (
          <LoadProgressCompact progress={loadProgress} />
        )}
        {/* The red ``Error:`` line only renders for a real failure.  While
            loading, ``last_error`` is not shown — the colour rule from the
            owner: no red unless the state is failed.  The compact
            load-progress bar owns the loading state. */}
        {model.status === "failed" && model.last_error && (
          <p className="break-words text-xs text-red-400">
            <span className="font-semibold">Error:</span> {model.last_error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
