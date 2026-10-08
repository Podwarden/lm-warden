"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

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

export function ModelCard({ model }: { model: ModelRow }) {
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
              href={`/models/${model.id}`}
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
        <div className="flex shrink-0 items-center gap-1.5">
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
        {model.last_error && (
          <p className="break-words text-xs text-red-400">
            <span className="font-semibold">Error:</span> {model.last_error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
