"use client";

// TemplateList (#162) — presentational grid for the Templates manager page.
//
// Pure props in / callback out: the container page (templates/page.tsx) owns
// the SWR fetch and the DELETE round-trip; this component only renders the
// cards and surfaces the delete intent through onDelete(id).
//
// Two flavours of template:
//   - source === "builtin": shipped with the image, immutable. Carries a
//     "built-in" badge and exposes NO delete control.
//   - source === "user": operator-saved (e.g. via the try-stack "save working
//     combo" affordance). Exposes a delete control wired to onDelete.
import { Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export interface TemplateDTO {
  id: string;
  label: string;
  source: "builtin" | "user";
  hf_repo: string;
  // The merged template store can carry an engine combo (channel + vLLM
  // version, optional pinned image) or leave it unset for templates that
  // inherit the system default engine.
  engine: { channel: string; vllm_version: string; image: string | null } | null;
  // Knob fields the GET /api/models/templates list also carries (see backend
  // template_to_dict). Optional here because the TemplateList card only needs
  // id/label/source/hf_repo/engine — the add-model wizard reads these to
  // prefill its form (Task 10). Kept on the shared DTO so both consumers
  // agree on one wire shape.
  hf_revision?: string;
  dtype?: string;
  max_model_len?: number | null;
  tensor_parallel_size?: number;
  /** #286: replicas; absent on an older API (read as 1). */
  data_parallel_size?: number;
  gpu_memory_utilization?: number;
  trust_remote_code?: boolean;
}

export function TemplateList({
  templates,
  onDelete,
}: {
  templates: TemplateDTO[];
  onDelete: (id: string) => void;
}) {
  if (templates.length === 0) {
    return (
      <div
        data-testid="templates-empty-state"
        className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-slate-700 bg-slate-900/30 px-6 py-12 text-center"
      >
        <p className="text-base font-medium text-slate-200">No templates yet.</p>
        <p className="max-w-md text-sm text-slate-400">
          Built-in templates ship with the image. Save a working engine combo
          from a model&apos;s try-stack panel to create your own.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2" data-testid="template-list">
      {templates.map((t) => (
        <Card key={t.id}>
          <CardContent className="flex items-start justify-between gap-3">
            <div className="min-w-0 space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-medium text-slate-100">{t.label}</span>
                {t.source === "builtin" && (
                  <Badge variant="info" data-testid={`builtin-${t.id}`}>
                    built-in
                  </Badge>
                )}
              </div>
              <p className="truncate font-mono text-xs text-slate-400" title={t.hf_repo}>
                {t.hf_repo}
              </p>
              {t.engine && (
                <p className="text-xs text-slate-500">
                  {t.engine.channel} · vLLM {t.engine.vllm_version}
                  {t.engine.image ? ` · ${t.engine.image}` : ""}
                </p>
              )}
            </div>
            {t.source === "user" && (
              <Button
                variant="ghost"
                size="sm"
                aria-label={`Delete template ${t.label}`}
                data-testid={`delete-${t.id}`}
                onClick={() => onDelete(t.id)}
              >
                <Trash2 className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
