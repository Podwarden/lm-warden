"use client";

// Templates manager (#162). Lists the merged template store (built-in +
// user-saved) and lets operators delete their own templates. Creation happens
// implicitly via the try-stack "save working combo" affordance on the model
// detail page — there is no manual create form here on purpose (the engine
// combo a template captures only makes sense relative to a model that booted).
import useSWR from "swr";
import { authFetch, authFetchJSON } from "@/lib/auth-fetch";
import { Skeleton } from "@/components/ui/skeleton";
import { TemplateList, type TemplateDTO } from "@/components/templates/template-list";

export default function TemplatesPage() {
  const { data, error, isLoading, mutate } = useSWR<TemplateDTO[]>(
    "/api/models/templates",
    authFetchJSON,
  );

  const templates = data ?? [];

  async function onDelete(id: string) {
    // Optimistically drop the row, then confirm against the server. On error
    // SWR rolls back to the last-good list via the revalidate below.
    await authFetch(`/api/models/templates/${id}`, { method: "DELETE" });
    await mutate();
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Templates</h1>
      </div>

      <p className="max-w-2xl text-sm text-slate-400">
        Engine templates prefill the add-model wizard with a known-good HF repo
        and engine combo. Built-in templates ship with the image; user templates
        are saved from a model&apos;s try-stack panel.
      </p>

      {isLoading && (
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      )}

      {error && !isLoading && (
        <p className="text-sm text-red-500">
          Failed to load templates
          {error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}

      {!isLoading && !error && (
        <TemplateList templates={templates} onDelete={onDelete} />
      )}
    </div>
  );
}
