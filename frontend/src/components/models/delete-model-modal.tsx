"use client";

/**
 * Delete-model confirmation modal with optional "free cache" chain.
 *
 * Closes #105: deleting a model row used to leave its HF cache blob
 * behind, and the only way to reclaim that disk was to remember the
 * repo name and go to /cache. This modal opt-in chains the cache
 * delete onto the row delete so the operator never has to remember.
 *
 * Two-step chain — order matters:
 *   1. DELETE /api/models/{id}        (always runs)
 *   2. DELETE /api/cache/models/{repo}?force=true   (only if step 1
 *      succeeded AND the checkbox was ticked)
 *
 * Step 2 is best-effort. If it fails (cache row already gone, GC
 * collision, FS error) we still report success on the row and ask
 * the operator to clean up at /cache. The opposite ordering would
 * orphan the model row if cache delete fails — never do that.
 *
 * Closing the modal during an in-flight request is suppressed; the
 * parent re-renders to the right state once the chain settles.
 */

import { useRef, useState } from "react";
import Link from "next/link";
import { authFetch } from "@/lib/auth-fetch";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";

interface DeleteModelModalProps {
  open: boolean;
  onClose: () => void;
  modelId: string;
  servedModelName: string;
  hfRepo: string;
  /** Called only when the row delete itself succeeded — parent should
   *  refresh its SWR list / navigate away. Cache-delete outcome is
   *  reported via `cacheFailedNotice` rendered inline, so the parent
   *  doesn't have to plumb a toast. */
  onDeleted: () => void;
}

export function DeleteModelModal({
  open,
  onClose,
  modelId,
  servedModelName,
  hfRepo,
  onDeleted,
}: DeleteModelModalProps) {
  const [freeCache, setFreeCache] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Surfaced inline (not a toast) when the row was deleted but the
  // cache leg failed — keeps the message visible and discoverable
  // rather than disappearing on a 5s timer.
  const [cacheFailedNotice, setCacheFailedNotice] = useState<string | null>(
    null,
  );
  // Synchronous double-submit guard. setBusy(true) doesn't take effect
  // until React re-renders, so a rapid double-click on Delete can fire
  // two concurrent DELETEs; the second 404s on the row that the first
  // already removed. Mirrors the same pattern in /models/[id]/page.tsx.
  const inflight = useRef(false);
  // Focus the safe "Cancel" action by default for destructive confirms,
  // matching the rest of the app's confirm-modal hygiene.
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  function handleClose() {
    if (busy) return;
    setError(null);
    setCacheFailedNotice(null);
    setFreeCache(false);
    onClose();
  }

  async function runDelete() {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setError(null);
    setCacheFailedNotice(null);
    try {
      // ---- Step 1: row delete ------------------------------------
      const rowRes = await authFetch(`/api/models/${modelId}`, {
        method: "DELETE",
      });
      if (!rowRes.ok) {
        let detail = `HTTP ${rowRes.status}`;
        try {
          const body = await rowRes.json();
          if (body && typeof body.detail === "string") detail = body.detail;
        } catch {
          /* non-JSON body — fall through to status code */
        }
        setError(detail);
        return;
      }

      // ---- Step 2 (optional): cache delete -----------------------
      // Best-effort. Only runs if the operator opted in. We never
      // skip step 1 because of step 2's outcome.
      if (freeCache) {
        try {
          const cacheRes = await authFetch(
            `/api/cache/models/${encodeURIComponent(hfRepo)}?force=true`,
            { method: "DELETE" },
          );
          if (!cacheRes.ok && cacheRes.status !== 404) {
            // 404 means the cache row was already gone (GC, manual
            // delete, or simply never populated) — treat as success
            // for the chained-delete contract. Any other non-OK
            // status surfaces as a follow-up notice but does NOT
            // roll back the row delete.
            setCacheFailedNotice(
              "Model removed; cache delete failed — visit /cache to retry.",
            );
          }
        } catch {
          setCacheFailedNotice(
            "Model removed; cache delete failed — visit /cache to retry.",
          );
        }
      }

      // Notify the parent regardless of cache outcome — the row IS
      // gone, the list should refresh, and the operator should land
      // back on /models.
      onDeleted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Delete model"
      initialFocusRef={cancelRef}
    >
      <div className="space-y-4 text-sm">
        <p className="text-slate-300">
          Delete{" "}
          <span className="font-mono text-slate-100">{servedModelName}</span>?
          This removes the model from the warden registry.
        </p>

        <label className="flex items-start gap-3 rounded-md border border-slate-700 bg-slate-900/40 p-3">
          <input
            type="checkbox"
            checked={freeCache}
            onChange={(e) => setFreeCache(e.target.checked)}
            disabled={busy}
            className="mt-0.5 h-4 w-4 cursor-pointer rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-400"
            data-testid="free-cache-checkbox"
          />
          <span className="flex-1">
            <span className="block font-medium text-slate-100">
              Also free cache
            </span>
            <span className="mt-1 block text-xs text-slate-400">
              Delete the cached weights for{" "}
              <span className="font-mono text-slate-300">{hfRepo}</span> from
              the HF cache. Frees disk; the next pull re-downloads.
            </span>
          </span>
        </label>

        {error && (
          <div
            role="alert"
            data-testid="delete-error"
            className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          >
            {error}
          </div>
        )}

        {cacheFailedNotice && (
          <div
            // role="alert" (not "status") — partial-failure announcement
            // on a destructive-action path. Screen-reader users with
            // polite live regions would otherwise not be interrupted;
            // assertive is the correct urgency for "your model is gone
            // but the cache delete you opted into failed".
            role="alert"
            data-testid="cache-failed-notice"
            className="rounded-md border border-amber-700 bg-amber-900/30 p-3 text-sm text-amber-200"
          >
            {cacheFailedNotice}{" "}
            <Link
              href="/cache"
              className="font-semibold underline underline-offset-2 hover:text-amber-100"
            >
              Open cache
            </Link>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-2">
          <Button
            ref={cancelRef}
            variant="ghost"
            onClick={handleClose}
            disabled={busy}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={runDelete}
            disabled={busy}
            data-testid="delete-confirm"
          >
            {busy ? "Deleting…" : "Delete"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
