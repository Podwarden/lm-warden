"use client";

/**
 * Force-unload confirmation for a model stuck in a transient status.
 *
 * Closes #244: #236 opened `POST /api/models/{id}/unload?force=true` for
 * `loading` and `unloading` — the statuses a stranded row actually sits
 * in — but the detail page kept mirroring the OLD gate (`loaded` /
 * `failed` only), so the escape hatch was reachable only by curl with a
 * JWT and a CSRF token. That is exactly the operator who cannot use it.
 *
 * Why a modal and not a bare button: from these statuses the action is
 * genuinely destructive. It terminates whatever process the supervisor
 * still holds — which may be an engine mid-startup that is about to come
 * up — and releases its port and GPU claim. The confirmation focuses
 * Cancel so a stray click on the row button cannot fire it, and asks
 * nothing more than a second click so an operator mid-recovery is not
 * fighting the UI.
 *
 * On success the backend writes the row to `pulled` (weights stay on
 * disk) so the model can be loaded again — the modal tells the operator
 * that up front, because "what happens to my model?" is the question
 * they are asking before they click.
 */

import { useRef, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";

/** The statuses `_unloadable_statuses(force=True)` adds over the plain
 *  gate (app/models/routes_api.py). Kept as a type so the copy below has
 *  to handle each one explicitly. */
export type ForceUnloadStatus = "loading" | "unloading";

interface ForceUnloadModalProps {
  open: boolean;
  onClose: () => void;
  modelId: string;
  servedModelName: string;
  status: ForceUnloadStatus;
  /** Called only when the backend accepted the force unload (202) —
   *  the parent should revalidate the row so the new status reflects. */
  onUnloaded: () => void;
}

export function ForceUnloadModal({
  open,
  onClose,
  modelId,
  servedModelName,
  status,
  onUnloaded,
}: ForceUnloadModalProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Synchronous double-submit guard — same pattern as DeleteModelModal and
  // the detail page's runAction: setBusy lands on the next render, a fast
  // second click would fire a concurrent POST before the disabled prop does.
  const inflight = useRef(false);
  // Focus the safe action by default for a destructive confirm.
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  function handleClose() {
    if (busy) return;
    setError(null);
    onClose();
  }

  async function runForceUnload() {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch(`/api/models/${modelId}/unload?force=true`, {
        method: "POST",
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          if (body && typeof body.detail === "string") detail = body.detail;
        } catch {
          /* non-JSON body — fall through to status code */
        }
        setError(detail);
        return;
      }
      onUnloaded();
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
      title="Force unload"
      initialFocusRef={cancelRef}
    >
      <div className="space-y-4 text-sm">
        <p className="text-slate-300">
          Force unload{" "}
          <span className="font-mono text-slate-100">{servedModelName}</span>?
          It is currently{" "}
          <span className="font-mono text-slate-100">{status}</span>, and a
          normal unload is refused from that status.
        </p>

        <div className="rounded-md border border-amber-700 bg-amber-900/30 p-3 text-xs text-amber-200">
          {status === "loading" ? (
            <p>
              This kills the engine process if one still exists — including
              one that is mid-startup and about to come up — and releases its
              port and GPU claim. If the load is still making progress in Live
              logs, cancel and wait instead.
            </p>
          ) : (
            <p>
              This kills whatever the previous unload left behind and releases
              its port and GPU claim. If Live logs still show the engine
              shutting down, cancel and wait instead.
            </p>
          )}
          <p className="mt-2">
            If the warden restarted mid-{status === "loading" ? "load" : "unload"},
            there is no process to kill: the row is just stranded, and this
            is the only way out of it from the UI.
          </p>
        </div>

        <p className="text-xs text-slate-400">
          The model returns to{" "}
          <span className="font-mono text-slate-300">pulled</span>. Weights
          stay on disk; you can load it again straight away.
        </p>

        {error && (
          <div
            role="alert"
            data-testid="force-unload-error"
            className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          >
            {error}
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
            onClick={runForceUnload}
            disabled={busy}
            data-testid="force-unload-confirm"
          >
            {busy ? "Force unloading…" : "Force unload"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
