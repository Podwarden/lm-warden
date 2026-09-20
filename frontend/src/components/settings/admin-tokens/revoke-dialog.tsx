"use client";

import { useRef, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import { ADMIN_TOKENS_URL, responseError, type AdminToken } from "@/lib/admin-tokens";

interface RevokeDialogProps {
  token: AdminToken;
  /** The row this token was refreshed from, when it is still listed. */
  predecessor?: AdminToken;
  onClose: () => void;
}

export function RevokeAdminTokenDialog({ token, predecessor, onClose }: RevokeDialogProps) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Destructive confirm: focus lands on the safe action.
  const cancelRef = useRef<HTMLButtonElement>(null);

  async function revoke() {
    setError(null);
    setBusy(true);
    try {
      const r = await authFetch(`${ADMIN_TOKENS_URL}/${encodeURIComponent(token.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) {
        setError(await responseError(r, `Failed to revoke the token (HTTP ${r.status})`));
        return;
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open onClose={busy ? () => {} : onClose} title="Revoke admin token" initialFocusRef={cancelRef}>
      <div className="space-y-4">
        <p className="text-sm text-slate-400">
          Revoke <strong className="text-slate-100">{token.name}</strong>? Requests with it are
          refused from now on, and its open streams are closed. The row stays in the list,
          dimmed, so its activity stays readable.
        </p>
        {predecessor?.status === "grace" && (
          // M3 (final-review 2026-09-19): revoking a successor does not end
          // its predecessor's grace window -- they are separate rows. Warn only
          // while that window is actually open.
          <p className="text-sm text-amber-400">
            Its previous secret (<span className="font-mono">{predecessor.prefix}</span>) stays
            valid until its grace ends — revoke that row too.
          </p>
        )}
        {error && (
          <p
            role="alert"
            className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
          >
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2">
          <Button ref={cancelRef} type="button" variant="outline" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="button" variant="destructive" onClick={() => void revoke()} disabled={busy}>
            {busy ? "Revoking…" : "Revoke"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
