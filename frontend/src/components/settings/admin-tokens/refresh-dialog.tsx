"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import {
  ADMIN_TOKENS_URL,
  DEFAULT_GRACE_HOURS,
  GRACE_OPTIONS,
  responseError,
  type AdminToken,
  type AdminTokenIssued,
} from "@/lib/admin-tokens";
import { SecretReveal } from "./secret-reveal";

interface RefreshDialogProps {
  token: AdminToken;
  base: string;
  onClose: () => void;
}

function graceNote(hours: number): string {
  if (hours === 0) return "The old secret stopped working just now.";
  return `The old secret keeps working for ${hours === 1 ? "1 hour" : `${hours} hours`}, so you can swap it without downtime.`;
}

export function RefreshAdminTokenDialog({ token, base, onClose }: RefreshDialogProps) {
  const [grace, setGrace] = useState<number>(DEFAULT_GRACE_HOURS);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [issued, setIssued] = useState<AdminTokenIssued | null>(null);
  const graceId = useId();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const r = await authFetch(
        `${ADMIN_TOKENS_URL}/${encodeURIComponent(token.id)}/rotate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ grace_hours: grace }),
        },
      );
      if (!r.ok) {
        setError(await responseError(r, `Failed to refresh the token (HTTP ${r.status})`));
        return;
      }
      setIssued((await r.json()) as AdminTokenIssued);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open onClose={busy ? () => {} : onClose} title={issued ? "Admin token refreshed" : `Refresh ${token.name}`}>
      {issued ? (
        <SecretReveal
          plaintext={issued.plaintext}
          base={base}
          onDone={onClose}
          note={<p className="text-sm text-slate-400">{graceNote(grace)}</p>}
        />
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <p className="text-sm text-slate-400">
            Refreshing issues a new secret for <strong className="text-slate-100">{token.name}</strong>{" "}
            with the same term, starting now. The old secret keeps working for the grace period.
          </p>
          <label htmlFor={graceId} className="block space-y-1">
            <span className="text-sm">Grace period</span>
            <select
              id={graceId}
              value={grace}
              onChange={(e) => setGrace(Number(e.target.value))}
              className="h-9 w-full rounded-md border border-slate-600 bg-slate-900 px-3 text-sm text-slate-100"
            >
              {GRACE_OPTIONS.map((o) => (
                <option key={o.hours} value={o.hours}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          {error && (
            <p
              role="alert"
              className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
            >
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy}>
              {busy ? "Refreshing…" : "Refresh"}
            </Button>
          </div>
        </form>
      )}
    </Modal>
  );
}
