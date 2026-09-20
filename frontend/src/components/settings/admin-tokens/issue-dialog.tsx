"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import {
  ADMIN_TOKENS_URL,
  DEFAULT_EXPIRY,
  EXPIRY_DAYS,
  responseError,
  type AdminTokenIssued,
  type ExpiryChoice,
} from "@/lib/admin-tokens";
import { SecretReveal } from "./secret-reveal";

// Mirrors app/admin_tokens/routes_api.py::AdminTokenCreate (UX hints only;
// the server is the source of truth).
const NAME_MAX = 64;

interface IssueDialogProps {
  base: string;
  onClose: () => void;
}

// Mounted only while open (the tab renders it conditionally), so closing it
// discards the plaintext with the component.
export function IssueAdminTokenDialog({ base, onClose }: IssueDialogProps) {
  const [name, setName] = useState("");
  const [expiry, setExpiry] = useState<ExpiryChoice>(DEFAULT_EXPIRY);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [issued, setIssued] = useState<AdminTokenIssued | null>(null);
  const nameId = useId();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const trimmed = name.trim();
    if (trimmed.length < 1 || trimmed.length > NAME_MAX) {
      setError(`Name must be between 1 and ${NAME_MAX} characters`);
      return;
    }
    setBusy(true);
    try {
      const r = await authFetch(ADMIN_TOKENS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmed,
          expires_in_days: expiry === "never" ? null : expiry,
        }),
      });
      if (!r.ok) {
        setError(await responseError(r, `Failed to issue the token (HTTP ${r.status})`));
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
    <Modal
      open
      onClose={busy ? () => {} : onClose}
      title={issued ? "Admin token issued" : "Issue admin token"}
    >
      {issued ? (
        <SecretReveal plaintext={issued.plaintext} base={base} onDone={onClose} />
      ) : (
        <form onSubmit={submit} noValidate className="space-y-4">
          <label htmlFor={nameId} className="block space-y-1">
            <span className="text-sm">Name</span>
            <Input
              id={nameId}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="ci-deploy"
              autoComplete="off"
              maxLength={NAME_MAX}
            />
          </label>

          <fieldset className="space-y-2">
            <legend className="text-sm">Expires</legend>
            <div className="flex flex-wrap gap-4">
              {EXPIRY_DAYS.map((d) => (
                <label key={d} className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="radio"
                    name="admin-token-expiry"
                    value={d}
                    checked={expiry === d}
                    onChange={() => setExpiry(d)}
                  />
                  {d} days
                </label>
              ))}
            </div>
            <label className="flex items-center gap-2 border-t border-slate-700 pt-2 text-sm text-slate-300">
              <input
                type="radio"
                name="admin-token-expiry"
                value="never"
                checked={expiry === "never"}
                onChange={() => setExpiry("never")}
              />
              Never
            </label>
            {expiry === "never" && (
              <p
                data-testid="never-warning"
                role="note"
                className="rounded-md border border-amber-700 bg-amber-900/30 p-2 text-xs text-amber-200"
              >
                A token that never expires works until someone revokes it. If it
                leaks, it keeps working until you notice. Prefer an expiry and
                refresh the token before it runs out.
              </p>
            )}
          </fieldset>

          {error && (
            <p
              role="alert"
              className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
            >
              {error}
            </p>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy}>
              {busy ? "Issuing…" : "Issue"}
            </Button>
          </div>
        </form>
      )}
    </Modal>
  );
}
