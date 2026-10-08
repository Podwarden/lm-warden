"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import { copyToClipboard } from "@/lib/utils";

interface RotateTokenDialogProps {
  open: boolean;
  tokenId: string;
  onClose: () => void;
  /** Called after the success modal closes, with the new token's id. The
   *  details page navigates there (spec §4.2); the list row doesn't pass it. */
  onRotated?: (successorId: string) => void;
}

const GRACE_MIN = 0;
const GRACE_MAX = 720;
const DAYS_MIN = 0;
const DAYS_MAX = 3650;

interface RotateResponse {
  id: string;
  // #150 — the freshly-minted row keeps the ORIGINAL name; the old row
  // is renamed to `"{name} (old N)"`. Both names are surfaced so the
  // success modal can tell the operator exactly where each token went.
  name: string;
  plaintext: string;
  prefix?: string;
  rotated_from: string;
  renamed_to: string;
  // #185 — the grace the SERVER applied. The success copy narrates from this
  // rather than from what this component remembers submitting, so a future
  // server-side clamp cannot make the modal lie about a hard cut.
  grace_hours?: number;
}

// #185 — rotate is two distinct acts wearing one button. "grace" swaps creds
// without downtime; "immediate" (grace_hours=0) cuts a leaked key off on its
// very next request. Modelling it as a mode rather than a number means the
// hard revoke cannot happen without someone choosing it — see the blank-field
// trap in submit().
type RotateMode = "grace" | "immediate";

export function RotateTokenDialog({ open, tokenId, onClose, onRotated }: RotateTokenDialogProps) {
  const [mode, setMode] = useState<RotateMode>("grace");
  const [graceHours, setGraceHours] = useState("24");
  const [expiresInDays, setExpiresInDays] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Plaintext lives in component-local state and is wiped on close. The
  // §11.6 spec is explicit: "After the user dismisses the dialog, the
  // plaintext must NOT remain accessible — clear it from state on close."
  const [rotated, setRotated] = useState<RotateResponse | null>(null);
  const [copied, setCopied] = useState(false);
  // Tracks a clipboard write failure (HTTP context, permissions denied,
  // missing API). Same reasoning as the create dialog: plaintext is
  // surfaced once and we cannot silently swallow a failed copy.
  const [copyFailed, setCopyFailed] = useState(false);

  const graceId = useId();
  const daysId = useId();
  const modeName = useId();

  function reset() {
    // `mode` MUST be reset here. Leaving it behind would arm immediate
    // revoke for the NEXT token the operator rotates — the exact incident
    // #185 was filed about, arriving from the opposite direction.
    setMode("grace");
    setGraceHours("24");
    setExpiresInDays("");
    setError(null);
    setRotated(null);
    setCopied(false);
    setCopyFailed(false);
  }

  function handleClose() {
    const successorId = rotated?.id ?? null;
    reset();
    onClose();
    if (successorId && onRotated) onRotated(successorId);
  }

  async function submit() {
    setError(null);

    let grace: number;
    if (mode === "immediate") {
      grace = 0;
    } else {
      // Blank is NOT zero. `Number("") === 0` and `Number.isInteger(0)` is
      // true, so before #185 clearing this field — the natural gesture for
      // "never mind, leave the default" — silently POSTed grace_hours: 0 and
      // hard-revoked the predecessor. The radio above makes that path
      // near-unreachable; this guard is what keeps it that way.
      if (!graceHours.trim()) {
        setError("Enter a grace period in hours, or choose “Revoke the old token immediately”.");
        return;
      }
      grace = Number(graceHours);
      if (!Number.isInteger(grace) || grace < GRACE_MIN || grace > GRACE_MAX) {
        setError(`Grace hours must be an integer between ${GRACE_MIN} and ${GRACE_MAX}`);
        return;
      }
    }

    const body: Record<string, unknown> = { grace_hours: grace };
    if (expiresInDays.trim()) {
      const days = Number(expiresInDays);
      if (!Number.isInteger(days) || days < DAYS_MIN || days > DAYS_MAX) {
        setError(`Expires in days must be an integer between ${DAYS_MIN} and ${DAYS_MAX} (0 = never)`);
        return;
      }
      body.expires_in_days = days;
    }

    setBusy(true);
    try {
      const r = await authFetch(`/api/tokens/${encodeURIComponent(tokenId)}/rotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => null);
        const msg = (detail && typeof detail === "object" && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : null) ?? `Failed to rotate token (HTTP ${r.status})`;
        setError(msg);
        return;
      }
      const payload = (await r.json()) as RotateResponse;
      setRotated(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  async function handleCopy() {
    if (!rotated) return;
    setCopyFailed(false);
    // See create-token-dialog: shared copyToClipboard handles the
    // non-secure-context (#149) fallback and only throws when both
    // navigator.clipboard and document.execCommand("copy") fail.
    try {
      await copyToClipboard(rotated.plaintext);
      setCopied(true);
    } catch {
      setCopyFailed(true);
    }
  }

  return (
    <Modal
      open={open}
      onClose={busy ? () => {} : handleClose}
      title={rotated ? "Token rotated" : "Rotate token"}
    >
      {rotated ? (
        <div className="space-y-3">
          <p className="text-sm text-slate-300">
            This is the only time you&apos;ll see this token — copy it now.
          </p>
          <pre className="select-all whitespace-pre-wrap break-all rounded-md border border-slate-700 bg-slate-950 p-3 font-mono text-sm">
            {rotated.plaintext}
          </pre>
          {copyFailed && (
            <p className="text-sm text-red-500">
              Copy failed — select and copy the token manually.
            </p>
          )}
          <p className="text-xs text-slate-500" data-testid="rotate-outcome">
            New active token: <span className="font-mono">{rotated.name}</span>.
            {rotated.grace_hours === 0 ? (
              <>
                {" "}The previous token was renamed to{" "}
                <span className="font-mono">{rotated.renamed_to}</span> and is{" "}
                <span className="font-medium text-red-400">revoked now</span> — it
                will be rejected on its next request. Requests already running on
                the engine are not cancelled.
              </>
            ) : (
              <>
                {" "}The previous token was renamed to{" "}
                <span className="font-mono">{rotated.renamed_to}</span> and will
                keep working through the grace period, then expire.
              </>
            )}
          </p>
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button type="button" onClick={handleClose}>Done</Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-slate-300">
            Issue a new token to replace this one. Choose what happens to the
            existing token.
          </p>

          <fieldset className="space-y-2">
            <legend className="sr-only">What happens to the old token</legend>

            <label className="flex items-start gap-3 rounded-md border border-slate-700 bg-slate-900/40 p-3">
              <input
                type="radio"
                name={modeName}
                value="grace"
                checked={mode === "grace"}
                onChange={() => setMode("grace")}
                disabled={busy}
                className="mt-0.5 h-4 w-4 cursor-pointer border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-400"
                data-testid="rotate-mode-grace"
              />
              <span className="flex-1">
                <span className="block font-medium text-slate-100">
                  Keep the old token working for a grace period
                </span>
                <span className="mt-1 block text-xs text-slate-400">
                  Swap credentials in your clients without downtime. The old
                  token stops working when the window closes.
                </span>
              </span>
            </label>

            {mode === "grace" && (
              // Rendered only in grace mode so "clear the box" can no longer
              // be read as grace_hours: 0 (#185) — there is no box to clear
              // in immediate mode, and a blank one here is a validation error.
              <label htmlFor={graceId} className="block space-y-1 pl-7">
                <span className="text-sm">Grace period (hours)</span>
                <Input
                  id={graceId}
                  type="number"
                  value={graceHours}
                  onChange={(e) => setGraceHours(e.target.value)}
                  min={GRACE_MIN}
                  max={GRACE_MAX}
                  inputMode="numeric"
                  disabled={busy}
                />
                <span className="text-xs text-slate-500">
                  How long the old token keeps working. Default 24h, max 720h
                  (30 days).
                </span>
              </label>
            )}

            <label className="flex items-start gap-3 rounded-md border border-slate-700 bg-slate-900/40 p-3">
              <input
                type="radio"
                name={modeName}
                value="immediate"
                checked={mode === "immediate"}
                onChange={() => setMode("immediate")}
                disabled={busy}
                className="mt-0.5 h-4 w-4 cursor-pointer border-slate-600 bg-slate-800 text-red-500 focus:ring-red-400"
                data-testid="rotate-mode-immediate"
              />
              <span className="flex-1">
                <span className="block font-medium text-slate-100">
                  Revoke the old token immediately
                </span>
                <span className="mt-1 block text-xs text-slate-400">
                  For a leaked or abused key. The old token is rejected on its
                  very next request. Requests already running on the engine are{" "}
                  <span className="font-medium text-slate-300">not</span>{" "}
                  cancelled — they finish or hit the read timeout.
                </span>
              </span>
            </label>
          </fieldset>

          <label htmlFor={daysId} className="block space-y-1">
            <span className="text-sm">New token expires in days (optional)</span>
            <Input
              id={daysId}
              type="number"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(e.target.value)}
              placeholder="inherit from old token"
              min={DAYS_MIN}
              max={DAYS_MAX}
              inputMode="numeric"
            />
            <span className="text-xs text-slate-500">
              Leave blank to inherit the predecessor&apos;s expiry. 0 = never.
            </span>
          </label>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={handleClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="button" onClick={submit} disabled={busy}>
              {busy ? "Rotating…" : "Rotate"}
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
}
