"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import { copyToClipboard } from "@/lib/utils";

interface CreateTokenDialogProps {
  open: boolean;
  onClose: () => void;
}

// Mirror app/tokens/routes_api.py:TokenCreate constraints so the user gets
// immediate validation feedback. Backend is still source of truth — these
// are UX hints, not security.
const NAME_MIN = 1;
const NAME_MAX = 64;
const DAYS_MIN = 0;
const DAYS_MAX = 3650;
// priority mirrors the DB CHECK trigger (0..9). Default 5 matches the SQL
// column default in 0018_tokens_rate_priority.sql.
const PRIORITY_MIN = 0;
const PRIORITY_MAX = 9;
const PRIORITY_DEFAULT = 5;

interface CreateResponse {
  id: string;
  name: string;
  plaintext: string;
  prefix: string;
  preview: string;
  expires_at: string | null;
}

export function CreateTokenDialog({ open, onClose }: CreateTokenDialogProps) {
  const [name, setName] = useState("");
  const [expiresInDays, setExpiresInDays] = useState("365");
  const [priority, setPriority] = useState(String(PRIORITY_DEFAULT));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Holds the freshly-minted plaintext token. We deliberately keep this in
  // local component state (not a ref, not props) so closing the modal
  // unmounts the secret. The §11.6 spec calls this out: "the plaintext
  // must NOT remain accessible — clear it from state on close".
  const [created, setCreated] = useState<CreateResponse | null>(null);
  const [copied, setCopied] = useState(false);
  // Tracks a clipboard write failure (HTTP context, permissions denied,
  // jsdom-style missing API). Plaintext is shown ONCE, so silently
  // swallowing this state would let an operator close the dialog
  // believing they had captured the token — irrecoverable footgun.
  const [copyFailed, setCopyFailed] = useState(false);

  const nameId = useId();
  const daysId = useId();
  const priorityId = useId();

  function reset() {
    setName("");
    setExpiresInDays("365");
    setPriority(String(PRIORITY_DEFAULT));
    setError(null);
    setCreated(null);
    setCopied(false);
    setCopyFailed(false);
  }

  function handleClose() {
    // Wipe state — including the plaintext token — before bubbling the
    // close event. The parent will revalidate the list separately.
    reset();
    onClose();
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const trimmed = name.trim();
    if (trimmed.length < NAME_MIN || trimmed.length > NAME_MAX) {
      setError(`Name must be between ${NAME_MIN} and ${NAME_MAX} characters`);
      return;
    }

    let days = 365;
    if (expiresInDays.trim()) {
      const parsed = Number(expiresInDays);
      if (!Number.isInteger(parsed) || parsed < DAYS_MIN || parsed > DAYS_MAX) {
        setError(`Expires in days must be an integer between ${DAYS_MIN} and ${DAYS_MAX} (0 = never)`);
        return;
      }
      days = parsed;
    }

    const parsedPrio = Number(priority);
    if (!Number.isInteger(parsedPrio) || parsedPrio < PRIORITY_MIN || parsedPrio > PRIORITY_MAX) {
      setError(`Priority must be an integer between ${PRIORITY_MIN} and ${PRIORITY_MAX}`);
      return;
    }

    setBusy(true);
    try {
      const r = await authFetch("/api/tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmed,
          expires_in_days: days,
          priority: parsedPrio,
        }),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => null);
        const msg = (detail && typeof detail === "object" && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : null) ?? `Failed to create token (HTTP ${r.status})`;
        setError(msg);
        return;
      }
      const body = (await r.json()) as CreateResponse;
      setCreated(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  async function handleCopy() {
    if (!created) return;
    setCopyFailed(false);
    // copyToClipboard (lib/utils.ts) tries navigator.clipboard first then
    // falls back to a hidden-textarea + execCommand("copy") — required
    // for a LAN-HTTP deployment (#149) where the UI is served over plain
    // HTTP and navigator.clipboard is undefined. It throws only when
    // BOTH paths fail, so the "select manually" hint stays accurate.
    try {
      await copyToClipboard(created.plaintext);
      setCopied(true);
    } catch {
      setCopyFailed(true);
    }
  }

  return (
    // Don't let backdrop click / Escape close the modal while a POST is
    // in flight (same reasoning as add-model-modal): we'd unmount before
    // setError fires and the user wouldn't see the failure.
    <Modal
      open={open}
      onClose={busy ? () => {} : handleClose}
      title={created ? "Token created" : "Create API token"}
    >
      {created ? (
        <div className="space-y-3">
          <p className="text-sm text-slate-300">
            This is the only time you&apos;ll see this token — copy it now.
          </p>
          <pre data-testid="new-token" className="select-all whitespace-pre-wrap break-all rounded-md border border-slate-700 bg-slate-950 p-3 font-mono text-sm">
            {created.plaintext}
          </pre>
          {copyFailed && (
            <p className="text-sm text-red-500">
              Copy failed — select and copy the token manually.
            </p>
          )}
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-slate-500">
              Prefix: <span className="font-mono">{created.prefix}</span>
              {created.expires_at && (
                <> · Expires: <span className="font-mono">{created.expires_at}</span></>
              )}
            </span>
            <div className="flex gap-2">
              <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
                {copied ? "Copied" : "Copy"}
              </Button>
              <Button type="button" onClick={handleClose}>Done</Button>
            </div>
          </div>
        </div>
      ) : (
        <form onSubmit={submit} noValidate className="space-y-4">
          <label htmlFor={nameId} className="block space-y-1">
            <span className="text-sm">Name</span>
            <Input
              id={nameId}
              name="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              aria-required="true"
              placeholder="ci-bot"
              autoComplete="off"
              maxLength={NAME_MAX}
            />
          </label>

          <label htmlFor={daysId} className="block space-y-1">
            <span className="text-sm">Expires in days</span>
            <Input
              id={daysId}
              type="number"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(e.target.value)}
              placeholder="365"
              min={DAYS_MIN}
              max={DAYS_MAX}
              inputMode="numeric"
            />
            <span className="text-xs text-slate-500">
              0 = never expires. Default 365.
            </span>
          </label>

          <label htmlFor={priorityId} className="block space-y-1">
            <span className="text-sm">Priority (0–9)</span>
            <Input
              id={priorityId}
              type="number"
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
              min={PRIORITY_MIN}
              max={PRIORITY_MAX}
              step={1}
              inputMode="numeric"
            />
            <span className="text-xs text-amber-400">
              STRICT scheduling — higher priority is ALWAYS served first.
              A priority-0 token can wait indefinitely behind a steady stream
              of priority-9 traffic (starvation by design). Default 5.
            </span>
          </label>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={handleClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy}>
              {busy ? "Creating…" : "Create"}
            </Button>
          </div>
        </form>
      )}
    </Modal>
  );
}
