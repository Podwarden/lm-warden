"use client";

// "Everything else" (plan §4.4): the routing map's last row decides what
// happens to Claude models no rule matches. Change opens this dialog; Apply
// PATCHes `passthrough_unmatched` alone, straight away (it is not part of the
// Timeouts form any more).

import { useEffect, useId, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { errorDetail } from "@/components/tokens/use-token-actions";
import { Modal } from "@/components/ui/modal";
import { btn, META } from "@/components/router/styles";
import { cn } from "@/lib/utils";

export interface PassthroughDialogProps {
  open: boolean;
  /** The current `passthrough_unmatched`. */
  value: boolean;
  onClose: () => void;
  /** Called after a successful PATCH (re-fetch settings). */
  onSaved: () => void;
}

const OPTION =
  "flex cursor-pointer items-start gap-3 rounded-[10px] border border-vw-rule-soft/70 px-3.5 py-3 has-[:checked]:border-vw-live/60 has-[:checked]:bg-vw-ok-bg/15";

export function PassthroughDialog({ open, value, onClose, onSaved }: PassthroughDialogProps) {
  const [choice, setChoice] = useState(value);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const name = useId();

  useEffect(() => {
    if (open) {
      setChoice(value);
      setError(null);
    }
  }, [open, value]);

  async function apply() {
    setBusy(true);
    setError(null);
    try {
      const r = await authFetch("/api/router/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passthrough_unmatched: choice }),
      });
      if (!r.ok) {
        setError(await errorDetail(r, `Failed to save (HTTP ${r.status})`));
        return;
      }
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Every other Claude model">
      <fieldset className="m-0 space-y-2.5 border-0 p-0">
        <legend className="mb-2.5 text-[13.5px] text-chat-muted">
          What happens to Claude models and /v1 paths that no rule matches.
        </legend>
        <label className={OPTION}>
          <input
            type="radio"
            name={name}
            className="mt-1 accent-current"
            checked={choice}
            onChange={() => setChoice(true)}
          />
          <span>
            <span className="block font-semibold">Pass through to Anthropic</span>
            <span className={cn(META, "block")}>
              On the user&apos;s own login. Needs a key with May relay to Anthropic; other keys
              get 403 <code>relay_not_allowed</code>.
            </span>
          </span>
        </label>
        <label className={OPTION}>
          <input
            type="radio"
            name={name}
            className="mt-1 accent-current"
            checked={!choice}
            onChange={() => setChoice(false)}
          />
          <span>
            <span className="block font-semibold">Answer 404</span>
            <span className={cn(META, "block")}>
              Nothing is relayed. Claude Code reports the model as not found.
            </span>
          </span>
        </label>
      </fieldset>
      {error && (
        <div
          role="alert"
          className="mt-3 rounded-[10px] border border-vw-danger/55 bg-vw-danger-bg/30 px-3 py-2 text-[13px] text-vw-danger-fg"
        >
          {error}
        </div>
      )}
      <div className="mt-4 flex justify-end gap-2">
        <button type="button" className={btn()} onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className={btn("primary")}
          disabled={busy || choice === value}
          onClick={() => void apply()}
        >
          Apply
        </button>
      </div>
    </Modal>
  );
}
