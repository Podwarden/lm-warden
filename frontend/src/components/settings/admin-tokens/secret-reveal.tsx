"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/utils";
import { curlExample } from "@/lib/admin-tokens";

interface SecretRevealProps {
  plaintext: string;
  base: string;
  note?: React.ReactNode;
  onDone: () => void;
}

// Shown once, right after issue or refresh. The plaintext lives only in the
// parent dialog's state; the dialog is unmounted on close, so it is gone.
export function SecretReveal({ plaintext, base, note, onDone }: SecretRevealProps) {
  const [copied, setCopied] = useState(false);
  // A failed copy must be visible: the secret cannot be shown again.
  const [copyFailed, setCopyFailed] = useState(false);

  async function copy() {
    setCopyFailed(false);
    try {
      await copyToClipboard(plaintext);
      setCopied(true);
    } catch {
      setCopyFailed(true);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-amber-400">
        You will not see this again. Copy it now — only a hash is stored.
      </p>
      <pre
        data-testid="admin-token-plaintext"
        className="select-all whitespace-pre-wrap break-all rounded-md border border-slate-700 bg-slate-950 p-3 font-mono text-sm text-slate-200"
      >
        {plaintext}
      </pre>
      {copyFailed && (
        <p
          role="alert"
          className="rounded-md border border-red-700 bg-red-900/30 p-3 text-sm text-red-200"
        >
          Copy failed — select the token and copy it by hand.
        </p>
      )}
      {note}
      <div className="space-y-1">
        <p className="text-xs text-slate-500">
          Put it in <code>VW_ADMIN_TOKEN</code>, then try it:
        </p>
        <pre
          data-testid="admin-token-curl"
          className="whitespace-pre-wrap break-all rounded-md border border-slate-700 bg-slate-900 p-2 font-mono text-xs text-slate-400"
        >
          {curlExample(base)}
        </pre>
      </div>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" size="sm" onClick={() => void copy()}>
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button type="button" size="sm" onClick={onDone}>
          Done
        </Button>
      </div>
    </div>
  );
}
