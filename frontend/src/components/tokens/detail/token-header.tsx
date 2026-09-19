"use client";

// Token details header (spec 2026-09-18 §4.2; mockup `.head`, `.title-row`,
// `.sub`, `.actions`, `.paused-band`). The mockup's `.crumbs` row ("API
// tokens / <name>") is the app-wide breadcrumb strip now — the page registers
// the token's name with `useBreadcrumb`. The strip (30 px) sits above <main>
// where the crumbs row (13 px × 1.5 + 10 px margin = 29.5 px) used to sit
// inside it, so the title row lands where it did.

import { useEffect, useRef, useState, type FormEvent } from "react";
import type { TokenDetail } from "@/lib/token-series";
import { fmtDateTime, hhmm, parseSqliteUtc } from "@/lib/token-format";
import { badge, btn, type BadgeTone } from "./styles";

export type DetailStatusLabel = "Paused" | "Revoked" | "Expired" | "Grace" | "Active";

/** Spec §4.2: Paused → Revoked → Expired → Grace → Active. `is_revoked` is the
 *  server's `revoked_at <= now`, so a rotated key whose grace ran out is
 *  Revoked and one still inside its window is Grace. */
export function deriveDetailStatus(
  t: Pick<TokenDetail, "is_paused" | "is_revoked" | "is_expired" | "rotated_at">,
): { label: DetailStatusLabel; tone: BadgeTone } {
  if (t.is_paused) return { label: "Paused", tone: "paused" };
  if (t.is_revoked) return { label: "Revoked", tone: "revoked" };
  if (t.is_expired) return { label: "Expired", tone: "revoked" };
  if (t.rotated_at != null) return { label: "Grace", tone: "grace" };
  return { label: "Active", tone: "active" };
}

export interface TokenHeaderProps {
  token: TokenDetail;
  nowSec: number;
  pausing: boolean;
  testing: boolean;
  deleting: boolean;
  pauseError: string | null;
  onRename: (name: string) => Promise<boolean>;
  onPauseToggle: () => void;
  onTest: () => void;
  onRotate: () => void;
  onDelete: () => void;
}

function sameLocalDay(a: number, b: number): boolean {
  return new Date(a * 1000).toDateString() === new Date(b * 1000).toDateString();
}

export function TokenHeader({
  token, nowSec, pausing, testing, deleting, pauseError,
  onRename, onPauseToggle, onTest, onRotate, onDelete,
}: TokenHeaderProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(token.name);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const status = deriveDetailStatus(token);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  function startEdit() {
    setDraft(token.name);
    setEditing(true);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const v = draft.trim();
    if (!v) return;
    if (v === token.name) {
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      if (await onRename(v)) setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  const created = parseSqliteUtc(token.created_at);
  const expires = parseSqliteUtc(token.expires_at);
  const expiryText =
    expires == null ? "Never expires"
      : token.is_expired ? `Expired ${fmtDateTime(expires, nowSec)}`
        : `Expires ${fmtDateTime(expires, nowSec)}`;
  const pausedAt = parseSqliteUtc(token.paused_at);
  const pausedSince =
    pausedAt == null ? "" : sameLocalDay(pausedAt, nowSec) ? hhmm(pausedAt) : fmtDateTime(pausedAt, nowSec);
  // Spec §4.2: Pause is disabled on a revoked or expired key. Resume stays
  // available so a paused key that later expired can still be un-paused.
  const pauseDisabled = pausing || (!token.is_paused && (token.is_revoked || token.is_expired));

  return (
    <>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            {editing ? (
              <form className="flex items-center gap-2" onSubmit={submit}>
                <input
                  ref={inputRef}
                  aria-label="Token name"
                  maxLength={64}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") setEditing(false);
                  }}
                  className="w-[min(420px,70vw)] rounded-md border border-chat-accent-strong bg-chat-surface px-2.5 py-0.5 text-[22px] font-semibold text-chat-fg"
                />
                <button type="submit" className={btn("primary")} disabled={saving}>
                  Save name
                </button>
                <button type="button" className={btn()} onClick={() => setEditing(false)}>
                  Cancel
                </button>
              </form>
            ) : (
              <>
                <h1 className="m-0 text-[26px] font-semibold tracking-[-0.01em]">{token.name}</h1>
                <button
                  type="button"
                  aria-label="Rename token"
                  title="Rename"
                  onClick={startEdit}
                  className="cursor-pointer rounded border-0 bg-transparent p-1 text-chat-dim hover:text-chat-fg"
                >
                  <svg className="inline align-baseline" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                  </svg>
                </button>
              </>
            )}
            <span className={badge(status.tone)}>{status.label}</span>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-3.5 text-[13px] text-chat-muted">
            <span className="!font-mono text-[13px] text-chat-muted">{token.prefix}…</span>
            {created != null && <span>Created {fmtDateTime(created, nowSec)}</span>}
            <span>{expiryText}</span>
          </div>
        </div>

        <div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={btn("pause")} onClick={onPauseToggle} disabled={pauseDisabled}>
              {token.is_paused ? "Resume" : "Pause"}
            </button>
            <button type="button" className={btn()} onClick={onTest} disabled={testing}>
              Test
            </button>
            <button type="button" className={btn()} onClick={onRotate} disabled={token.rotated_at != null}>
              Rotate
            </button>
            <button type="button" className={btn("danger")} onClick={onDelete} disabled={deleting}>
              Delete
            </button>
          </div>
          {pauseError && (
            <p role="alert" className="mb-0 mt-1.5 text-right text-[12px] text-vw-danger-fg">
              {pauseError}
            </p>
          )}
        </div>
      </div>

      {token.is_paused && (
        <div
          role="status"
          className="mt-4 rounded-md border border-chat-accent/35 bg-vw-amber-bg/[.28] px-3.5 py-2.5 text-[13px] text-vw-band-fg"
        >
          <strong className="font-semibold text-chat-fg">Paused since {pausedSince}.</strong>{" "}
          New requests with this key get{" "}
          <span className="!font-mono text-[13px]">403 token paused</span>. Requests that were already
          running will finish. Resume to let it back in immediately.
        </div>
      )}
    </>
  );
}
