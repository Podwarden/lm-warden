"use client";

// The key holder's prompt cache (/key/settings). A shared key gets one button, "Keep my
// cache separate", which opens the warning in place: nothing is saved until the holder
// confirms. An isolated key gets no way back (O2: only an admin can share it again).
// Reset is there only for an isolated key and always asks first. The usage card shows
// the key's own figures only: never who dropped its parts, never other keys, the tier
// sizes or the isolation cost (the API does not send them; this never draws them).

import { useState } from "react";
import { btn, CARD, CARD_TITLE } from "@/components/tokens/detail/styles";
import {
  fmtGB,
  fmtGiB,
  fmtInt,
  fmtPct,
  sinceAgoWords,
  windowsHaveStore,
  type KvIsolation,
  type KvKeyReport,
  type ModelIsolation,
} from "@/lib/kv";
import { CacheUseCard } from "./cache-use-card";
import { ConfirmDialog, Notice } from "./parts";

export function HolderPromptCache({
  isolation,
  models,
  onIsolate,
  onReset,
}: {
  isolation: KvIsolation;
  models: ModelIsolation[];
  /** PATCH /api/me/key {kv_isolation: "isolated"}; resolves to an error or null. */
  onIsolate: () => Promise<string | null>;
  /** POST /api/me/key/kv-reset; resolves to an error or null. */
  onReset: () => Promise<string | null>;
}) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const isolated = isolation === "isolated";
  const refusing = models.filter((m) => m.cache_isolation === "unsupported").map((m) => m.id);

  async function isolate() {
    setBusy(true);
    setError(null);
    try {
      const err = await onIsolate();
      setError(err);
      if (!err) setAsking(false);
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    setBusy(true);
    setResetError(null);
    try {
      const err = await onReset();
      setResetError(err);
      if (!err) setResetOpen(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className={CARD} aria-label="Prompt cache" data-testid="kv-holder-card">
      <h2 className={CARD_TITLE}>Prompt cache</h2>
      <p className="m-0 text-[13px] text-chat-muted">
        Sharing:{" "}
        <b className="font-semibold text-chat-fg">{isolated ? "isolated, only this key uses its cache" : "shared with other keys"}</b>
      </p>
      <p className="m-0 mt-1 text-[12.5px] leading-[1.45] text-chat-dim">
        {isolated
          ? "Your requests read and write only your own cache. Only an admin can switch this key back to shared."
          : "Your requests reuse cached prompt parts that other keys computed, and theirs reuse yours. Nobody sees anyone's text, but a fast answer can hint that someone sent the same text before."}
      </p>

      {!isolated && !asking && (
        <button
          type="button"
          className={btn("default", "mt-3")}
          onClick={() => {
            setError(null);
            setAsking(true);
          }}
        >
          Keep my cache separate
        </button>
      )}

      {!isolated && asking && (
        <div
          role="group"
          aria-label="Keep this key's cache separate?"
          data-testid="kv-holder-warning"
          className="mt-3 rounded-md border border-chat-accent/40 bg-vw-amber-bg/30 px-3.5 py-3 text-[13px] leading-[1.5] text-vw-band-fg"
        >
          <b className="block font-semibold text-chat-fg">Keep this key&apos;s cache separate?</b>
          <ul className="mb-0 mt-1.5 list-disc space-y-0.5 pl-5">
            <li>
              <b className="font-semibold text-chat-fg">This key&apos;s next turn computes its prompt again.</b> Long sessions get
              slower once, then fast again.
            </li>
            <li>Only this key will read and write its cache, from then on.</li>
            <li>You cannot switch back yourself. Only an admin can.</li>
            {refusing.map((id) => (
              <li key={id}>{id} cannot keep a separate cache, so it will refuse this key.</li>
            ))}
          </ul>
          {error && (
            <p role="alert" className="mb-0 mt-2 text-vw-danger-fg">
              {error}
            </p>
          )}
          <div className="mt-3 flex gap-2">
            <button type="button" className={btn()} onClick={() => setAsking(false)} disabled={busy}>
              Cancel
            </button>
            <button type="button" className={btn("primary")} onClick={() => void isolate()} disabled={busy}>
              Keep it separate
            </button>
          </div>
        </div>
      )}

      <div className="mt-4 border-t border-chat-rule pt-3.5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-[13px] text-chat-muted">Reset my cache</div>
            <p className="m-0 mt-1 text-[12.5px] leading-[1.45] text-chat-dim">
              {isolated
                ? "Drops everything this key has cached. Your next turn computes its prompt again."
                : "Your key shares its cache, so there is nothing of its own to reset."}
            </p>
          </div>
          {isolated && (
            <button
              type="button"
              className={btn("danger", "px-2.5 py-1 text-[12px]")}
              onClick={() => {
                setResetError(null);
                setResetOpen(true);
              }}
            >
              Reset
            </button>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={resetOpen}
        title="Reset your cache?"
        lead="Everything this key has cached is dropped. The key keeps working."
        points={["Your next turn computes its prompt again. On a long session that can take a minute or more."]}
        confirm="Reset cache"
        danger
        busy={busy}
        error={resetError}
        onCancel={() => setResetOpen(false)}
        onConfirm={() => void reset()}
      />
    </section>
  );
}

/** The holder's cache use: the windows, then the key's own store figures (since the store started). */
export function HolderCacheUse({ report }: { report: KvKeyReport }) {
  const s = report.store;
  const since = sinceAgoWords(s?.since);
  const gpuOnly = !s && !windowsHaveStore(report.windows);
  // Read field by field: whatever else a reply carries (who dropped the parts, other keys)
  // is never drawn here.
  const rows: [string, string, string][] = s
    ? [
        ["Your key keeps", fmtGiB(s.ram_bytes), fmtGB(s.nvme_bytes)],
        ["Your share of the store", fmtPct(s.ram_share), fmtPct(s.nvme_share)],
        ["Your parts dropped to make room", fmtInt(s.ram_evictions), fmtInt(s.nvme_evictions)],
      ]
    : [];
  return (
    <CacheUseCard windows={report.windows} holder testId="kv-holder-usage">
      <div className="mt-4 border-t border-chat-rule pt-3">
        {s ? (
          <>
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr className="text-[11px] text-chat-dim">
                  <th className="text-left font-normal">
                    <span className="sr-only">Figure</span>
                  </th>
                  <th className="w-[90px] text-right font-normal">RAM</th>
                  <th className="w-[90px] text-right font-normal">Disk</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([label, ram, disk]) => (
                  <tr key={label}>
                    <td className="py-0.5 text-chat-muted">{label}</td>
                    <td className="py-0.5 text-right tabular-nums">{ram}</td>
                    <td className="py-0.5 text-right tabular-nums">{disk}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {since && (
              <p className="m-0 mt-1.5 text-[12px] text-chat-dim">Store figures cover the time since the KV cache store started, {since}.</p>
            )}
          </>
        ) : gpuOnly ? (
          <Notice tone="info">This server keeps prompts in the GPU cache only.</Notice>
        ) : (
          <Notice tone="bad">The KV cache store is not answering, so its figures are unavailable. Your requests keep working.</Notice>
        )}
      </div>
      <p className="m-0 mt-3 border-t border-chat-rule pt-3 text-[12px] text-chat-dim">
        This shows only your key. It never shows other keys, who caused a drop, or whether anyone else sent the same text.
      </p>
    </CacheUseCard>
  );
}
