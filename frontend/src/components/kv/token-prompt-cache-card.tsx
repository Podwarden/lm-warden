"use client";

// The admin's "Prompt cache" card on the token details page: sharing (both ways),
// the per-tier caps, Save, and Reset. The warning shows as soon as the choice
// differs from the saved one, before Save; isolated -> shared also asks in a dialog
// (owner, 2026-11-22). Caps only act under the fair eviction policy, so under
// oldest first they are disabled and say why. Reset (POST .../kv-reset) always asks
// first and is off for a shared key, which has no cache of its own (the API's 409).

import { useId, useRef, useState } from "react";
import { btn, CARD, CARD_TITLE } from "@/components/tokens/detail/styles";
import { cn } from "@/lib/utils";
import type { EvictPolicy, KvIsolation, ModelIsolation } from "@/lib/kv";
import { ConfirmDialog, Mark, ModelIsolationList } from "./parts";

export interface KvPatch {
  kv_isolation?: KvIsolation;
  kv_max_pct_ram?: number | null;
  kv_max_pct_nvme?: number | null;
}

export interface TokenPromptCacheCardProps {
  token: { id: string; kv_isolation: KvIsolation; kv_max_pct_ram?: number | null; kv_max_pct_nvme?: number | null };
  policy: EvictPolicy;
  /** The loaded models the key may use; listed with their marks while the key is isolated. */
  models: ModelIsolation[];
  onSave: (patch: KvPatch) => Promise<string | null>;
  onReset: () => Promise<string | null>;
}

const CHOICES = ["sharing", "isolated"] as const;

const capText = (v: number | null | undefined) => (v == null ? "" : String(v));

/** "" -> null (no cap); 1..100 -> the number; anything else -> undefined (invalid). */
function parseCap(s: string): number | null | undefined {
  const t = s.trim();
  if (t === "") return null;
  if (!/^\d+$/.test(t)) return undefined;
  const n = Number(t);
  return n >= 1 && n <= 100 ? n : undefined;
}

export function TokenPromptCacheCard({ token, policy, models, onSave, onReset }: TokenPromptCacheCardProps) {
  const saved = token.kv_isolation;
  const [choice, setChoice] = useState<KvIsolation>(saved);
  const [ram, setRam] = useState(capText(token.kv_max_pct_ram));
  const [nvme, setNvme] = useState(capText(token.kv_max_pct_nvme));
  const [seen, setSeen] = useState(token);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState<"none" | "to-shared" | "reset">("none");
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const labelId = useId();
  const radios = useRef<(HTMLButtonElement | null)[]>([]);
  function pick(v: KvIsolation) {
    setChoice(v);
    setError(null);
  }

  // A poll with new saved values resets a clean form; an edit in progress is kept.
  if (token !== seen) {
    if (choice === seen.kv_isolation) setChoice(token.kv_isolation);
    if (ram === capText(seen.kv_max_pct_ram)) setRam(capText(token.kv_max_pct_ram));
    if (nvme === capText(seen.kv_max_pct_nvme)) setNvme(capText(token.kv_max_pct_nvme));
    setSeen(token);
  }

  const capsOff = policy !== "fair";
  const ramCap = parseCap(ram);
  const nvmeCap = parseCap(nvme);
  const capsInvalid = ramCap === undefined || nvmeCap === undefined;
  const patch: KvPatch = {};
  if (choice !== saved) patch.kv_isolation = choice;
  if (ramCap !== undefined && ramCap !== (token.kv_max_pct_ram ?? null)) patch.kv_max_pct_ram = ramCap;
  if (nvmeCap !== undefined && nvmeCap !== (token.kv_max_pct_nvme ?? null)) patch.kv_max_pct_nvme = nvmeCap;
  const dirty = Object.keys(patch).length > 0 || (capsInvalid && !capsOff);
  const switching = choice !== saved;

  async function send() {
    setSaving(true);
    setError(null);
    try {
      const err = await onSave(patch);
      setError(err);
      if (!err) setDialog("none");
    } finally {
      setSaving(false);
    }
  }

  function save() {
    if (capsInvalid) {
      setError("Caps are whole numbers from 1 to 100, or empty for no cap.");
      return;
    }
    if (patch.kv_isolation === "sharing") setDialog("to-shared");
    else void send();
  }

  async function reset() {
    setResetting(true);
    setResetError(null);
    try {
      const err = await onReset();
      setResetError(err);
      if (!err) setDialog("none");
    } finally {
      setResetting(false);
    }
  }

  const isolated = saved === "isolated";

  return (
    <section className={CARD} aria-label="Prompt cache" data-testid="token-kv-card">
      <h2 className={cn(CARD_TITLE, "flex items-center gap-2")}>
        Prompt cache
        <Mark tone={isolated ? "info" : "idle"}>{isolated ? "Isolated" : "Shared"}</Mark>
      </h2>

      <div className="mb-1.5 text-[13px] text-chat-muted" id={labelId}>
        Sharing
      </div>
      {/* The radiogroup pattern: one tab stop (the checked radio), arrow keys move and select. */}
      <div role="radiogroup" aria-labelledby={labelId} className="grid grid-cols-2 overflow-hidden rounded-md border border-chat-rule">
        {CHOICES.map((v, i) => (
          <button
            key={v}
            ref={(el) => {
              radios.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={choice === v}
            tabIndex={choice === v ? 0 : -1}
            onClick={() => pick(v)}
            onKeyDown={(e) => {
              const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
              if (!step) return;
              e.preventDefault();
              const j = (i + step + CHOICES.length) % CHOICES.length;
              pick(CHOICES[j]);
              radios.current[j]?.focus();
            }}
            className={cn(
              "cursor-pointer border-0 py-1.5 text-[13px]",
              choice === v ? "bg-chat-surface-2 font-medium text-chat-accent" : "bg-transparent text-chat-muted hover:text-chat-fg",
            )}
          >
            {v === "sharing" ? "Shared" : "Isolated"}
          </button>
        ))}
      </div>
      <p className="m-0 mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
        Shared: this key reuses cached prompt parts that other shared keys computed, and they reuse its parts. Isolated:
        this key reads and writes only its own cache, on the GPU, in RAM and on disk. That costs this key more compute and
        never slows other keys.
      </p>

      {switching && (
        <div role="status" data-testid="kv-switch-warning" className="mt-3 rounded-md border border-chat-accent/40 bg-vw-amber-bg/30 px-3 py-2.5 text-[12.5px] leading-[1.45] text-vw-band-fg">
          <b className="font-semibold text-chat-fg">Before you save: this key&apos;s next turn computes its prompt again.</b>
          <ul className="mb-0 mt-1 list-disc space-y-0.5 pl-4">
            {choice === "isolated" ? (
              <>
                <li>What it cached while shared stays shared and ages out as usual.</li>
                <li>The holder can isolate it themselves later, but only an admin can switch it back.</li>
              </>
            ) : (
              <>
                <li>Its own cached parts in RAM and on disk are dropped first.</li>
                <li>From then on it shares with every other shared key.</li>
              </>
            )}
          </ul>
        </div>
      )}

      {isolated && models.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[12px] text-chat-dim">Models this key may use</div>
          <ModelIsolationList models={models} isolated />
        </div>
      )}

      <div className="mt-3.5 space-y-2.5">
        {(
          [
            ["kv-cap-ram", "Most of the RAM tier", ram, setRam],
            ["kv-cap-nvme", "Most of the disk tier", nvme, setNvme],
          ] as const
        ).map(([id, label, value, set]) => (
          <div key={id} className="flex items-center justify-between gap-3">
            <label htmlFor={id} className="text-[13px] text-chat-muted">
              {label}
            </label>
            <span className="inline-flex items-center gap-1.5">
              <input
                id={id}
                inputMode="numeric"
                placeholder="no cap"
                value={value}
                disabled={capsOff}
                onChange={(e) => {
                  set(e.target.value);
                  setError(null);
                }}
                className="w-[86px] rounded-md border border-chat-rule bg-chat-page px-2 py-1 text-right text-[13px] tabular-nums text-chat-fg disabled:opacity-45"
              />
              <span className="text-[12px] text-chat-dim">%</span>
            </span>
          </div>
        ))}
      </div>
      <p className="m-0 mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
        A hard limit, 1 to 100 % of each tier. Empty means no cap. A key at its cap drops its own oldest parts first.
        {capsOff && (
          <>
            <br />
            <b className="font-semibold text-chat-muted">Has no effect now:</b> caps work only with the fair eviction policy
            (Settings, Prompt cache).
          </>
        )}
      </p>

      <div className="mt-4 flex items-center gap-2">
        <button type="button" aria-label="Save changes to the prompt cache" className={btn("primary")} disabled={!dirty || saving} onClick={save}>
          Save changes
        </button>
        {error ? (
          <span role="alert" className="text-[12px] text-vw-danger-fg">
            {error}
          </span>
        ) : (
          dirty && <span className="text-[12px] text-chat-accent">Unsaved</span>
        )}
      </div>

      <div className="mt-[18px] border-t border-chat-rule pt-3.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] text-chat-muted">Reset this key&apos;s cache</span>
          <button
            type="button"
            aria-label="Reset this key's cache"
            className={btn("danger", "px-2.5 py-1 text-[12px]")}
            disabled={!isolated}
            onClick={() => {
              setResetError(null);
              setDialog("reset");
            }}
          >
            Reset
          </button>
        </div>
        <p className="m-0 mt-1.5 text-[12px] leading-[1.45] text-chat-dim">
          {isolated
            ? "Drops this key's own cached prompts everywhere. Its next turn computes its prompt again. Use it when a holder wants a clean start."
            : "Only an isolated key has a cache of its own to reset."}
        </p>
      </div>

      <ConfirmDialog
        open={dialog === "reset"}
        title="Reset this key's cache?"
        lead="Drops every cached prompt this key owns, on the GPU, in RAM and on disk. The key keeps working."
        points={["Its next turn computes its prompt again. On a long session that can take a minute or more."]}
        confirm="Reset cache"
        danger
        busy={resetting}
        error={resetError}
        onCancel={() => setDialog("none")}
        onConfirm={() => void reset()}
      />
      <ConfirmDialog
        open={dialog === "to-shared"}
        title="Switch this key back to shared?"
        lead="Its next turn computes its prompt again."
        points={[
          "Its own cached parts in RAM and on disk are dropped first.",
          "From then on it shares with every other shared key, and a fast answer can hint that someone sent the same text before.",
        ]}
        confirm="Switch to shared"
        busy={saving}
        error={error}
        onCancel={() => setDialog("none")}
        onConfirm={() => void send()}
      />
    </section>
  );
}
