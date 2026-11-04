"use client";

// The try-it box: runs the draft settings over pasted text on the server (POST /api/guard/test,
// nothing stored) and shows the text as the model would see it, with masked findings.

import { useId, useState } from "react";

import { sendJSON } from "@/lib/guard/api";
import { G } from "@/lib/guard/strings";
import type { GuardSettings, TestResult } from "@/lib/guard/types";
import { payload } from "@/lib/guard/types";

export function TryIt({ settings }: { settings: GuardSettings }) {
  const id = useId();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    const r = await sendJSON("/api/guard/test", "POST", { text, guard: payload(settings) });
    setBusy(false);
    if (!r.ok) {
      setResult(null);
      setError(r.error);
      return;
    }
    setResult(r.data as TestResult);
  }

  return (
    <section className="space-y-2">
      <h3 className="text-[13px] font-semibold">
        {G.tryIt.title} <span className="font-normal text-chat-dim">· {G.tryIt.hint}</span>
      </h3>
      <label htmlFor={id} className="sr-only">
        {G.tryIt.label}
      </label>
      <textarea
        id={id}
        rows={4}
        spellCheck={false}
        autoComplete="off"
        className="w-full rounded-md border border-chat-rule bg-chat-page text-chat-fg p-2 font-mono text-[12px]"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        type="button"
        className="rounded-md border border-chat-rule px-3 py-1 text-[13px] hover:bg-chat-surface disabled:opacity-45"
        disabled={!text || busy}
        onClick={() => void run()}
      >
        {busy ? G.tryIt.running : G.tryIt.run}
      </button>
      {error && (
        <p role="alert" className="text-[12px] text-vw-danger-fg">
          {error}
        </p>
      )}
      {result && (
        <div className="space-y-1.5">
          <pre className="whitespace-pre-wrap break-all rounded-md border border-chat-rule bg-chat-page text-chat-fg p-2 text-[12px]">{result.preview}</pre>
          <p className="text-[12px] text-chat-dim">
            {result.findings.length ? G.tryIt.matches(result.findings.length, result.ms) : G.tryIt.none}
          </p>
          {result.findings.length > 0 && (
            <ul className="space-y-0.5 text-[12px]">
              {result.findings.map((f, i) => (
                <li key={i}>
                  {f.label} · <span className="font-mono">{f.masked}</span> · {G.mode[f.action === "block" ? "block" : "sanitize"]}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
