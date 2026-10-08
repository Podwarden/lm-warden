"use client";

// Verify (plan §5.2, §5.4): a real request from this tab, over the path the
// tool will use, with a plaintext the operator holds — created here or pasted
// into a masked field. Plain `fetch`, never authFetch: the admin JWT must not
// ride on a /v1 call, and cookies are omitted. The result is reduced to
// status, latency, the answering model and the first words of the answer;
// for the Anthropic protocol with routing on, the route comes from the
// router's decision log (the truth, not the answer text).

import { useEffect, useId, useRef, useState } from "react";
import { authFetchJSON } from "@/lib/auth-fetch";
import { cn } from "@/lib/utils";
import {
  buildVerifyRequest,
  describeRoute,
  explainVerify,
  renderTemplate,
  verifyBlocker,
  type ConnectMode,
  type ConnectProtocol,
  type ConnectRouter,
  type ConnectVerify,
  type TemplateVars,
  type VerifyExplanation,
} from "@/lib/connect";
import type { DecisionOut } from "@/lib/router";
import { FIELD, FIELD_LABEL, META, btn, FOCUS } from "./styles";

interface Props {
  verify: ConnectVerify;
  mode: ConnectMode;
  protocol: ConnectProtocol;
  router: ConnectRouter | null;
  /** Everything but the key: origin, model, header, context, verify_model. */
  vars: TemplateVars;
  /** A key created on this page; when set, no paste field. */
  createdKey: string | null;
  pasted: string;
  onPaste: (v: string) => void;
  /** Clears the result when the tool, mode, model or key changes. */
  resetKey: string;
}

async function readBody(r: Response): Promise<unknown> {
  const text = await r.text().catch(() => "");
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function VerifyRow({ verify, mode, protocol, router, vars, createdKey, pasted, onPaste, resetKey }: Props) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<VerifyExplanation | null>(null);
  const inputId = useId();
  const hintId = useId();
  const reasonId = useId();
  const run = useRef(0);

  const key = createdKey ?? (pasted.trim() || null);
  const blocker = verifyBlocker(mode, router, !!key);

  useEffect(() => {
    run.current += 1;
    setResult(null);
    setBusy(false);
  }, [resetKey, key]);

  async function send() {
    if (!key || blocker) return;
    const mine = ++run.current;
    setBusy(true);
    setResult(null);
    const allVars = { ...vars, key };
    const req = buildVerifyRequest(verify, allVars);
    const model = typeof allVars.verify_model === "string" ? allVars.verify_model : (allVars.model ?? undefined);
    const t0 = performance.now();
    let out: VerifyExplanation;
    try {
      const r = await fetch(req.url, req.init);
      const body = await readBody(r);
      req.done();
      out = explainVerify(r.status, body, protocol, {
        latencyMs: performance.now() - t0,
        model: model ?? undefined,
        key,
        expect: verify.expect,
      });
      if (r.ok && protocol === "anthropic" && router?.enabled) {
        try {
          const d = await authFetchJSON<{ decisions: DecisionOut[] }>("/api/router/decisions?limit=1");
          const latest = d.decisions[0];
          const route = latest && latest.model_in === model ? describeRoute(latest) : null;
          if (route) out = { ...out, text: `${out.text} · ${route}` };
        } catch {
          // The answer stands on its own; the route is a bonus.
        }
      }
    } catch (err) {
      req.done();
      const abort = err instanceof DOMException && err.name === "AbortError";
      out = explainVerify(abort ? "abort" : "network", null, protocol, { origin: window.location.origin, key });
    }
    if (mine !== run.current) return;
    setResult(out);
    setBusy(false);
  }

  const covers = renderTemplate(verify.covers, vars);
  const notCovered = verify.not_covered ? renderTemplate(verify.not_covered, vars) : null;

  return (
    <section aria-labelledby="connect-verify-h" className="mt-5 border-t border-vw-rule-soft/45 pt-4">
      <h3 id="connect-verify-h" className="m-0 text-[15px] font-semibold">
        Test it from here
      </h3>
      <p className={cn(META, "m-0 mt-1 max-w-[78ch]")}>
        Sends <code className="!font-mono text-[12px]">{`${verify.method} ${verify.path}`}</code> from this tab. Tests: {covers}
      </p>
      {notCovered && <p className={cn(META, "m-0 mt-0.5 max-w-[78ch]")}>Not tested: {notCovered}</p>}

      <div className="mt-3 flex flex-wrap items-end gap-2.5 max-[759px]:flex-col max-[759px]:items-stretch">
        {createdKey ? (
          <p className="m-0 self-center text-[13px] text-chat-muted">Uses the key you just created.</p>
        ) : (
          <div className="grid min-w-0 gap-1.5 min-[760px]:w-[300px]">
            <label htmlFor={inputId} className={FIELD_LABEL}>
              Paste the key to test it
            </label>
            <input
              id={inputId}
              type="password"
              autoComplete="off"
              spellCheck={false}
              aria-describedby={hintId}
              placeholder="vw_…"
              value={pasted}
              onChange={(e) => onPaste(e.target.value)}
              className={FIELD}
            />
          </div>
        )}
        <button
          type="button"
          className={btn("default", `h-9 max-[759px]:h-10 ${FOCUS}`)}
          disabled={!!blocker || busy}
          aria-busy={busy}
          aria-describedby={blocker ? reasonId : undefined}
          onClick={() => void send()}
        >
          {busy ? "Sending…" : "Send a test request"}
        </button>
      </div>
      {blocker && (
        <p id={reasonId} className={cn(META, "m-0 mt-1.5")}>
          {blocker}
        </p>
      )}
      {!createdKey && (
        <p id={hintId} className={cn(META, "m-0 mt-1.5")}>
          A pasted key is kept in this tab only and sent nowhere but this test.
        </p>
      )}

      <div
        role="status"
        data-testid="connect-verify-status"
        className={cn(result?.tone === "ok" && "mt-2.5 text-[13.5px] font-medium text-vw-ok-fg [overflow-wrap:anywhere]")}
      >
        {result?.tone === "ok" ? result.text : null}
      </div>
      <div
        role="alert"
        data-testid="connect-verify-alert"
        className={cn(result?.tone === "danger" && "mt-2.5 text-[13.5px] font-medium text-vw-danger-fg [overflow-wrap:anywhere]")}
      >
        {result?.tone === "danger" ? result.text : null}
      </div>
    </section>
  );
}
