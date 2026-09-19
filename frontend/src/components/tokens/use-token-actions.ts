"use client";

// Test / Delete / Pause / Rotate-open, shared by the list row (token-row.tsx)
// and the token details page (spec 2026-09-18 §4.1). The logic and the
// operator-facing strings are the row's, moved here unchanged; Pause is new.

import { useCallback, useRef, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { detailOf } from "@/lib/token-series";

export interface TestResult {
  ok: boolean;
  status: number;
  ms: number;
  detail?: string;
  /** Number of loaded models this token can reach (200 responses only). */
  models?: number;
  /** The key is paused: requests get 403 `token paused` (spec §3.1). */
  paused?: boolean;
}

export interface TokenActions {
  testing: boolean;
  testResult: TestResult | null;
  /** POSTs /test. Resolves `null` without a request while a test is already
   *  running (a second click before the button disables). */
  runTest: () => Promise<TestResult | null>;
  busy: boolean;
  deleteError: string | null;
  /** Confirms with window.confirm, then DELETEs. Resolves `true` on success,
   *  `false` when the DELETE failed, and `null` when nothing was sent (the
   *  confirm was cancelled, or a delete is already in flight) — so callers
   *  refresh only when something may have changed. */
  remove: () => Promise<boolean | null>;
  pausing: boolean;
  pauseError: string | null;
  setPaused: (paused: boolean) => Promise<boolean>;
  rotateOpen: boolean;
  openRotate: () => void;
  closeRotate: () => void;
}

// Controller ruling (2026-09-18): PATCH /api/tokens/{id} returns Pydantic 422s
// whose `detail` is an ARRAY of { msg } objects (e.g. priority out of
// range), as well as plain-string 409 details. `detailOf` (token-series.ts)
// already handles both shapes — reuse it rather than `String(body.detail)`,
// which would render "[object Object]" for the array case.
export async function errorDetail(r: Response, fallback: string): Promise<string> {
  const body = await r.json().catch(() => null);
  return detailOf(body) ?? fallback;
}

export function useTokenActions(
  token: { id: string; name: string },
  onChange: () => void,
): TokenActions {
  const [busy, setBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [pausing, setPausing] = useState(false);
  const [pauseError, setPauseError] = useState<string | null>(null);
  const [rotateOpen, setRotateOpen] = useState(false);
  const path = `/api/tokens/${encodeURIComponent(token.id)}`;
  // Re-entrancy guards. Refs rather than the `busy`/`testing` state, which a
  // second click in the same frame would still read as false.
  const deleting = useRef(false);
  const testingRef = useRef(false);

  const remove = useCallback(async (): Promise<boolean | null> => {
    if (deleting.current) return null;
    // confirm() is fine for a destructive-but-recoverable op like this.
    if (typeof window !== "undefined" && !window.confirm(
      `Delete token "${token.name}"? Any client using it will lose access immediately.`,
    )) {
      return null;
    }
    deleting.current = true;
    setBusy(true);
    setDeleteError(null);
    try {
      const r = await authFetch(path, { method: "DELETE" });
      if (!r.ok) {
        setDeleteError(await errorDetail(r, `Failed to delete token (HTTP ${r.status})`));
        return false;
      }
      return true;
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Network error");
      return false;
    } finally {
      deleting.current = false;
      setBusy(false);
    }
  }, [path, token.name]);

  const runTest = useCallback(async (): Promise<TestResult | null> => {
    if (testingRef.current) return null;
    testingRef.current = true;
    setTesting(true);
    setTestResult(null);
    const t0 = performance.now();
    let result: TestResult;
    try {
      const r = await authFetch(`${path}/test`, { method: "POST" });
      const ms = Math.round(performance.now() - t0);
      if (!r.ok) {
        result = { ok: false, status: r.status, ms, detail: await errorDetail(r, r.statusText) };
      } else {
        const body = (await r.json().catch(() => null)) as
          | { ok: boolean; revoked: boolean; expired: boolean; paused?: boolean; proxy_reachable: boolean; allowed_models: string[] }
          | null;
        if (!body) {
          result = { ok: false, status: r.status, ms, detail: "empty response body" };
        } else {
          const issues: string[] = [];
          if (body.revoked) issues.push("token is revoked");
          if (body.expired) issues.push("token is expired");
          if (body.paused) issues.push("token is paused");
          if (!body.proxy_reachable) issues.push("proxy /healthz unreachable");
          if (body.allowed_models.length === 0) issues.push("no loaded models match this token's scope");
          result = issues.length > 0
            ? { ok: false, status: r.status, ms, detail: issues.join("; "), paused: !!body.paused, models: body.allowed_models.length }
            : { ok: true, status: r.status, ms, detail: `${body.allowed_models.length} model(s) reachable`, models: body.allowed_models.length, paused: false };
        }
      }
    } catch (err) {
      result = {
        ok: false,
        status: 0,
        ms: Math.round(performance.now() - t0),
        detail: err instanceof Error ? err.message : "Network error",
      };
    }
    testingRef.current = false;
    setTestResult(result);
    setTesting(false);
    return result;
  }, [path]);

  const setPaused = useCallback(async (paused: boolean): Promise<boolean> => {
    setPausing(true);
    setPauseError(null);
    try {
      const r = await authFetch(path, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paused }),
      });
      if (!r.ok) {
        setPauseError(await errorDetail(r, `Failed to ${paused ? "pause" : "resume"} token (HTTP ${r.status})`));
        return false;
      }
      onChange();
      return true;
    } catch (err) {
      setPauseError(err instanceof Error ? err.message : "Network error");
      return false;
    } finally {
      setPausing(false);
    }
  }, [path, onChange]);

  const openRotate = useCallback(() => setRotateOpen(true), []);
  const closeRotate = useCallback(() => {
    setRotateOpen(false);
    onChange();
  }, [onChange]);

  return {
    testing, testResult, runTest,
    busy, deleteError, remove,
    pausing, pauseError, setPaused,
    rotateOpen, openRotate, closeRotate,
  };
}
