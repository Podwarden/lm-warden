"use client";
// GET /api/system/backends — which backends this build has, and what the
// ACTIVE driver lets each of them do.
//
// Hand-typed for the same reason as system-info.ts and stats-v2.ts: the route
// returns a bare dict with no FastAPI response_model, so openapi-typescript
// collapses it to `unknown` and a drift here has to be caught by the backend
// unit tests plus this file's own component tests instead.
//
// The rendering half of this contract lives in @/lib/backend-fields — this
// module only fetches. Keeping the fetch and the rule apart is what lets the
// rule stay a pure function that a test can call with a code and no server.
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { VersionPinReasonCode } from "@/lib/backend-fields";

export interface BackendCapability {
  name: string;
  display_name: string;
  /** The engine version baked into THIS warden image; null outside a build. */
  version: string | null;
  /**
   * The ENGINE fact: can this engine be version-pinned at all, on any driver?
   *
   * NOT the field to gate a control on — see `version_pin_available`. Both
   * ship because the pair is what lets a client disable a control AND explain
   * it (app/system/routes_engine.py, and the Backend protocol docstring).
   */
  supports_version_pin: boolean;
  /** The DEPLOYMENT fact. Gate controls on THIS one. */
  version_pin_available: boolean;
  /** Non-null exactly when `version_pin_available` is false. Render verbatim. */
  version_pin_reason: string | null;
  /**
   * The same decision, machine-readable. Branch on THIS; render the sentence.
   * Grepping the sentence for "docker" would make a copy-edit a behaviour
   * change.
   */
  version_pin_reason_code: VersionPinReasonCode;
}

export interface SystemBackends {
  default: string;
  driver: string;
  engine_version: string | null;
  backends: BackendCapability[];
}

/**
 * The capability row for one model's backend, or `undefined` while loading.
 *
 * `backend` is the model row's own value; `null`/`""` resolves to the default
 * backend, matching decision D6 (`models.backend` is nullable and NULL means
 * vLLM) and `fieldAppliesTo`'s identical fallback.
 *
 * Returns `undefined` — never a fabricated row — when the fetch has not landed
 * or the build does not know this backend. Callers must treat `undefined` as
 * "not known yet" and NOT as "unavailable": flashing a control off and back on
 * is the reason try-stack-panel.tsx only locks its selector once it KNOWS.
 */
export function useBackendCapability(
  backend: string | null | undefined,
): BackendCapability | undefined {
  const { data } = useSWR<SystemBackends>("/api/system/backends", authFetchJSON);
  // The array check is not defensive typing. This route was added mid-rollout,
  // and a UI pod that outruns its api pod gets a 404 body or an older shape --
  // `data` truthy, `data.backends` undefined. Reading `.find` off that throws
  // inside render, and because this hook is called from the model DETAIL page,
  // the whole page white-screens over a capability lookup it only needed to
  // decide whether to draw one card. Unknown resolves to `undefined`, which
  // every caller already handles as "not known yet".
  if (!data || !Array.isArray(data.backends)) return undefined;
  const effective = backend || data.default || "vllm";
  return data.backends.find((b) => b.name === effective);
}
