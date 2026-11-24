// Typed client for the model-aliases API (`/api/aliases`). Built on the api.ts
// helpers (getJSON / postJSON) and the generated schema types, with the raw
// authFetch used where the generated types lose the response body we need
// (PATCH / DELETE carry the 422 detail for inline field errors).
import { getJSON, postJSON } from "./api";
import { authFetch } from "./auth-fetch";
import type { components } from "./api-types.generated";

export type GeneratedAliasTargetOut = components["schemas"]["AliasTargetOut"];
export type GeneratedAliasIn = components["schemas"]["AliasIn"];
export type GeneratedAliasPatch = components["schemas"]["AliasPatch"];
export type GeneratedAliasOut = components["schemas"]["AliasOut"];

export type AliasParamMode = "default" | "force" | "cap";

/** One entry of the alias params map: a request parameter's value and mode. */
export interface AliasParamValue {
  value: unknown;
  mode: AliasParamMode;
}

export interface AliasOut {
  id: string;
  name: string;
  targets: GeneratedAliasTargetOut[];
  guard: "inherit" | "sanitize" | "block";
  thinking_policy: "inherit" | "client" | "on" | "off";
  params: Record<string, AliasParamValue>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

/** Shape the create/update POST sends (identical for both). */
export interface AliasBody {
  name: string;
  targets: string[];
  guard: string;
  thinking_policy: string;
  params: Record<string, AliasParamValue>;
  enabled: boolean;
}

export class AliasApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, body: unknown) {
    super(`${status} /api/aliases`);
    this.name = "AliasApiError";
    this.status = status;
    this.body = body;
  }
}

const jsonHeaders = { "Content-Type": "application/json" };

async function readError(r: Response): Promise<never> {
  const body = await r.json().catch(() => null);
  throw new AliasApiError(r.status, body);
}

/** POST /api/aliases (create). Throws AliasApiError with the parsed 422 body. */
export async function createAlias(body: AliasBody): Promise<AliasOut> {
  const r = await authFetch("/api/aliases", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  });
  if (!r.ok) return readError(r);
  return (await r.json()) as AliasOut;
}

/** PATCH /api/aliases/{id}. Throws AliasApiError with the parsed 422 body. */
export async function patchAlias(id: string, body: AliasBody): Promise<AliasOut> {
  const r = await authFetch(`/api/aliases/${id}`, {
    method: "PATCH",
    headers: jsonHeaders,
    body: JSON.stringify(body),
  });
  if (!r.ok) return readError(r);
  return (await r.json()) as AliasOut;
}

/** PATCH /api/aliases/{id} with just the enabled flag (toggle switch). */
export async function setAliasEnabled(id: string, enabled: boolean): Promise<AliasOut> {
  const r = await authFetch(`/api/aliases/${id}`, {
    method: "PATCH",
    headers: jsonHeaders,
    body: JSON.stringify({ enabled }),
  });
  if (!r.ok) return readError(r);
  return (await r.json()) as AliasOut;
}

/** DELETE /api/aliases/{id}. Throws AliasApiError on failure. */
export async function deleteAlias(id: string): Promise<void> {
  const r = await authFetch(`/api/aliases/${id}`, { method: "DELETE" });
  if (!r.ok) return readError(r);
}

/** GET /api/aliases via the generated-path getJSON helper. */
export async function listAliases(): Promise<AliasOut[]> {
  const out = await getJSON("/api/aliases");
  return out.aliases as AliasOut[];
}
