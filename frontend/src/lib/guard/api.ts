// Calls to the guard's admin API. Errors carry the server's `detail` (a rule the server rejected
// says why), never a request body.

import { authFetch } from "@/lib/auth-fetch";

export async function sendJSON(url: string, method: "PUT" | "PATCH" | "POST", body: unknown): Promise<{ ok: boolean; data: unknown; error: string | null }> {
  try {
    const r = await authFetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data: unknown = null;
    try {
      data = await r.json();
    } catch {
      /* no body */
    }
    if (!r.ok) {
      const d = (data as { detail?: unknown } | null)?.detail;
      return { ok: false, data, error: typeof d === "string" ? d : `HTTP ${r.status}` };
    }
    return { ok: true, data, error: null };
  } catch {
    return { ok: false, data: null, error: "Network error" };
  }
}
