// Wire types for the guard's admin API (app/guard/routes_api.py, app/tokens/routes_api.py).

export type GuardMode = "sanitize" | "block";

export interface CustomRule {
  id: string;
  type: "regex" | "fixed";
  regex?: string;
  values?: string[];
  keywords?: string[];
  label?: string;
}

export interface GuardSettings {
  mode: GuardMode;
  packs: string[];
  rules: CustomRule[];
  overrides: Record<string, string>;
  restore: "all" | "text_only" | "off";
  trailer: "new_only" | "always" | "off";
  fail: "closed" | "open";
}

/** A key's stored choice: null = use the platform setting. */
export type KeyGuard = null | { off: true } | { override: Partial<GuardSettings> };

export interface PackInfo {
  id: string;
  label: string;
  default: boolean;
  rules: { id: string; label: string; public_prefix: string | null }[];
}

export interface PacksResponse {
  default: string[];
  packs: PackInfo[];
}

export interface GuardEventRow {
  ts: number;
  request_id: string | null;
  sess_ref: string | null;
  action: string;
  rule_id: string | null;
  role: string | null;
  msg_index: number | null;
  kind: string | null;
  length: number | null;
  ph: string | null;
  relayed: boolean;
}

export interface GuardEvents {
  events: GuardEventRow[];
  totals: Record<string, number>;
}

export interface TestFinding {
  rule_id: string;
  label: string;
  start: number;
  end: number;
  length: number;
  masked: string;
  action: string;
}

export interface TestResult {
  ms: number;
  preview: string;
  findings: TestFinding[];
}

export function defaults(packs: PacksResponse | undefined): GuardSettings {
  return {
    mode: "sanitize",
    packs: packs && Array.isArray(packs.default) ? [...packs.default] : [],
    rules: [],
    overrides: {},
    restore: "all",
    trailer: "new_only",
    fail: "closed",
  };
}

/** Settings as the API returns them (possibly partial, possibly with `source`) → a full form value. */
export function normalise(raw: Partial<GuardSettings> & { source?: unknown }, packs: PacksResponse | undefined): GuardSettings {
  const d = defaults(packs);
  return {
    mode: raw.mode === "block" ? "block" : "sanitize",
    packs: Array.isArray(raw.packs) ? [...raw.packs] : d.packs,
    rules: Array.isArray(raw.rules) ? raw.rules.map((r) => ({ ...r })) : [],
    overrides: raw.overrides && typeof raw.overrides === "object" ? { ...raw.overrides } : {},
    restore: raw.restore ?? d.restore,
    trailer: raw.trailer ?? d.trailer,
    fail: raw.fail ?? d.fail,
  };
}

/** What the API is sent: custom rules carry only the fields their type uses. */
export function payload(s: GuardSettings): GuardSettings {
  return {
    ...s,
    rules: s.rules.map((r) =>
      r.type === "regex"
        ? { id: r.id.trim(), type: "regex", regex: r.regex ?? "" }
        : { id: r.id.trim(), type: "fixed", values: (r.values ?? []).map((v) => v.trim()).filter(Boolean) },
    ),
  };
}
