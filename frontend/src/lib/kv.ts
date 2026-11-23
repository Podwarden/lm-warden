// Per-key KV cache (plan 2): wire types, number formats and the words the cache
// controls share. Approved mockups: docs/superpowers/mockups/2026-11-21-kv-plan2/.
//
// Two kinds of figures, never mixed: the windows (1 h / 24 h / 7 d) come from the
// key's request history and survive a store outage; the store figures (bytes kept,
// shares, drops) come from the store itself and cover the time since it started.

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import type { components } from "@/lib/api-types.generated";

type S = components["schemas"];
export type KvWindows = S["KvWindows"];
export type KvWindow = S["KvWindow"];
export type KvKeyStore = S["KvKeyStore"];
export type KvKeyStoreAdmin = S["KvKeyStoreAdmin"];
export type KvKeyReport = S["KvKeyReport"];
export type KvUsage = S["KvUsage"];
export type KvModelUsage = S["KvModelUsage"];
export type KvTenantUsage = S["KvTenantUsage"];
export type TokenCacheOut = S["TokenCacheOut"];
export type KvIsolation = "sharing" | "isolated";
export type CacheIsolation = "isolated" | "unsupported";
export type KvStoreMode = "off" | "ram" | "ram+nvme";
export type EvictPolicy = "lru" | "fair";

export const WINDOW_KEYS = ["1h", "24h", "7d"] as const;
export type WindowKey = (typeof WINDOW_KEYS)[number];
export const WINDOW_LABEL: Record<WindowKey, string> = { "1h": "1 h", "24h": "24 h", "7d": "7 d" };

/** One model a key may use, with whether its engine can keep the key's cache apart. */
export interface ModelIsolation {
  id: string;
  cache_isolation: CacheIsolation;
}

const GIB = 1024 ** 3;

/** Prompt tokens the mockup way: 402k, 7.0M, 0. */
export function fmtTokens(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(Math.round(n));
}

/** RAM in GiB, one decimal: 3.4 GiB. */
export function fmtGiB(bytes: number): string {
  return `${(bytes / GIB).toFixed(1)} GiB`;
}

/** Disk in decimal GB, whole: 52 GB. */
export function fmtGB(bytes: number): string {
  const gb = bytes / 1e9;
  return `${gb >= 10 ? Math.round(gb) : gb.toFixed(1)} GB`;
}

/** A 0..1 fraction as "18 %"; null as "—". */
export function fmtPct(f: number | null | undefined): string {
  return f == null ? "—" : `${Math.round(f * 100)} %`;
}

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

/** How long ago the store started: "3 d ago", "5 h ago", "12 min ago". */
export function sinceAgo(since: string | null | undefined, nowMs = Date.now()): string | null {
  if (!since) return null;
  const t = Date.parse(since);
  if (Number.isNaN(t)) return null;
  const s = Math.max(0, (nowMs - t) / 1000);
  if (s >= 86_400) return `${Math.floor(s / 86_400)} d ago`;
  if (s >= 3600) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.max(1, Math.floor(s / 60))} min ago`;
}

/** The same in words, for the holder: "3 days ago". */
export function sinceAgoWords(since: string | null | undefined, nowMs = Date.now()): string | null {
  const short = sinceAgo(since, nowMs);
  if (!short) return null;
  const [n, unit] = short.split(" ");
  const word = unit === "d" ? "day" : unit === "h" ? "hour" : "minute";
  return `${n} ${word}${n === "1" ? "" : "s"} ago`;
}

/** A use-against-share bar: fill and tick as fractions of the bar, over when past the share. */
export function shareBar(used: number, share: number): { fill: number; tick: number | null; over: boolean } {
  const scale = Math.max(used, share);
  if (scale <= 0) return { fill: 0, tick: null, over: false };
  return { fill: used / scale, tick: share > 0 ? share / scale : null, over: share > 0 && used > share };
}

/** Whether the store reported on any window (null RAM and disk everywhere: GPU cache only). */
export function windowsHaveStore(w: KvWindows): boolean {
  return WINDOW_KEYS.some((k) => w[k].ram !== null || w[k].nvme !== null);
}

/** The eviction policy in effect (runtime setting ``kv_evict_policy``, default oldest first). */
export function useEvictPolicy(): EvictPolicy {
  const { data } = useSWR<Record<string, string | null>>("/api/settings/runtime", authFetchJSON, {
    revalidateOnFocus: false,
  });
  return data?.kv_evict_policy === "fair" ? "fair" : "lru";
}

export const KV_STORE_LABEL: Record<KvStoreMode, string> = { off: "off", ram: "RAM", "ram+nvme": "RAM + disk" };

/** Shown on a turn, a router rule and the models lists. */
export const MARK = {
  isolated: "isolated",
  refused: "isolated keys refused",
  notAtProvider: "not isolated at the provider",
  notAtProviderTip:
    "Isolated keys may use this rule, but the provider may still cache their prompts. Their requests are sent without cache marks.",
  keptSeparate: "cache kept separate",
  cannotSeparate: "cannot keep this key's cache separate: refused",
  switchedHere: "This key switched to isolated here, so the next turn computed its prompt again.",
} as const;

/** A store tenant as people know it: the web chat and internal requests share one tenant,
 *  a deleted key (a tag no family has) is "removed key". */
export function tenantLabel(tokenId: string, name: string, selfId?: string): string {
  if (selfId && tokenId === selfId) return "this key";
  if (tokenId === "internal") return "web chat (no key)";
  if (tokenId === "unknown") return "removed key";
  return name;
}

/** The loaded models a key may use: all of them when ``allowed`` is null (no restriction). */
export function modelsForKey(models: ModelIsolation[], allowed: string[] | null | undefined): ModelIsolation[] {
  if (allowed == null) return models;
  const set = new Set(allowed);
  return models.filter((m) => set.has(m.id));
}
