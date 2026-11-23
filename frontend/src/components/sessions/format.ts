import { fmtApproxText, S } from "@/lib/sessions/strings";
import type { Turn } from "@/lib/sessions/types";

export function pct(n: number | null | undefined): number | null {
  return n === null || n === undefined || !Number.isFinite(n) ? null : Math.round(n * 100);
}

/** "62% · rewrite", "99%", relayed "upstream 99%", "—" when unknown. */
export function fmtShareCause(t: Turn): string {
  const share = t.prompt_tokens > 0 && t.cached_tokens !== null ? t.cached_tokens / t.prompt_tokens : null;
  const p = pct(share);
  if (p === null) return "—";
  if (t.relayed) return `${S.turn.upstream} ${p}%`;
  // A key holder gets the own lens only (the fleet lens is admin-only).
  const dc = t.divergence_cause ?? t.divergence_cause_own;
  const cause = dc ? S.turn.causeShort[dc] ?? dc : null;
  return cause ? `${p}% · ${cause}` : `${p}%`;
}

export function fmtSecs(s: number | null): string {
  if (s === null) return "—";
  return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
}

/** "42 s", "42 min", "1 h 17 min", "2 h 02 min", "412 h" (a day or more: hours only). */
export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  if (s < 60) return `${Math.max(0, Math.round(s))} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s >= 86_400) return `${Math.round(s / 3600).toLocaleString()} h`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s - h * 3600) / 60);
  return m === 60 ? `${h + 1} h 00 min` : `${h} h ${String(m).padStart(2, "0")} min`;
}

/** "≈ 40 s", "≈ 19 min", "≈ 31 h"; "—" when unknown. */
export function fmtApprox(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  return fmtApproxText(s);
}

/** "Mon 08:40" within the last 6 days, else "Oct 12 08:40" (local time). */
export function fmtStart(t: number, now: number = Date.now() / 1000): string {
  const d = new Date(t * 1000);
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (now - t < 6 * 86_400) return `${d.toLocaleDateString("en-US", { weekday: "short" })} ${hm}`;
  return `${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })} ${hm}`;
}

/** "870", "48 M", "9.1 B": prompt token totals run into billions. */
export function fmtTokens(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  const one = (v: number) => (v >= 100 ? Math.round(v).toString() : v.toFixed(1).replace(/\.0$/, ""));
  if (a >= 1e9) return `${one(n / 1e9)} B`;
  if (a >= 1e6) return `${one(n / 1e6)} M`;
  if (a >= 1e3) return `${one(n / 1e3)} k`;
  return Math.round(n).toString();
}
