// Display formatters for the token details page, ported from the approved
// mockup's script (docs/superpowers/specs/2026-09-18-token-details-mockup.html,
// "formatting" block). The mockup formats in UTC because its data is
// synthetic; these format in the browser's LOCAL time, which is what the
// custom From/To row promises ("Your local time"). All inputs are epoch SECONDS.

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

export const pad2 = (n: number): string => String(n).padStart(2, "0");

const date = (sec: number) => new Date(sec * 1000);

/** SQLite emits naive UTC ("2026-09-06 13:59:00"); `new Date()` would read it
 *  as local time on some engines, so append "Z" (same rule as token-row.tsx). */
export function parseSqliteUtc(value: string | null | undefined): number | null {
  if (!value) return null;
  const iso = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

export function hhmm(sec: number): string {
  const d = date(sec);
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

export function day(sec: number): string {
  const d = date(sec);
  return `${MON[d.getMonth()]} ${d.getDate()}`;
}

export function fmtDateTime(sec: number, nowSec: number = Date.now() / 1000): string {
  const y = date(sec).getFullYear();
  const sameYear = y === date(nowSec).getFullYear();
  return `${day(sec)}${sameYear ? "" : ` ${y}`}, ${hhmm(sec)}`;
}

/** Mockup: span <= 1440 ? hhmm : span <= 3*1440 ? day + " " + hhmm : day. */
export function fmtWhen(sec: number, spanMinutes: number): string {
  if (spanMinutes <= 1440) return hhmm(sec);
  if (spanMinutes <= 3 * 1440) return `${day(sec)} ${hhmm(sec)}`;
  return day(sec);
}

export function compact(v: number): string {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}k`;
  if (v >= 10) return `${Math.round(v)}`;
  if (v >= 1) return v.toFixed(1);
  if (v === 0) return "0";
  return v.toFixed(2);
}

export function secs(v: number | null): string {
  if (v == null) return "–";
  if (v < 1) return `${Math.round(v * 1000)} ms`;
  if (v < 60) return `${v.toFixed(1)} s`;
  return `${(v / 60).toFixed(1)} min`;
}

export function relativeAgo(sec: number | null, nowSec: number): string {
  if (sec == null) return "Never";
  const d = Math.max(0, nowSec - sec);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  const days = Math.floor(d / 86400);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

/** `<input type="datetime-local" step="60">` value in local time. */
export function toInputValue(sec: number): string {
  const d = date(sec);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

export function fromInputValue(v: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(v);
  if (!m) return null;
  const [, Y, M, D, h, mi] = m.map(Number);
  return Math.floor(new Date(Y, M - 1, D, h, mi).getTime() / 1000);
}
