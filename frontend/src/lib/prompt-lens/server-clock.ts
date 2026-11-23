// The server's clock as the dock sees it, for ticking figures (elapsed, the
// queue wait) that compare "now" with server timestamps. A client clock that
// is off by minutes would otherwise show minutes of elapsed time.
//
// The skew is estimated from events as they arrive: client receive time minus
// the event's server `ts` is the skew plus the delivery delay, so the smallest
// recent value is the best estimate. Note an event only when it has just
// arrived (a stale one would read as skew). Until anything is noted the skew
// is 0 and the client clock is used as is.

/** Recent estimates kept (a clock that is corrected moves on after these). */
const KEEP = 64;

let recent: number[] = [];

/** An event with server time `ts` just arrived (client time `at`, seconds). */
export function noteServerTs(ts: number, at: number = Date.now() / 1000): void {
  if (!Number.isFinite(ts) || ts <= 0) return;
  recent.push(at - ts);
  if (recent.length > KEEP) recent = recent.slice(recent.length - KEEP);
}

/** Client minus server, seconds (0 before any event). */
export function serverSkew(): number {
  return recent.length ? Math.min(...recent) : 0;
}

/** The server's "now", seconds since the epoch. */
export function serverNow(): number {
  return Date.now() / 1000 - serverSkew();
}

/** Tests only. */
export function resetServerClock(): void {
  recent = [];
}
