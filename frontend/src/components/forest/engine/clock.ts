/**
 * Spec §10.6 — the forest is a live view only: it shows history LOOK s behind the server's now, advancing at 1×.
 * `ForestClock` is the only owner of "what time is shown". Plain numbers; no three.js.
 */

/** The render delay: the scene is shown LOOK s late, so every growth is known LOOK s before it is shown and the
 * camera can set off toward it (README §4.5). */
export const LOOK = 30;
/** Behind the bound by more than this (s), the clock jumps instead of catching up. */
export const CATCH_UP_MAX = 10;

/** One advance of `rel`. `continuous: false` (a resume after a hidden tab, or a large catch-up) tells the scene to
 * rebuild at `to` instead of animating every turn between `from` and `to`. */
export interface ClockStep {
  from: number;
  to: number;
  continuous: boolean;
}

export class ForestClock {
  /** History seconds relative to the window's t0; always ≤ nowRel − LOOK. */
  rel: number;
  lastStep: ClockStep;
  private now: number;

  constructor(nowRel: number) {
    this.now = nowRel;
    this.rel = nowRel - LOOK;
    this.lastStep = { from: this.rel, to: this.rel, continuous: true };
  }

  /** The server's now (relative to t0). The shell calls it every frame (server now + elapsed wall time). */
  setNow(nowRel: number): void {
    this.now = nowRel;
  }

  get nowRel(): number {
    return this.now;
  }

  /** The newest time that may be shown. */
  get bound(): number {
    return this.now - LOOK;
  }

  /**
   * Advance by `dt` real seconds at 1×, never past the bound and never backward (if now moves back, time holds).
   * A small lag behind the bound is made up at up to 2×; more than CATCH_UP_MAX s is one discontinuous jump.
   */
  tick(dt: number): void {
    const from = this.rel, b = this.bound;
    const lag = b - (from + dt);
    if (lag > CATCH_UP_MAX) return this.jump(b);
    const to = lag > 0 ? from + dt + Math.min(dt, lag) : Math.max(from, Math.min(from + dt, b));
    this.rel = to;
    this.lastStep = { from, to, continuous: true };
  }

  /** The tab was hidden: jump to the bound (nowRel − LOOK) and do not replay the gap (never backward). */
  resumeAfterHidden(nowRel: number): void {
    this.now = nowRel;
    this.jump(Math.max(this.rel, this.bound));
  }

  private jump(to: number): void {
    this.lastStep = { from: this.rel, to, continuous: false };
    this.rel = to;
  }
}

/**
 * The skew estimate (final review C1). The shown time runs on the browser's own monotonic clock; the server's `now` is
 * used only to estimate `skew = serverNow − localNow`. A response's `now` can be stale or late by a few seconds (the
 * server's cache, the network), so the estimate is the median of the last SAMPLES samples, and the applied skew follows
 * it at most MAX_RATE s per s: the estimated server time always advances at 1 ± MAX_RATE×. The one exception is a
 * difference over STEP_S, applied as one discontinuous step (also the first sample).
 */
export const SKEW = { SAMPLES: 5, MAX_RATE: 0.05, STEP_S: 10 } as const;

export class SkewEstimator {
  private samples: number[] = [];
  private skew: number | null = null;
  private target = 0;
  private lastMs: number | null = null;

  /** Moves the applied skew toward the target for the local time elapsed since the last call. */
  private advance(localMs: number): void {
    if (this.lastMs !== null && this.skew !== null && localMs > this.lastMs) {
      const max = (SKEW.MAX_RATE * (localMs - this.lastMs)) / 1000;
      this.skew += Math.max(-max, Math.min(max, this.target - this.skew));
    }
    if (this.lastMs === null || localMs > this.lastMs) this.lastMs = localMs;
  }

  /** A server time `serverNow` (s) received at local monotonic time `localMs`. True when this was a step. */
  sample(serverNow: number, localMs: number): boolean {
    this.advance(localMs);
    this.samples.push(serverNow - localMs / 1000);
    if (this.samples.length > SKEW.SAMPLES) this.samples.shift();
    const v = [...this.samples].sort((a, b) => a - b), m = v.length >> 1;
    this.target = v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
    if (this.skew !== null && Math.abs(this.target - this.skew) <= SKEW.STEP_S) return false;
    this.skew = this.target;
    return true;
  }

  /** The estimated server time at local monotonic time `localMs` (0 before the first sample). */
  now(localMs: number): number {
    this.advance(localMs);
    return this.skew === null ? 0 : localMs / 1000 + this.skew;
  }
}
