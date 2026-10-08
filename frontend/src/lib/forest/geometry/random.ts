/**
 * Ports the mockup's `seed` / `R` / `SEED` (forest-real.html l.63-65). The mockup kept `seed` in a global; here each
 * tree build owns one `Rng`, so a worker can build many trees without them sharing state.
 */
export class Rng {
  /** Mockup `let seed=7`. */
  seed = 7;

  /** Mockup `R`: an LCG, `seed = (seed·1664525 + 1013904223) mod 2^32`, returned in [0, 1). */
  R(): number {
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  /**
   * Mockup `SEED(...k)`: FNV-1a over `k.join('|')` (one step per code point, `charCodeAt(0)` as the mockup does), then
   * two draws. Every random choice belongs to one session/turn and never shifts when something else grows.
   */
  SEED(...k: (string | number)[]): void {
    let h = 2166136261;
    for (const c of k.join("|")) {
      h ^= c.charCodeAt(0);
      h = Math.imul(h, 16777619) >>> 0;
    }
    this.seed = h;
    this.R();
    this.R();
  }
}

/** FNV-1a of a string as an unsigned 32-bit integer (same hash as `SEED`, without the draws). */
export function fnv(s: string): number {
  let h = 2166136261;
  for (const c of s) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}
