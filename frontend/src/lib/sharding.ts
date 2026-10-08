/**
 * Client-side mirror of `app/models/sharding.py` (#112).
 *
 * Sharded weights on the HF Hub use the `<prefix>-NNNNN-of-NNNNN.<ext>` naming
 * convention — both `transformers` (safetensors) and `llama.cpp` (GGUF) follow
 * it. The Add-model modal groups every shard family under a single
 * disclosure-triangle row instead of flooding the file table with N×repeated
 * radio rows that all resolve to the same selection.
 *
 * Single-file weights (single `.safetensors`, single `.gguf`,
 * `pytorch_model.bin`) do NOT match — they pass through as standalone rows.
 *
 * MUST stay aligned with the backend regex: each shard from the *same* family
 * must be picked up by both implementations or fit-preview math gets the
 * wrong aggregate size.
 */

export const SHARD_NAME_RE =
  /^(?<prefix>.+)-(?<idx>\d{5})-of-(?<total>\d{5})\.(?<ext>safetensors|gguf)$/;

export interface ShardMatch {
  prefix: string;
  idx: number;
  total: number;
  ext: "safetensors" | "gguf";
}

/**
 * Returns the shard match for a sharded filename, or null when the filename
 * is a single-file weights / non-weights file.
 */
export function matchShard(filename: string): ShardMatch | null {
  const m = SHARD_NAME_RE.exec(filename);
  if (!m || !m.groups) return null;
  return {
    prefix: m.groups.prefix,
    idx: parseInt(m.groups.idx, 10),
    total: parseInt(m.groups.total, 10),
    ext: m.groups.ext as "safetensors" | "gguf",
  };
}

export interface ShardFamily<T> {
  /** Family key: `<prefix>-of-<total>.<ext>`. Stable across SWR refetches. */
  key: string;
  prefix: string;
  total: number;
  ext: "safetensors" | "gguf";
  /** Members sorted by shard index ascending. */
  members: T[];
  /** First (lowest-idx) member — used as the "representative" the operator picks. */
  representative: T;
  /** Sum of all member sizes — what fit-preview classifies against. */
  aggregateSize: number;
}

/**
 * Group a list of files by their shard family. Returns the families in
 * insertion order (first appearance of a family member determines its
 * position), plus the list of files that aren't sharded (preserved in
 * their original order).
 *
 * The `sizeOf` accessor lets callers pass any object that has a numeric
 * "size" property without us widening the input type to `unknown`.
 */
export function groupShardFamilies<T extends { filename: string }>(
  files: T[],
  sizeOf: (f: T) => number = (f) => (f as { size?: number }).size ?? 0,
): { families: ShardFamily<T>[]; loose: T[] } {
  const familiesByKey = new Map<string, ShardFamily<T>>();
  const loose: T[] = [];
  const insertionOrder: string[] = [];

  for (const f of files) {
    const match = matchShard(f.filename);
    if (!match) {
      loose.push(f);
      continue;
    }
    const key = `${match.prefix}-of-${match.total}.${match.ext}`;
    let fam = familiesByKey.get(key);
    if (!fam) {
      fam = {
        key,
        prefix: match.prefix,
        total: match.total,
        ext: match.ext,
        members: [],
        representative: f,
        aggregateSize: 0,
      };
      familiesByKey.set(key, fam);
      insertionOrder.push(key);
    }
    fam.members.push(f);
    fam.aggregateSize += sizeOf(f);
  }

  // Sort members by shard idx so disclosure shows them in `00001…00004` order.
  for (const fam of familiesByKey.values()) {
    fam.members.sort((a, b) => {
      const ai = matchShard(a.filename)?.idx ?? 0;
      const bi = matchShard(b.filename)?.idx ?? 0;
      return ai - bi;
    });
    // The first-idx shard is the canonical "representative" — fit-preview
    // results are keyed by that filename server-side, so picking it here
    // matches the existing per-file lookup pattern.
    fam.representative = fam.members[0];
  }

  return {
    families: insertionOrder.map((k) => familiesByKey.get(k)!),
    loose,
  };
}
