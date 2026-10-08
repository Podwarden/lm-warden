// Seeds the session forest's input, `request_history`, straight into the temp SQLite database of the local stack
// (forest-stack.ts). Nothing goes through a backend endpoint: the forest reads the table, so the table is the fixture.
//
// What is seeded (times relative to the seed's "now"):
//   key A "e2e-alpha"
//     - three spells of agent sessions over the 48 h window (40 h, 20 h and 6 h ago), one with a subagent;
//     - a session from 16 h ago that came back 14 h ago (a tree that has already moved: a ghost at its old spot);
//     - sessions R1-R3, last turns 4-11 h ago: `resumeSession` adds the next turn of one of them during a motion run,
//       > 30 min after the previous one, so the live view flies that tree forward (a transplant);
//     - five flowers (one-shot, no tools), one of them with no completion (an embeddings call: a mushroom);
//     - `startGrowing` grows a fresh batch tree during each motion run: a turn every 2 s, passing 60 requests mid-run,
//       with a second session joining part way through.
//   key B "e2e-beta" (the key holder's): two trees and a flower, so the key-scoped view has its own forest.
// Uses node:sqlite (Node ≥ 22.13), with the app's own pragmas (WAL, busy timeout) so it never fights uvicorn's writer.

import { createHash, randomBytes, randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

export interface SeedKey {
  id: string;
  name: string;
  /** The inference key (vw_…). Only the key holder test uses it, to sign in. */
  plaintext: string;
}

export interface SeedResult {
  keyA: SeedKey;
  keyB: SeedKey;
  /** Session keys of R1-R3, the sessions `resumeSession` may bring back (one per motion run). */
  resumeSessionKeys: string[];
  /** Rows seeded per key (static part only). */
  rows: { a: number; b: number };
}

const MODEL_ID = 'e2e-model-row';
const MODEL = 'e2e-model';
const VARIANT = 'e2e-variant';
const TOOLS = ['read', 'edit', 'shell', 'web', 'agent'] as const;

function open(dbPath: string): DatabaseSync {
  const db = new DatabaseSync(dbPath);
  db.exec('PRAGMA busy_timeout = 30000');
  db.exec('PRAGMA journal_mode = WAL');
  return db;
}

/** A 16-hex session key, the shape the proxy writes (HMAC[:16]). */
const skey = () => randomBytes(8).toString('hex');
const hmac8 = () => randomBytes(4).toString('hex');

/** vw_ + 56 lowercase base32 characters, like app/auth/bearer.py. */
function inferenceKey(): string {
  const abc = 'abcdefghijklmnopqrstuvwxyz234567';
  const b = randomBytes(56);
  return `vw_${[...b].map((x) => abc[x % 32]).join('')}`;
}

function insertKey(db: DatabaseSync, name: string): SeedKey {
  const plaintext = inferenceKey();
  const id = randomBytes(16).toString('hex');
  db.prepare(
    'INSERT INTO api_tokens(id, name, prefix, hash, scope, priority) VALUES (?, ?, ?, ?, ?, ?)',
  ).run(id, name, plaintext.slice(0, 8), createHash('sha256').update(plaintext).digest('hex'), 'inference', 5);
  return { id, name, plaintext };
}

interface Turn {
  /** Request start, unix seconds. */
  start: number;
  duration: number;
  prompt: number;
  completion: number;
  cached: number | null;
  finish: string;
  turnIndex: number;
  toolsOut: [string, string][];
  toolsIn: [string, number, boolean][];
}

interface RowOpts {
  key: SeedKey;
  session: string;
  parent?: string | null;
  batch?: string | null;
}

function insertTurn(db: DatabaseSync, o: RowOpts, t: Turn): void {
  const finished = t.start + t.duration;
  db.prepare(
    `INSERT INTO request_history(
       id, finished_at, model_id, model, token_name, token_id, variant_id, prompt_tokens, completion_tokens,
       cached_tokens, duration_s, ttft_s, finish_reason, started_iso, session_key, parent_session_key, turn_index,
       batch_id, tools_out, tools_in)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    randomUUID(),
    finished,
    MODEL_ID,
    MODEL,
    o.key.name,
    o.key.id,
    VARIANT,
    t.prompt,
    t.completion,
    t.cached,
    t.duration,
    Math.min(t.duration, 0.4 + t.prompt / 20000),
    t.finish,
    new Date(t.start * 1000).toISOString(),
    o.session,
    o.parent ?? null,
    t.turnIndex,
    o.batch ?? null,
    t.toolsOut.length ? JSON.stringify(t.toolsOut) : null,
    t.toolsIn.length ? JSON.stringify(t.toolsIn) : null,
  );
}

/** A deterministic agent-like session: the context grows turn by turn, mostly cached, with tool calls. */
function agentTurns(start: number, n: number, opts: { gap?: number; sys?: number; seed?: number; from?: number } = {}): Turn[] {
  const gap = opts.gap ?? 45;
  const seed = opts.seed ?? 1;
  let ctx = opts.sys ?? 9000;
  const out: Turn[] = [];
  for (let i = 0; i < n; i++) {
    const last = i === n - 1;
    const fam = TOOLS[(i + seed) % TOOLS.length];
    const prevCtx = ctx;
    ctx += 400 + ((i * 7919 + seed * 31) % 2600);
    out.push({
      start: start + i * gap,
      duration: 3 + ((i + seed) % 5),
      prompt: ctx,
      completion: 120 + ((i * 131 + seed) % 900),
      cached: i === 0 ? 0 : prevCtx,
      finish: last ? 'stop' : 'tool_calls',
      turnIndex: (opts.from ?? 0) + i,
      toolsOut: last ? [] : [[fam, hmac8()]],
      toolsIn: i === 0 ? [] : [[TOOLS[(i - 1 + seed) % TOOLS.length], 300 + ((i * 97) % 4000), i % 9 === 4]],
    });
  }
  return out;
}

function flower(db: DatabaseSync, key: SeedKey, start: number, prompt: number, completion: number): void {
  insertTurn(
    db,
    { key, session: skey() },
    { start, duration: 1.5, prompt, completion, cached: null, finish: 'stop', turnIndex: 0, toolsOut: [], toolsIn: [] },
  );
}

function session(db: DatabaseSync, o: RowOpts, turns: Turn[]): number {
  for (const t of turns) insertTurn(db, o, t);
  return turns.length;
}

/** The static part of the fixture: both keys and their history. */
export function seedForest(dbPath: string, now = Date.now() / 1000): SeedResult {
  const db = open(dbPath);
  const H = 3600;
  try {
    db.exec('BEGIN IMMEDIATE');
    const keyA = insertKey(db, 'e2e-alpha');
    const keyB = insertKey(db, 'e2e-beta');
    let a = 0, b = 0;

    // key A, spell 1 (40 h ago): an orchestrator with a subagent
    const s1 = skey();
    a += session(db, { key: keyA, session: s1 }, agentTurns(now - 40 * H, 14, { seed: 1, sys: 14000 }));
    a += session(db, { key: keyA, session: skey(), parent: s1 }, agentTurns(now - 40 * H + 200, 6, { seed: 2, sys: 5000 }));
    // spell 2 (20 h ago): a long session
    a += session(db, { key: keyA, session: skey() }, agentTurns(now - 20 * H, 24, { seed: 3, gap: 30 }));
    // spell 3 (6 h ago): two sessions side by side
    a += session(db, { key: keyA, session: skey() }, agentTurns(now - 6 * H, 10, { seed: 4 }));
    a += session(db, { key: keyA, session: skey() }, agentTurns(now - 6 * H + 600, 8, { seed: 5, sys: 3000 }));
    // a session from 16 h ago that came back 14 h ago: already moved (a ghost stands at its old spot)
    const moved = skey();
    a += session(db, { key: keyA, session: moved }, agentTurns(now - 16 * H, 6, { seed: 6 }));
    a += session(db, { key: keyA, session: moved }, agentTurns(now - 14 * H, 5, { seed: 7, sys: 20000, from: 6 }));
    // R1-R3 (each its own spell, > 1 h from any other): last turns 4, 9 and 11 h ago. Each motion run brings one back
    // live (resumeSession), so two runs on one stack (the headless and the gpu project) both see a transplant.
    const resume = [4, 9, 11].map((h, k) => {
      const key = skey();
      a += session(db, { key: keyA, session: key }, agentTurns(now - h * H, 8, { seed: 8 + k }));
      return key;
    });
    // flowers, one mushroom (no completion)
    flower(db, keyA, now - 30 * H, 800, 300);
    flower(db, keyA, now - 12 * H, 1200, 450);
    flower(db, keyA, now - 2 * H, 600, 200);
    flower(db, keyA, now - 40 * 60, 900, 120);
    flower(db, keyA, now - 25 * 60, 2400, 0);
    a += 5;

    // key B: its own two trees and a flower
    b += session(db, { key: keyB, session: skey() }, agentTurns(now - 30 * H, 12, { seed: 9, sys: 6000 }));
    b += session(db, { key: keyB, session: skey() }, agentTurns(now - 3 * H - 1800, 9, { seed: 10 }));
    flower(db, keyB, now - 4 * H, 700, 260);
    b += 1;

    db.exec('COMMIT');
    db.exec('PRAGMA wal_checkpoint(FULL)');
    return {
      keyA,
      keyB,
      resumeSessionKeys: resume,
      rows: { a, b },
    };
  } catch (e) {
    try {
      db.exec('ROLLBACK');
    } catch {
      /* not in a transaction */
    }
    throw e;
  } finally {
    db.close();
  }
}

/**
 * The growing tree, a fresh batch tree of key A per call (its id is returned):
 *   - `prior` turns already finished over the last few minutes (a tree that is on screen before anything is sampled);
 *   - then a turn every `everyMs`, finished now. With the default 35 prior turns the tree's data passes 60 requests
 *     ~52 s in (while a motion run samples): the size step the mockup's 0.8/1.0 tree scale made there (fixed by R25,
 *     every tree at scale 1) stays covered by the girth bound;
 *   - after `newSessionAfterMs`, the turns continue in a second session of the same batch: a new limb joins a tree
 *     already on screen (as a new session joins its spell's tree in real traffic).
 * The view runs 30 s behind now, so start this ≥ 35 s before sampling. stop() returns the number of live turns.
 */
export function startGrowing(
  dbPath: string,
  key: SeedKey,
  opts: { everyMs?: number; newSessionAfterMs?: number; prior?: number } = {},
): { treeId: string; stop: () => number } {
  const db = open(dbPath);
  const batch = `e2e-grow-${randomBytes(3).toString('hex')}`;
  const t0 = Date.now();
  const first = skey(), second = skey();
  const prior = opts.prior ?? 35;
  for (const t of agentTurns(t0 / 1000 - 8 - prior * 6, prior, { gap: 6, seed: 12, sys: 4000 }))
    insertTurn(db, { key, session: first, batch }, { ...t, duration: 2, finish: 'tool_calls' });
  let i = 0, j = 0, ctx = 4000 + prior * 1500;
  const add = () => {
    const now = Date.now() / 1000;
    const switched = opts.newSessionAfterMs !== undefined && Date.now() - t0 >= opts.newSessionAfterMs;
    if (switched && j === 0) ctx = 6000; // the new session's own first prompt
    const prev = ctx;
    ctx += 300 + ((i * 7919) % 1500);
    const fam = TOOLS[i % TOOLS.length];
    const k = switched ? j++ : prior + i;
    insertTurn(
      db,
      { key, session: switched ? second : first, batch },
      {
        start: now - 1.2,
        duration: 1.2,
        prompt: ctx,
        completion: 80 + ((i * 131) % 600),
        cached: k === 0 ? 0 : prev,
        finish: 'tool_calls',
        turnIndex: k,
        toolsOut: [[fam, hmac8()]],
        toolsIn: k === 0 ? [] : [[TOOLS[(i + 4) % TOOLS.length], 500 + ((i * 97) % 3000), false]],
      },
    );
    i++;
  };
  add();
  const timer = setInterval(add, opts.everyMs ?? 2000);
  return {
    treeId: `${key.id}:batch:${batch}`,
    stop: () => {
      clearInterval(timer);
      db.close();
      return i;
    },
  };
}

/**
 * The next turn, finished now, of the first of `sessionKeys` whose last turn is > 35 min old: > 30 min after it, so
 * the live view transplants that tree forward. Returns the session key used.
 */
export function resumeSession(dbPath: string, key: SeedKey, sessionKeys: string[]): string {
  const db = open(dbPath);
  try {
    const now = Date.now() / 1000;
    const pick = sessionKeys.find((k) => {
      const row = db.prepare('SELECT MAX(finished_at) AS last, COUNT(*) AS n FROM request_history WHERE session_key = ?').get(k) as
        | { last: number | null; n: number }
        | undefined;
      return row?.last != null && now - row.last > 35 * 60;
    });
    if (!pick) throw new Error('forest-seed: every resumable session was already resumed; restart the stack');
    const { n } = db.prepare('SELECT COUNT(*) AS n FROM request_history WHERE session_key = ?').get(pick) as { n: number };
    const [t] = agentTurns(now - 4, 1, { seed: 11, sys: 30000, from: n });
    insertTurn(db, { key, session: pick }, { ...t, duration: 4, finish: 'tool_calls', toolsOut: [['edit', hmac8()]] });
    return pick;
  } finally {
    db.close();
  }
}

/**
 * A farm-like week for the 7 d framing (quiet-hours compression): `keys` more keys (default two), each working
 * 09:00-18:00 (by the stack's clock, offset so no day touches the last 12 h) on the 6 days before today, in three 3 h
 * spells of two busy sessions each (`turns` turns, default 150, one every 30 s, like an agent farm); idle nights.
 * `name` prefixes the keys (one call per prefix). Returns the number of rows.
 * The perf matrix (forest.spec.ts) adds a bigger farm on top (`FOREST_PERF_FARM`): a realistic farm week.
 */
export function seedFarmWeek(
  dbPath: string,
  now = Date.now() / 1000,
  opts: { keys?: number; turns?: number; name?: string } = {},
): number {
  const db = open(dbPath);
  const H = 3600, D = 86400;
  const nKeys = opts.keys ?? 2, turns = opts.turns ?? 150, name = opts.name ?? 'e2e-farm';
  let n = 0;
  try {
    db.exec('BEGIN IMMEDIATE');
    const keys = Array.from({ length: nKeys }, (_, k) => insertKey(db, `${name}-${k + 1}`));
    for (let d = 1; d <= 6; d++)
      keys.forEach((key, k) => {
        const day = now - d * D - 21 * H; // "09:00" that day: 12 h before the same time today's start
        for (let s = 0; s < 3; s++)
          for (let j = 0; j < 2; j++) {
            const start = day + s * 3 * H + j * 80 * 60 + k * 420;
            n += session(db, { key, session: skey() }, agentTurns(start, turns, { gap: 30, seed: 20 + d * 7 + s * 3 + j + k, sys: 6000 + 4000 * (k % 2) }));
          }
      });
    db.exec('COMMIT');
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  } finally {
    db.close();
  }
  return n;
}

/** Keys named `<name>-…` exist: that farm is already seeded. */
export function farmSeeded(dbPath: string, name = 'e2e-farm'): boolean {
  const db = open(dbPath);
  try {
    return (db.prepare("SELECT COUNT(*) AS n FROM api_tokens WHERE name LIKE ? || '-%'").get(name) as { n: number }).n > 0;
  } finally {
    db.close();
  }
}

/** Rows in request_history (every key). */
export function historyRows(dbPath: string): number {
  const db = open(dbPath);
  try {
    return (db.prepare('SELECT COUNT(*) AS n FROM request_history').get() as { n: number }).n;
  } finally {
    db.close();
  }
}
