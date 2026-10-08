// Global setup for playwright.forest.config.ts: a throwaway local LM Warden stack for the session forest e2e.
// No docker, no GPU, no model: the forest only reads request_history.
//
//   1. a temp data dir; migrations applied and an admin seeded with first-run setup marked done, through the app's
//      own code (app.db.migrations, the same rows tests/conftest.py::seed_admin_user writes);
//   2. two inference keys and their request_history rows (forest-seed.ts);
//   3. uvicorn app.main:app on a free port (VW_DATA_DIR = the temp dir);
//   4. the UI on a free port: the standalone production build (`next build` first, unless FOREST_E2E_SKIP_BUILD=1),
//      or `next dev` with FOREST_E2E_NEXT=dev;
//   5. a small reverse proxy on a third port standing in for Caddy (/ui/* → next, everything else → FastAPI): next.config
//      has no /api rewrite since the unified-port change, so the browser must see both on one origin.
// The tests read the stack from env (FOREST_E2E_*). The returned function is the global teardown: it stops the proxy
// and kills uvicorn and next (process groups), then removes the temp dir (FOREST_E2E_KEEP=1 keeps it).
//
// Env: VW_VENV (default ../.venv of the main checkout, then ../.venv), FOREST_E2E_NEXT=prod|dev,
// FOREST_E2E_SKIP_BUILD=1, FOREST_E2E_KEEP=1.

import { type ChildProcess, execFileSync, spawn } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, openSync, readdirSync, rmSync } from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { seedForest } from './forest-seed';

const FRONTEND = path.resolve(__dirname, '../../..');
const REPO = path.resolve(FRONTEND, '..');
const ADMIN_PW = 'forest-e2e-pw';

function venvPython(): string {
  const cands = [
    process.env.VW_VENV && path.join(process.env.VW_VENV, 'bin/python'),
    path.join(REPO, '.venv/bin/python'),
    // a worktree under <checkout>/.wt/<name>: the main checkout's venv
    path.resolve(REPO, '../../.venv/bin/python'),
  ].filter(Boolean) as string[];
  const hit = cands.find((p) => existsSync(p));
  if (!hit) throw new Error(`forest-stack: no Python venv found (tried ${cands.join(', ')}); set VW_VENV`);
  return hit;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.unref();
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address() as net.AddressInfo;
      s.close(() => resolve(port));
    });
  });
}

async function waitFor(url: string, child: ChildProcess, what: string, timeoutMs: number): Promise<void> {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    if (child.exitCode !== null) throw new Error(`forest-stack: ${what} exited with ${child.exitCode}`);
    try {
      const r = await fetch(url);
      if (r.status < 500) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`forest-stack: ${what} not reachable at ${url} after ${timeoutMs} ms`);
}

/** Kill a detached child's whole process group (next and npx spawn their own children), then wait for it. */
async function killGroup(child: ChildProcess | null): Promise<void> {
  // exitCode / signalCode are set once the child has exited (by code or by a signal): its 'exit' already fired
  const exited = () => !child || child.exitCode !== null || child.signalCode !== null;
  if (!child || child.pid === undefined || exited()) return;
  const gone = new Promise<void>((r) => {
    child.once('exit', () => r());
    if (exited()) r(); // raced with the exit between the check above and the listener
  });
  try {
    process.kill(-child.pid, 'SIGTERM');
  } catch {
    /* already gone */
  }
  const t = setTimeout(() => {
    try {
      process.kill(-child.pid!, 'SIGKILL');
    } catch {
      /* gone */
    }
  }, 8000);
  await gone;
  clearTimeout(t);
}

/** Only what the children need from the caller's env: no inherited VW_* (or any other app setting) leaks in. */
const ENV_ALLOW = ['PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'TEMP', 'TMP', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM'];
function cleanEnv(extra: Record<string, string>): NodeJS.ProcessEnv {
  const env: Record<string, string> = {};
  for (const k of ENV_ALLOW) if (process.env[k] !== undefined) env[k] = process.env[k]!;
  // ProcessEnv declares NODE_ENV (next's typings); a child without it gets its own default
  return { ...env, ...extra } as NodeJS.ProcessEnv;
}

/**
 * The directory of the standalone build's server.js. Next nests it under the inferred workspace root (a parent
 * directory with another lockfile moves it to .next/standalone/<path from that root>/server.js).
 */
function standaloneDir(root: string): string {
  let level = [root];
  for (let depth = 0; depth < 8 && level.length; depth++) {
    const hit = level.find((d) => existsSync(path.join(d, 'server.js')));
    if (hit) return hit;
    level = level.flatMap((d) =>
      readdirSync(d, { withFileTypes: true })
        .filter((e) => e.isDirectory() && e.name !== 'node_modules' && e.name !== '.next')
        .map((e) => path.join(d, e.name)),
    );
  }
  throw new Error('forest-stack: no standalone build (.next/standalone/**/server.js); run next build');
}

/** Migrations + admin + setup done, through the app's own migration code (mirrors tests/conftest.py). */
const INIT_PY = `
import asyncio, json, sqlite3, sys
from pathlib import Path
import bcrypt
from app.db.database import open_db
from app.db.migrations import apply_migrations
db_path, pw = Path(sys.argv[1]), sys.argv[2]
async def migrate():
    async with open_db(db_path) as db:
        await apply_migrations(db)
asyncio.run(migrate())
h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(4)).decode()
with sqlite3.connect(db_path, isolation_level=None) as db:
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("BEGIN IMMEDIATE")
    db.execute("INSERT INTO users(username, password_hash) VALUES ('admin', ?)", (h,))
    db.execute("UPDATE setup_state SET step='done', draft=? WHERE id=1", (json.dumps({"allowed_gpu_indices": [0]}),))
    db.execute("COMMIT")
    db.execute("PRAGMA wal_checkpoint(FULL)")
`;

function startProxy(port: number, uiPort: number, apiPort: number): Promise<http.Server> {
  const target = (url = '/') => (url === '/ui' || url.startsWith('/ui/') || url.startsWith('/ui?') ? uiPort : apiPort);
  const server = http.createServer((req, res) => {
    const up = http.request(
      { host: '127.0.0.1', port: target(req.url), method: req.method, path: req.url, headers: req.headers },
      (ur) => {
        res.writeHead(ur.statusCode ?? 502, ur.headers);
        ur.pipe(res);
      },
    );
    up.on('error', () => {
      if (!res.headersSent) res.writeHead(502);
      res.end();
    });
    req.pipe(up);
  });
  // websockets (next dev's HMR): a raw pipe after replaying the request head
  server.on('upgrade', (req, socket, head) => {
    const s = net.connect(target(req.url), '127.0.0.1', () => {
      const lines = [`${req.method} ${req.url} HTTP/${req.httpVersion}`];
      for (let i = 0; i < req.rawHeaders.length; i += 2) lines.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
      s.write(lines.join('\r\n') + '\r\n\r\n');
      if (head.length) s.write(head);
      s.pipe(socket);
      socket.pipe(s);
    });
    s.on('error', () => socket.destroy());
    socket.on('error', () => s.destroy());
  });
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve(server)));
}

export default async function globalSetup(): Promise<() => Promise<void>> {
  const dataDir = mkdtempSync(path.join(os.tmpdir(), 'forest-e2e-'));
  const dbPath = path.join(dataDir, 'vllm-warden.db');
  const py = venvPython();
  const [apiPort, uiPort, proxyPort] = [await freePort(), await freePort(), await freePort()];
  const origin = `http://127.0.0.1:${proxyPort}`;
  let api: ChildProcess | null = null, ui: ChildProcess | null = null, proxy: http.Server | null = null;

  const teardown = async () => {
    const closed = new Promise<void>((r) => (proxy ? proxy.close(() => r()) : r()));
    proxy?.closeAllConnections(); // keep-alive sockets would hold close() open
    await closed;
    await killGroup(ui);
    await killGroup(api);
    if (process.env.FOREST_E2E_KEEP !== '1') rmSync(dataDir, { recursive: true, force: true });
  };

  try {
    // 1-2. schema, admin, keys, history
    execFileSync(py, ['-c', INIT_PY, dbPath, ADMIN_PW], { cwd: REPO, stdio: 'inherit', env: cleanEnv({ HF_HUB_OFFLINE: '1' }) });
    const seed = seedForest(dbPath);

    // 3. FastAPI
    const apiLog = openSync(path.join(dataDir, 'uvicorn.log'), 'a');
    api = spawn(py, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(apiPort)], {
      cwd: REPO,
      detached: true,
      stdio: ['ignore', apiLog, apiLog],
      env: cleanEnv({
        VW_DATA_DIR: dataDir,
        VW_HF_CACHE_DIR: path.join(dataDir, 'hf'),
        HF_HOME: path.join(dataDir, 'hf'),
        HF_HUB_OFFLINE: '1',
        VW_COOKIE_SECRET: 'forest-e2e-cookie-secret-0123456789abcdef',
        VW_FRONTEND_ORIGIN: origin,
        VW_CONTAINER_GPU_COUNT: '0',
        VW_WATCHDOG_ENABLED: '0',
        VW_DP_RANK_SCRAPER: '0',
      }),
    });
    await waitFor(`http://127.0.0.1:${apiPort}/healthz`, api, 'uvicorn', 60_000);

    // 4. the UI
    const uiLog = openSync(path.join(dataDir, 'next.log'), 'a');
    const nextBin = path.join(FRONTEND, 'node_modules/.bin/next');
    const uiEnv = cleanEnv({ PORT: String(uiPort), HOSTNAME: '127.0.0.1', NEXT_TELEMETRY_DISABLED: '1' });
    if (process.env.FOREST_E2E_NEXT === 'dev') {
      ui = spawn(nextBin, ['dev', '-p', String(uiPort), '-H', '127.0.0.1'], {
        cwd: FRONTEND, detached: true, stdio: ['ignore', uiLog, uiLog], env: uiEnv,
      });
    } else {
      if (process.env.FOREST_E2E_SKIP_BUILD !== '1') {
        execFileSync(nextBin, ['build'], { cwd: FRONTEND, stdio: 'inherit', env: { ...uiEnv, NODE_ENV: 'production' } });
      }
      const standalone = standaloneDir(path.join(FRONTEND, '.next/standalone'));
      // what the Dockerfile copies next to server.js
      cpSync(path.join(FRONTEND, '.next/static'), path.join(standalone, '.next/static'), { recursive: true });
      if (existsSync(path.join(FRONTEND, 'public'))) cpSync(path.join(FRONTEND, 'public'), path.join(standalone, 'public'), { recursive: true });
      ui = spawn(process.execPath, ['server.js'], {
        cwd: standalone, detached: true, stdio: ['ignore', uiLog, uiLog], env: { ...uiEnv, NODE_ENV: 'production' },
      });
    }
    await waitFor(`http://127.0.0.1:${uiPort}/ui/login`, ui, 'next', 180_000);

    // 5. one origin for the browser
    proxy = await startProxy(proxyPort, uiPort, apiPort);
    // warm the forest page (next dev compiles on first hit)
    await fetch(`${origin}/ui/forest`).catch(() => {});

    Object.assign(process.env, {
      FOREST_E2E_URL: origin,
      FOREST_E2E_DB: dbPath,
      FOREST_E2E_DATA_DIR: dataDir,
      FOREST_E2E_ADMIN_PW: ADMIN_PW,
      FOREST_E2E_SEED: JSON.stringify(seed),
      // keys the motion sample cache: a sample is only ever reused within the run that took it
      FOREST_E2E_RUN_ID: path.basename(dataDir),
    });
    console.log(`forest-stack: ${origin} (api :${apiPort}, ui :${uiPort}, data ${dataDir})`);
    return teardown;
  } catch (e) {
    await teardown();
    throw e;
  }
}
