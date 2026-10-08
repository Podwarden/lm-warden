#!/usr/bin/env python3
"""Live acceptance of POST /v1/responses with the real OpenAI Codex CLI.

usage: codex_acceptance.py BASE KEY_FILE ADMIN_TOKEN_FILE MODEL
           [--codex PATH] [--context N] [--keep] [--no-think] [--write-turn] [--capture DIR]

  BASE              the warden's origin, e.g. https://lmwarden.com (no /v1)
  KEY_FILE          a file holding an inference key (vw_...)
  ADMIN_TOKEN_FILE  a file holding an admin token (vwa_...) or an admin JWT
  MODEL             a loaded served model name
  --codex PATH      the codex binary (default: $CODEX_BIN, else `codex` on PATH)
  --no-think        set model_reasoning_effort = "none" in the config; then check
                    that no in-flight row reaches the thinking phase. (The
                    forwarded chat body itself is not readable through the
                    admin API, so enable_thinking=false is not asserted.)
  --write-turn      also run a turn in a workspace-write sandbox where Codex
                    creates a file (the apply_patch / shell edit path)
  --capture DIR     also dump every /v1/responses request body Codex sends to
                    DIR/NNN-request.json, scrubbed (absolute paths, hostnames,
                    uuids, vw_ keys), via a local forwarding proxy; use it to
                    replace turn 2 of tests/fixtures/responses/codex_turns.json
  --context N       model_context_window for the config (default: the model's
                    max_model_len from /v1/models, else 131072)

The warden must RUN THE BRANCH THAT SERVES /v1/responses. Codex runs in an
isolated CODEX_HOME in a temp directory, read-only sandbox, approvals never,
web search disabled. Checks:

  1. negatives: a wrong key gets the warden's own OpenAI-shaped 401, and
     previous_response_id gets the documented 400
  2. a plain non-streamed Responses request answers with a message
  3. turn 1 (`codex exec`): exit 0, no "Reconnecting", a shell command item,
     the answer says 3 (three files in the work directory); meanwhile the
     in-flight table shows a row whose session_id is the Codex thread id and
     its phase progresses
  4. turn 2 (`codex exec resume --last`): same thread id; the finished rows of
     the thread (found by request id, history keeps no session id) show
     cache_est_tokens > 0 and cached_tokens > 0

Prints metadata only; neither key is ever printed. Exit 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 codex-acceptance"
FAILED: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(label)
    return ok


def call(base: str, method: str, path: str, headers: dict, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={
            "user-agent": UA,
            **({"content-type": "application/json"} if data else {}),
            **headers,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def jget(base: str, path: str, token: str):
    status, text = call(base, "GET", path, {"authorization": f"Bearer {token}"}, timeout=30)
    return status, (json.loads(text) if status == 200 else text)


def scrub(text: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<key>")
    return re.sub(r"vwa?_[A-Za-z0-9_-]{6,}", "<key>", text)


# -- capture: a forwarding proxy that dumps scrubbed request bodies ---------------------------

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_PATH_RE = re.compile(
    r"(?<![\w.])/(?:Users|private|tmp|var|home|opt|root|Volumes)/[^\s\"'<>\\,;)]*"
)
_KEY_RE = re.compile(r"vwa?_[A-Za-z0-9_-]{4,}")


def scrub_capture(body: object, hosts: list[str]) -> object:
    """A request body with absolute paths, hostnames, uuids and keys replaced.

    UUIDs map to stable placeholders (the same id stays the same id, so the
    thread/session relationship survives)."""
    text = json.dumps(body)
    ids: dict[str, str] = {}

    def uid(m: re.Match[str]) -> str:
        n = ids.setdefault(m.group(0).lower(), len(ids) + 1)
        return f"00000000-0000-4000-8000-{n:012d}"

    text = _UUID_RE.sub(uid, text)
    text = _KEY_RE.sub("vw_SCRUBBED", text)
    text = _PATH_RE.sub("/scrubbed/path", text)
    for h in hosts:
        if h:
            text = text.replace(h, "warden.example")
    return json.loads(text)


def start_capture_proxy(
    upstream: str, out_dir: Path
) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Forward everything to ``upstream`` (streaming), dump POST /v1/responses bodies."""
    up = urllib.parse.urlsplit(upstream)
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}
    hosts = [up.netloc, up.hostname or ""]

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a: object) -> None:  # silence
            pass

        def _forward(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else None
            if (
                self.command == "POST"
                and self.path.split("?")[0].endswith("/v1/responses")
                and body
            ):
                try:
                    counter["n"] += 1
                    data = scrub_capture(json.loads(body), hosts)
                    (out_dir / f"{counter['n']:03d}-request.json").write_text(
                        json.dumps(data, indent=1)
                    )
                except Exception:  # noqa: BLE001 -- capture is best effort
                    pass
            conn_cls = (
                http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
            )
            conn = conn_cls(up.netloc, timeout=900)
            headers = {
                k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")
            }
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := resp.read(4096):
                self.wfile.write(chunk)
                self.wfile.flush()
            conn.close()
            self.close_connection = True

        do_GET = do_POST = do_PUT = do_DELETE = _forward

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def write_config(home: Path, base: str, model: str, context: int, no_think: bool = False) -> None:
    (home / "config.toml").write_text(
        f'model = "{model}"\n'
        'model_provider = "lmwarden"\n'
        f"model_context_window = {context}\n"
        + ('model_reasoning_effort = "none"\n' if no_think else "")
        + 'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        'web_search = "disabled"\n'
        "\n"
        "[model_providers.lmwarden]\n"
        'name = "LM Warden"\n'
        f'base_url = "{base}/v1"\n'
        'env_key = "LMWARDEN_KEY"\n'
        'wire_api = "responses"\n'
    )


def run_codex(codex: str, args: list[str], env: dict, cwd: Path, timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        [codex, *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


class Poller(threading.Thread):
    """Samples the in-flight table for rows of one session while a turn runs."""

    def __init__(self, base: str, admin: str) -> None:
        super().__init__(daemon=True)
        self.base, self.admin = base, admin
        self.stop = threading.Event()
        self.rows: list[dict] = []

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                status, body = jget(self.base, "/api/stats/requests", self.admin)
                if status == 200:
                    self.rows.extend(body.get("requests", []))
            except Exception:  # noqa: BLE001 -- a missed sample is fine
                pass
            time.sleep(0.25)


def agent_messages(out: str) -> list[str]:
    """The text of every agent_message item in `codex exec --json` output."""
    texts = []
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        item = ev.get("item") if isinstance(ev, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message":
            texts.append(str(item.get("text", "")))
    return texts


def says_three(texts: list[str]) -> bool:
    """A standalone 3 (or 'three') in any message or in them joined: a model
    that thinks may split its reply over several agent_message events."""
    pat = re.compile(r"(?<![\w.])(3|three)(?![\w.])", re.IGNORECASE)
    return any(pat.search(t) for t in [*texts, " ".join(texts)])


def thread_id(home: Path) -> str | None:
    """The Codex thread id: the newest rollout's session_meta id."""
    files = sorted(home.glob("sessions/**/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime)
    for f in reversed(files):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "session_meta":
                sid = (d.get("payload") or {}).get("id")
                if isinstance(sid, str):
                    return sid
    return None


def finished_rows(base: str, admin: str, ids: set[str]) -> list[dict]:
    """History rows by request id (history keeps no session id, only its source)."""
    status, body = jget(base, "/api/stats/live/finished?limit=200", admin)
    rows = body.get("requests", []) if status == 200 else []
    return sorted((r for r in rows if r.get("id") in ids), key=lambda r: r.get("finished_at") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("key_file")
    ap.add_argument("admin_token_file")
    ap.add_argument("model")
    ap.add_argument("--codex", default=os.environ.get("CODEX_BIN") or shutil.which("codex"))
    ap.add_argument("--context", type=int)
    ap.add_argument("--keep", action="store_true", help="keep the temp CODEX_HOME")
    ap.add_argument(
        "--no-think",
        action="store_true",
        help='set model_reasoning_effort = "none" and check thinking stays off',
    )
    ap.add_argument(
        "--write-turn", action="store_true", help="also run a workspace-write edit turn"
    )
    ap.add_argument("--capture", metavar="DIR", help="dump scrubbed /v1/responses request bodies")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    key = Path(args.key_file).read_text().strip()
    admin = Path(args.admin_token_file).read_text().strip()
    if not args.codex:
        print("no codex binary: pass --codex or set CODEX_BIN", file=sys.stderr)
        return 2

    auth = {"authorization": f"Bearer {key}"}
    context = args.context
    if context is None:
        status, text = call(base, "GET", "/v1/models", auth, timeout=30)
        context = 131072
        if status == 200:
            for m in json.loads(text).get("data", []):
                if m.get("id") == args.model and isinstance(m.get("max_model_len"), int):
                    context = m["max_model_len"]
    print(f"base {base}  model {args.model}  context {context}")

    # -- 1/2: negatives and a plain request, no Codex involved ------------------------------
    body = {"model": args.model, "input": "Reply with OK.", "max_output_tokens": 16}
    status, text = call(
        base, "POST", "/v1/responses", {"authorization": "Bearer vw_wrong_key_0000"}, body
    )
    err = (json.loads(text) if text.startswith("{") else {}).get("error", {})
    check(
        status == 401 and err.get("type") == "authentication_error",
        "wrong key -> OpenAI-shaped 401",
        f"status {status}, type {err.get('type')}",
    )
    status, text = call(
        base, "POST", "/v1/responses", auth, {**body, "previous_response_id": "resp_x"}
    )
    err = (json.loads(text) if text.startswith("{") else {}).get("error", {})
    check(
        status == 400 and "previous_response_id" in str(err.get("message")),
        "previous_response_id -> 400",
        f"status {status}",
    )
    status, text = call(base, "POST", "/v1/responses", auth, {**body, "stream": False})
    resp = json.loads(text) if status == 200 else {}
    texts = [
        c.get("text")
        for i in resp.get("output", [])
        if i.get("type") == "message"
        for c in i.get("content", [])
        if c.get("type") == "output_text"
    ]
    check(
        status == 200 and resp.get("object") == "response" and any(texts),
        "plain Responses request answers",
        f"status {status}, {len(''.join(t for t in texts if t))} chars",
    )
    if status == 404:
        print("  /v1/responses is 404: the warden is not running the Responses branch")
        return 1

    # -- 3/4: Codex itself --------------------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="codex-acceptance-"))
    home, work = tmp / "home", tmp / "work"
    home.mkdir()
    work.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (work / name).write_text(name + "\n")
    codex_base = base
    if args.capture:
        _proxy, codex_base = start_capture_proxy(base, Path(args.capture))
        print(f"capturing request bodies to {args.capture} via {codex_base}")
    write_config(home, codex_base, args.model, context, args.no_think)
    env = {**os.environ, "CODEX_HOME": str(home), "LMWARDEN_KEY": key, "NO_COLOR": "1"}
    common = ["--skip-git-repo-check", "--json"]
    try:
        poll = Poller(base, admin)
        poll.start()
        seen_ids: set[str] = set()
        t0 = time.time()
        rc, out = run_codex(
            args.codex,
            [
                "exec",
                *common,
                "Use the shell to count the files in the current directory, "
                "then answer with only that number.",
            ],
            env,
            work,
            900,
        )
        poll.stop.set()
        poll.join()
        out = scrub(out, key, admin)
        tid = thread_id(home)
        check(rc == 0, "turn 1: codex exec exits 0", f"rc {rc}, {time.time() - t0:.0f}s")
        if rc != 0:
            print("--- codex output (tail, keys scrubbed) ---\n" + out[-2000:])
        check("Reconnecting" not in out, "turn 1: no 'Reconnecting'")
        commands = [
            ln for ln in out.splitlines() if "command_execution" in ln or "exec_command" in ln
        ]
        check(bool(commands), "turn 1: a shell command item ran", f"{len(commands)} event lines")
        answers = agent_messages(out)
        check(
            says_three(answers),
            "turn 1: the answer says 3",
            f"{len(answers)} agent_message events",
        )
        check(tid is not None, "turn 1: a thread id was recorded", (tid or "")[:8])
        if tid is None:
            print(out[-1500:])
            return 1

        mine = [r for r in poll.rows if r.get("session_id") == tid]
        phases = sorted({r.get("phase") for r in mine if r.get("phase")})
        check(
            bool(mine),
            "in-flight table: a row carries the Codex thread id as session_id",
            f"{len(mine)} samples",
        )
        check(
            len(phases) >= 2 or "decode" in phases,
            "in-flight table: phases progress",
            ",".join(phases),
        )
        if args.no_think:
            check(
                "thinking" not in phases,
                "no-think: no row reached the thinking phase",
                ",".join(phases),
            )
        check(
            all(r.get("path") == "/v1/chat/completions" for r in mine) if mine else False,
            "in-flight table: path is the chat route",
        )

        seen_ids |= {r["id"] for r in mine if r.get("id")}
        time.sleep(2)  # request history is flushed in batches
        poll2 = Poller(base, admin)
        poll2.start()
        rc2, out2 = run_codex(
            args.codex,
            [
                "exec",
                "resume",
                "--last",
                *common,
                "Now list the three file names, comma separated.",
            ],
            env,
            work,
            900,
        )
        poll2.stop.set()
        poll2.join()
        out2 = scrub(out2, key, admin)
        seen_ids |= {r["id"] for r in poll2.rows if r.get("session_id") == tid and r.get("id")}
        check(rc2 == 0, "turn 2: codex exec resume --last exits 0", f"rc {rc2}")
        check("Reconnecting" not in out2, "turn 2: no 'Reconnecting'")
        check(thread_id(home) == tid, "turn 2: the same thread id")
        time.sleep(3)
        rows = finished_rows(base, admin, seen_ids)
        check(
            len(rows) >= 3,
            "history: finished rows for the thread",
            f"{len(rows)} rows of {len(seen_ids)} seen",
        )
        later = rows[1:]
        est = [r.get("cache_est_tokens") or 0 for r in later]
        cached = [r.get("cached_tokens") or 0 for r in later]
        check(
            any(v > 0 for v in est), "turn 2+: cache_est_tokens > 0", f"max {max(est, default=0)}"
        )
        check(
            any(v > 0 for v in cached),
            "turn 2+: measured cached_tokens > 0",
            f"max {max(cached, default=0)}",
        )
        sources = {r.get("session_source") for r in rows}
        check(
            sources <= {"session_id_header", "prompt_cache_key"},
            "history: session source is a Codex signal",
            ",".join(sorted(str(s) for s in sources)),
        )
        if args.write_turn:
            rc3, out3 = run_codex(
                args.codex,
                [
                    "exec",
                    "resume",
                    "--last",
                    *common,
                    "-c",
                    'sandbox_mode="workspace-write"',
                    "Create a file named hello.txt in the current directory containing exactly the "
                    "word hi. Use apply_patch or a shell command, then say done.",
                ],
                env,
                work,
                900,
            )
            out3 = scrub(out3, key, admin)
            check(rc3 == 0, "write turn: codex exits 0", f"rc {rc3}")
            check("Reconnecting" not in out3, "write turn: no 'Reconnecting'")
            hello = work / "hello.txt"
            check(
                hello.exists() and hello.read_text().strip() == "hi",
                "write turn: hello.txt holds 'hi'",
            )
            if not hello.exists():
                print("--- write turn output (tail, keys scrubbed) ---\n" + out3[-1500:])
        if args.capture:
            n = len(list(Path(args.capture).glob("*-request.json")))
            check(n > 0, "capture: request bodies dumped", f"{n} files in {args.capture}")
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else 'FAILED: ' + '; '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
