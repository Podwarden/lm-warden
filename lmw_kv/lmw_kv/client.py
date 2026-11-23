"""Blocking, fail-open client for the KV store daemon.

Plain ``socket`` with a hard per-call deadline, safe to call from many threads:
each thread has its own connection and no lock, so one thread's slow call (a
promotion in ``begin_load``) never delays or fails another thread's. A thread's connection closes when the thread
ends. Any timeout, connection error, malformed reply or ``ok: false`` makes the call return ``None`` (``0`` for ``lookup``); nothing is
raised. A failed connection is dropped (a late reply could not be matched to its
request) and not retried by that thread for ``RETRY_AFTER_S``, so a dead or hung daemon
costs a thread at most one timeout per half second. Failures are logged at most once
a minute.
"""

from __future__ import annotations

import itertools
import logging
import socket
import threading
import time
import weakref
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from lmw_kv import protocol as p
from lmw_kv.keys import Namespace

log = logging.getLogger("lmw_kv.client")

LOG_EVERY_S = 60.0
RETRY_AFTER_S = 0.5
#: ``begin_load`` may promote every key from NVMe: its deadline is the slow deadline
#: plus this much per key, at most ``LOAD_MAX_S``.
LOAD_PER_KEY_S = 0.05
LOAD_MAX_S = 30.0
#: ``register_ns`` runs once per namespace at engine start, never on a request path. It
#: may wait for the write queue to drain or a pool re-slab, so its deadline is generous,
#: and a call that fails on the transport (timeout, dropped connection) is tried once
#: more: registering is idempotent, and a failure here turns the engine's tier off for
#: its whole life.
REGISTER_TIMEOUT_S = 30.0
REGISTER_TRIES = 2


def _close_box(box: list[socket.socket | None]) -> None:
    sock, box[0] = box[0], None
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass


class _Conn:
    """One thread's connection, its unread bytes and its back-off.

    The socket is closed when this object is collected (the thread ended), so the
    daemon sees the connection drop and cleans up what the thread left open.
    """

    def __init__(self) -> None:
        self._box: list[socket.socket | None] = [None]
        self._finalizer = weakref.finalize(self, _close_box, self._box)
        self.buf = b""
        self.retry_at = 0.0

    @property
    def sock(self) -> socket.socket | None:
        return self._box[0]

    @sock.setter
    def sock(self, s: socket.socket | None) -> None:
        self._box[0] = s

    def close(self) -> None:
        _close_box(self._box)


class Client:
    def __init__(
        self,
        socket_path: str,
        timeout_s: float = 0.05,
        slow_timeout_s: float = 2.0,
        load_per_key_s: float = LOAD_PER_KEY_S,
        register_timeout_s: float = REGISTER_TIMEOUT_S,
    ) -> None:
        """``register_timeout_s`` is the deadline of each ``register_ns`` try (see
        ``REGISTER_TIMEOUT_S``). ``begin_load`` may promote every key from NVMe, so its
        deadline is ``slow_timeout_s + load_per_key_s * len(keys)``, at most
        ``LOAD_MAX_S``."""
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self.slow_timeout_s = slow_timeout_s
        self.register_timeout_s = register_timeout_s
        self.load_per_key_s = load_per_key_s
        self._local = threading.local()  # one connection per calling thread
        self._all: weakref.WeakSet[_Conn] = weakref.WeakSet()
        self._all_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._last_log = float("-inf")

    def close(self) -> None:
        """Close every live thread's connection; each thread's next call fails once, then reconnects."""
        with self._all_lock:
            conns = list(self._all)
        for c in conns:
            sock = c.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            c.close()

    # ---- ops --------------------------------------------------------------------

    def hello(self) -> dict[str, Any] | None:
        """``{"protocol", "minor", "ram_path", "ram_bytes"}`` (no ``minor`` before minor 1)."""
        return self._call({"op": "hello"})

    def register_ns(
        self,
        ns: Namespace,
        groups: dict[str, int],
        nvme: bool = True,
        *,
        owner: str = "",
        model: bytes = b"",
        role: str = "attention",
        rank: int = 0,
    ) -> bytes | None:
        """``nvme`` False: the namespace's blocks stay in RAM, never on NVMe.

        ``owner`` names who holds the namespace (NVMe is on only if every owner wants
        it); ``model`` (16 bytes) groups namespaces for accounting; ``role`` and ``rank``
        say which part of the model it is. A minor 0 daemon ignores them.

        A transport failure is tried once more (after the back-off); a refusal is not.
        """
        req = {
            "op": "register_ns",
            "namespace": asdict(ns),
            "groups": groups,
            "nvme": nvme,
            "owner": owner,
            "model": model.hex(),
            "role": role,
            "rank": rank,
        }
        for attempt in range(1, REGISTER_TRIES + 1):
            r, transport = self._call_once(req, self.register_timeout_s)
            if r is not None or not transport:
                break
            if attempt < REGISTER_TRIES:
                log.warning(
                    "KV daemon register_ns failed (try %d of %d); retrying", attempt, REGISTER_TRIES
                )
                time.sleep(max(0.0, self._conn().retry_at - time.monotonic()))
        if r is not None and attempt > 1:
            log.warning("KV daemon register_ns succeeded on try %d", attempt)
        return _unhex(r, "ns")

    def lookup(self, ns_id: bytes, keys: list[bytes]) -> int:
        r = self._call({"op": "lookup", "ns": ns_id.hex(), "keys": [k.hex() for k in keys]})
        n = r.get("n") if r else None
        return n if isinstance(n, int) and not isinstance(n, bool) else 0

    def begin_save(
        self,
        ns_id: bytes,
        key: bytes,
        parent: bytes,
        scope: bytes = b"",
        *,
        tenant: bytes = b"",
        scope_tag: bytes = b"",
        pos: bytes = b"",
    ) -> dict[str, int] | None:
        """``tenant`` pays for the block, ``scope_tag`` names its isolation scope (empty
        for a sharing key), ``pos`` is its position key (sent only when set; the daemon
        uses ``key`` otherwise)."""
        req: dict[str, Any] = {
            "op": "begin_save",
            "ns": ns_id.hex(),
            "key": key.hex(),
            "parent": parent.hex(),
            "scope": scope.hex(),
            "tenant": tenant.hex(),
            "scope_tag": scope_tag.hex(),
        }
        if pos:
            req["pos"] = pos.hex()
        r = self._call(req)
        off = r.get("offsets") if r else None
        return off if isinstance(off, dict) else None

    def commit_save(self, ns_id: bytes, key: bytes) -> bool | None:
        r = self._call({"op": "commit_save", "ns": ns_id.hex(), "key": key.hex()})
        return True if r is not None else None

    def abort_save(self, ns_id: bytes, key: bytes) -> bool | None:
        r = self._call({"op": "abort_save", "ns": ns_id.hex(), "key": key.hex()})
        return True if r is not None else None

    def begin_load(
        self, ns_id: bytes, keys: list[bytes], *, tenant: bytes = b""
    ) -> tuple[int, list[dict[str, int]], int] | None:
        """``(handle, slots, nvme_blocks)``; ``slots`` may be shorter than ``keys``;
        ``nvme_blocks`` of them were promoted from NVMe by this call. A ``tenant`` takes
        over the charge of the loaded blocks."""
        r = self._call(
            {
                "op": "begin_load",
                "ns": ns_id.hex(),
                "keys": [k.hex() for k in keys],
                "tenant": tenant.hex(),
            },
            self.load_timeout(len(keys)),
        )
        if (
            r is None
            or not isinstance(r.get("handle"), int)
            or not isinstance(r.get("slots"), list)
        ):
            return None
        nvme = r.get("nvme", 0)
        if not isinstance(nvme, int) or isinstance(nvme, bool) or nvme < 0:
            nvme = 0
        return r["handle"], r["slots"], min(nvme, len(r["slots"]))

    def load_timeout(self, n_keys: int) -> float:
        """The ``begin_load`` deadline for ``n_keys`` keys."""
        return min(self.slow_timeout_s + self.load_per_key_s * max(n_keys, 0), LOAD_MAX_S)

    def end_load(self, handle: int) -> bool | None:
        r = self._call({"op": "end_load", "handle": handle})
        return True if r is not None else None

    def record_hits(
        self,
        tag: str,
        ram_tokens: int,
        nvme_tokens: int,
        load_s: float,
        *,
        tenant: bytes = b"",
    ) -> bool | None:
        r = self._call(
            {
                "op": "record_hits",
                "tag": tag,
                "ram_tokens": ram_tokens,
                "nvme_tokens": nvme_tokens,
                "load_s": load_s,
                "tenant": tenant.hex(),
            }
        )
        return True if r is not None else None

    def pop_hits(self, tag: str) -> dict[str, Any] | None:
        """The hits recorded for ``tag`` (forgotten once read); ``{}`` when the daemon
        answered and nothing was recorded; None only when the call failed."""
        r = self._call({"op": "pop_hits", "tag": tag})
        if r is None:
            return None
        hits = r.get("hits")
        if hits is None:  # a daemon from before ``{}``: it answered, nothing recorded
            return {}
        return hits if isinstance(hits, dict) else None

    def stats(self) -> dict[str, Any] | None:
        r = self._call({"op": "stats"})
        st = r.get("stats") if r else None
        return st if isinstance(st, dict) else None

    # ---- minor 1 (a minor 0 daemon answers unknown_op: None) ---------------------

    def release_owner(self, owner: str) -> int | None:
        """Drop ``owner`` from every namespace; how many were left with no owner."""
        r = self._call({"op": "release_owner", "owner": owner}, self.slow_timeout_s)
        return _count(r, "released")

    def retire_scopes(self, tags: Sequence[bytes]) -> int | None:
        """Retire isolation scope tags for good (idempotent); how many were new."""
        r = self._call(
            {"op": "retire_scopes", "scope_tags": [t.hex() for t in tags]}, self.slow_timeout_s
        )
        return _count(r, "added")

    def set_policy(self, policy: dict[str, Any]) -> bool | None:
        """Send the eviction policy (PROTOCOL.md ``set_policy``); True once applied."""
        r = self._call({"op": "set_policy", "policy": policy}, self.slow_timeout_s)
        return True if r is not None else None

    # ---- transport --------------------------------------------------------------

    def _call(self, req: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any] | None:
        return self._call_once(req, timeout_s)[0]

    def _call_once(
        self, req: dict[str, Any], timeout_s: float | None = None
    ) -> tuple[dict[str, Any] | None, bool]:
        """(reply or None, whether a None came from the transport rather than a refusal)."""
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        conn = self._conn()
        if time.monotonic() < conn.retry_at:
            return None, True
        try:
            reply = self._roundtrip(conn, req, deadline)
        except (OSError, ValueError, p.ProtocolError) as exc:
            conn.close()
            conn.buf = b""
            conn.retry_at = time.monotonic() + RETRY_AFTER_S
            self._note(f"{req['op']}: {exc or type(exc).__name__}")
            return None, True
        if reply.get("ok") is not True:  # the connection is fine; this request was not
            self._note(f"{req['op']} refused: {reply.get('error')}")
            return None, False
        return reply, False

    def _conn(self) -> _Conn:
        c: _Conn | None = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = _Conn()
            with self._all_lock:
                self._all.add(c)
        return c

    def _roundtrip(self, conn: _Conn, req: dict[str, Any], deadline: float) -> dict[str, Any]:
        sock = conn.sock or self._connect(conn, deadline)
        rid = next(self._ids)
        sock.settimeout(_left(deadline))
        sock.sendall(p.encode({**req, "id": rid}))
        while b"\n" not in conn.buf:
            sock.settimeout(_left(deadline))
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            conn.buf += chunk
            if len(conn.buf) > p.MAX_LINE:
                raise ValueError("reply too long")
        line, _, conn.buf = conn.buf.partition(b"\n")
        reply = p.decode(line)
        if reply.get("id") != rid:
            raise ValueError("reply does not match the request")
        return reply

    def _connect(self, conn: _Conn, deadline: float) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(_left(deadline))
            s.connect(self.socket_path)
        except BaseException:
            s.close()
            raise
        conn.sock = s
        conn.buf = b""
        return s

    def _note(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_log >= LOG_EVERY_S:
            self._last_log = now
            log.warning("KV daemon call failed, continuing without it: %s", msg)


def _left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("deadline passed")
    return left


def _unhex(r: dict[str, Any] | None, field: str) -> bytes | None:
    v = r.get(field) if r else None
    try:
        return bytes.fromhex(v) if isinstance(v, str) else None
    except ValueError:
        return None


def _count(r: dict[str, Any] | None, field: str) -> int | None:
    v = r.get(field) if r else None
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
