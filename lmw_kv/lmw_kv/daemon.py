"""The KV store daemon: a ``Store`` served over a Unix socket (see ``protocol.py``).

Every request on one connection is answered in order. ``Store`` calls block and
may wait (NVMe promotion, flush), so each runs in a worker thread and a slow call
never stalls another connection. The daemon remembers, per connection, the saves
begun but not committed and the load handles not ended; when the connection
drops it aborts those saves and ends those loads, so a crashed engine leaks
neither slots nor pins.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import logging
import os
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from lmw_kv import protocol as p
from lmw_kv.fair import Policy
from lmw_kv.keys import Namespace
from lmw_kv.reuse import REUSE_PERSIST_S as _REUSE_PERSIST_S
from lmw_kv.reuse import TOP_MAX
from lmw_kv.store import Store, StoreConfig

log = logging.getLogger("lmw_kv.daemon")

KEY_BYTES = 32
SCOPE_MAX = 255
TAG_BYTES = 16  # tenant, scope tag, model
OWNER_MAX = 255  # UTF-8 bytes
ROLE_MAX = 32
LOAD_TAG_MAX = 255  # UTF-8 bytes of a begin_load tag
DEFAULT_ROLE = "attention"


def _hex(req: dict[str, Any], field: str, *, size: int | None = KEY_BYTES) -> bytes:
    v = req.get(field)
    if not isinstance(v, str):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: hex string required")
    try:
        raw = bytes.fromhex(v)
    except ValueError:
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: not hex") from None
    if size is not None and len(raw) != size:
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: {size} bytes required")
    if size is None and len(raw) > SCOPE_MAX:
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: at most {SCOPE_MAX} bytes")
    return raw


def _hex_list(req: dict[str, Any], field: str) -> list[bytes]:
    v = req.get(field)
    if not isinstance(v, list):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: list of hex strings required")
    return [_hex({field: x}, field) for x in v]


def _opt_hex(req: dict[str, Any], field: str, size: int) -> bytes:
    """An optional minor 1 byte field: missing or ``""`` is empty, else exactly ``size``."""
    if req.get(field) in (None, ""):
        return b""
    return _hex(req, field, size=size)


def _hex_tags(req: dict[str, Any], field: str) -> list[bytes]:
    v = req.get(field)
    if not isinstance(v, list):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: list of hex strings required")
    return [_hex({field: x}, field, size=TAG_BYTES) for x in v]


def _text(req: dict[str, Any], field: str, default: str, limit: int) -> str:
    v = req.get(field, default)
    if not isinstance(v, str):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: string required")
    try:
        n = len(v.encode("utf-8"))
    except UnicodeEncodeError:  # a lone surrogate from a JSON escape
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: not valid UTF-8") from None
    if n > limit:
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: at most {limit} bytes")
    return v


def _owner(req: dict[str, Any], *, required: bool = False) -> str:
    if required and "owner" not in req:
        raise p.ProtocolError(p.BAD_REQUEST, "owner: string required")
    return _text(req, "owner", "", OWNER_MAX)


def _int(req: dict[str, Any], field: str) -> int:
    v = req.get(field)
    if isinstance(v, bool) or not isinstance(v, int):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: integer required")
    return v


def _num(req: dict[str, Any], field: str) -> float:
    v = req.get(field)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise p.ProtocolError(p.BAD_REQUEST, f"{field}: number required")
    return float(v)


def _namespace(req: dict[str, Any]) -> Namespace:
    raw = req.get("namespace")
    if not isinstance(raw, dict):
        raise p.ProtocolError(p.BAD_REQUEST, "namespace: object required")
    try:
        fields = dict(raw)
        groups = fields.get("kv_groups")
        if not isinstance(groups, list):
            raise TypeError("kv_groups must be a list")
        fields["kv_groups"] = tuple(groups)
        return Namespace(**fields)
    except TypeError as exc:
        raise p.ProtocolError(p.BAD_REQUEST, f"namespace: {exc}") from None


def _groups(req: dict[str, Any]) -> dict[str, int]:
    raw = req.get("groups")
    if not isinstance(raw, dict) or not raw:
        raise p.ProtocolError(p.BAD_REQUEST, "groups: non-empty object required")
    for g, n in raw.items():
        if isinstance(n, bool) or not isinstance(n, int):
            raise p.ProtocolError(p.BAD_REQUEST, f"groups: {g!r} needs an integer byte size")
    return dict(raw)


def _top_args(req: dict[str, Any]) -> tuple[bytes | None, bytes | None, str, int]:
    """``top_reused``'s (model, tenant, view, n), checked."""
    view = req.get("view")
    if view not in ("admin", "holder"):
        raise p.ProtocolError(p.BAD_REQUEST, 'view: "admin" or "holder" required')
    n = req.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= TOP_MAX:
        raise p.ProtocolError(p.BAD_REQUEST, f"n: integer 1 to {TOP_MAX} required")
    model = _opt_hex(req, "model", TAG_BYTES) or None
    tenant = _opt_hex(req, "tenant", TAG_BYTES) or None
    if view == "holder" and tenant is None:
        raise p.ProtocolError(p.BAD_REQUEST, "tenant: required for the holder view")
    return model, tenant, view, n


class _Conn:
    """What one connection has open on the store."""

    def __init__(self) -> None:
        self.saves: set[tuple[bytes, bytes]] = set()
        self.loads: set[int] = set()


class Daemon:
    def __init__(self, store: Store, socket_path: str) -> None:
        self.store = store
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        # released once per connection after its leftovers are cleaned up (tests wait on it)
        self.released = threading.Semaphore(0)

    async def start(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
        old = os.umask(0o177)  # the socket is born 0600
        try:
            self._server = await asyncio.start_unix_server(
                self._serve, path=self.socket_path, limit=p.MAX_LINE
            )
        finally:
            os.umask(old)
        os.chmod(self.socket_path, 0o600)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        for t in list(self._tasks):  # before wait_closed: on 3.12.1+ it waits for every connection
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        conn = _Conn()
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    log.warning("request line over %d bytes; closing connection", p.MAX_LINE)
                    break
                if not line.endswith(b"\n"):
                    break  # EOF (possibly mid-line)
                writer.write(p.encode(await self._reply(conn, line)))
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            await asyncio.shield(self._release(conn))
            if task is not None:
                self._tasks.discard(task)

    async def _release(self, conn: _Conn) -> None:
        if conn.saves or conn.loads:
            log.info(
                "connection gone: aborting %d saves, ending %d loads",
                len(conn.saves),
                len(conn.loads),
            )
        await asyncio.to_thread(self._release_sync, conn)
        self.released.release()

    def _release_sync(self, conn: _Conn) -> None:
        for ns_id, key in conn.saves:
            self._safely(self.store.abort_save, ns_id, key)
        for handle in conn.loads:
            self._safely(self.store.end_load, handle)
        conn.saves.clear()
        conn.loads.clear()

    @staticmethod
    def _safely(fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            log.exception("cleanup after a dropped connection failed")

    async def _reply(self, conn: _Conn, line: bytes) -> dict[str, Any]:
        rid: Any = None
        try:
            req = p.decode(line)
            rid = req.get("id")
            op = req.get("op")
            if not isinstance(op, str) or op not in p.OPS:
                raise p.ProtocolError(p.UNKNOWN_OP, f"unknown op {op!r}")
            if op == "hello":  # on the loop: liveness must not queue behind busy threads
                body = self._dispatch(conn, op, req)
            else:
                body = await asyncio.to_thread(self._dispatch, conn, op, req)
            return {"id": rid, "ok": True, **body}
        except p.ProtocolError as exc:
            return {"id": rid, "ok": False, "error": str(exc)}
        except Exception as exc:
            log.exception("request failed")
            return {"id": rid, "ok": False, "error": f"{p.INTERNAL}: {exc}"}

    def _dispatch(self, conn: _Conn, op: str, req: dict[str, Any]) -> dict[str, Any]:
        s = self.store
        if op == "hello":
            hello: dict[str, Any] = {
                "protocol": p.VERSION,
                "minor": p.MINOR,
                "ram_path": str(s.cfg.ram_path),
                "ram_bytes": s.ram_bytes(),
            }
            pool_error = s.pool_error()
            if pool_error is not None:  # minor 2, only when there is no pool for a reason
                hello["pool_error"] = pool_error
            return hello
        if op == "register_ns":
            nvme = req.get("nvme", True)
            if not isinstance(nvme, bool):
                raise p.ProtocolError(p.BAD_REQUEST, "nvme: boolean required")
            ns, groups = _namespace(req), _groups(req)
            extra: dict[str, Any] = {}  # minor 1; none given is step 1's call, exactly
            owner = _owner(req)
            model = _opt_hex(req, "model", TAG_BYTES)
            role = _text(req, "role", DEFAULT_ROLE, ROLE_MAX)
            rank = req.get("rank", 0)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise p.ProtocolError(p.BAD_REQUEST, "rank: integer >= 0 required")
            if owner:
                extra["owner"] = owner
            if model:
                extra["model"] = model
            if role != DEFAULT_ROLE:
                extra["role"] = role
            if rank:
                extra["rank"] = rank
            staging = req.get("staging", False)  # minor 2
            if not isinstance(staging, bool):
                raise p.ProtocolError(p.BAD_REQUEST, "staging: boolean required")
            if staging:
                extra["staging"] = True
            try:
                ns_id = s.register_namespace(ns, groups, nvme=nvme, **extra)
            except ValueError as exc:
                raise p.ProtocolError(p.BAD_REQUEST, str(exc)) from None
            return {"ns": ns_id.hex()}
        if op == "lookup":
            return {"n": s.lookup(_hex(req, "ns"), _hex_list(req, "keys"))}
        if op == "begin_save":
            ns_id, key = _hex(req, "ns"), _hex(req, "key")
            parent, scope = _hex(req, "parent"), _hex(req, "scope", size=None)
            attrs = {
                "tenant": _opt_hex(req, "tenant", TAG_BYTES),
                "scope_tag": _opt_hex(req, "scope_tag", TAG_BYTES),
                "pos": _opt_hex(req, "pos", KEY_BYTES),  # empty: the store uses ``key``
            }
            attrs = {k: v for k, v in attrs.items() if v}
            depth = req.get("depth")
            if depth is not None and (
                isinstance(depth, bool) or not isinstance(depth, int) or depth < 0
            ):
                raise p.ProtocolError(p.BAD_REQUEST, "depth: int of 0 or more required")
            offsets = s.begin_save(ns_id, key, parent, scope, depth=depth, **attrs)
            if offsets is not None:
                conn.saves.add((ns_id, key))
            return {"offsets": offsets}
        if op in ("commit_save", "abort_save"):
            ns_id, key = _hex(req, "ns"), _hex(req, "key")
            conn.saves.discard((ns_id, key))
            (s.commit_save if op == "commit_save" else s.abort_save)(ns_id, key)
            return {}
        if op == "begin_load":
            ns_id, keys = _hex(req, "ns"), _hex_list(req, "keys")
            load_kw: dict[str, Any] = {}  # none given is the minor 0 call, exactly
            tenant = _opt_hex(req, "tenant", TAG_BYTES)
            if tenant:
                load_kw["tenant"] = tenant
            tag = _text(req, "tag", "", LOAD_TAG_MAX)  # minor 2
            if tag:
                load_kw["tag"] = tag
            handle = s.begin_load(ns_id, keys, **load_kw)
            if handle.slots:
                conn.loads.add(handle.id)
            return {"handle": handle.id, "slots": handle.slots, "nvme": handle.nvme_blocks}
        if op == "end_load":
            hid = _int(req, "handle")
            used = req.get("used")  # minor 2; null is unknown, as missing
            if used is not None and not isinstance(used, bool):
                raise p.ProtocolError(p.BAD_REQUEST, "used: boolean required")
            conn.loads.discard(hid)
            if used is None:
                s.end_load(hid)
            else:
                s.end_load(hid, used=used)
            return {}
        if op == "record_hits":
            tag = req.get("tag")
            if not isinstance(tag, str):
                raise p.ProtocolError(p.BAD_REQUEST, "tag: string required")
            hits = (tag, _int(req, "ram_tokens"), _int(req, "nvme_tokens"), _num(req, "load_s"))
            tenant = _opt_hex(req, "tenant", TAG_BYTES)
            if tenant:
                s.record_hits(*hits, tenant=tenant)
            else:
                s.record_hits(*hits)
            return {}
        if op == "pop_hits":
            tag = req.get("tag")
            if not isinstance(tag, str):
                raise p.ProtocolError(p.BAD_REQUEST, "tag: string required")
            # {} (not null) when nothing is recorded, so a client can tell it from a failure.
            return {"hits": s.pop_hits(tag) or {}}
        if op == "release_owner":
            owner = _owner(req, required=True)
            if not owner:  # it would drop the legacy owner of every minor 0 namespace
                raise p.ProtocolError(p.BAD_REQUEST, "owner: must not be empty")
            return {"released": s.release_owner(owner)}
        if op == "retire_scopes":
            # Every tag is checked before any is retired; the store writes them to disk.
            return {"added": s.retire_scopes(_hex_tags(req, "scope_tags"))}
        if op == "set_policy":
            doc = req.get("policy")
            if not isinstance(doc, dict):
                raise p.ProtocolError(p.BAD_REQUEST, "policy: object required")
            try:
                policy = Policy.from_wire(doc)
            except ValueError as exc:  # the previous policy stays
                raise p.ProtocolError(p.BAD_REQUEST, str(exc)) from None
            s.set_policy(policy)
            return {}
        if op == "top_reused":
            return {"rows": s.top_reused(*_top_args(req))}
        return {"stats": s.stats()}  # op == "stats"


#: Lock file in the NVMe directory: one daemon writes an NVMe store at a time.
NVME_LOCK_NAME = ".lock"


#: The file an operator creates on the mounted disk; with ``--nvme-marker`` the NVMe
#: directory is used only while it exists (a disk that failed to mount leaves a bare folder).
MARKER_NAME = ".lmw-kv-disk"


def nvme_dir_ready(nvme_dir: Path, marker: bool) -> bool:
    """True when the NVMe directory may be used: no marker wanted, or the marker is there."""
    return not marker or (nvme_dir / MARKER_NAME).is_file()


def _marker_reason(nvme_dir: Path) -> str:
    return f"NVMe tier off: {nvme_dir} has no {MARKER_NAME} marker"


#: Where the mount table is read (Linux). Missing elsewhere (macOS): only the st_dev rule.
MOUNTINFO = Path("/proc/self/mountinfo")

#: Filesystem types that are never a dedicated disk: the container's own root (overlay)
#: or memory (tmpfs, ramfs).
NOT_A_DISK_FS = ("overlay", "tmpfs", "ramfs")

#: The share of the NVMe filesystem's size the store always leaves free (spec 9: the
#: disk at most about 85% full): the cap in force is clamped to the room above it.
NVME_FS_MARGIN = 0.15

ROOT_DISK_HINT = "mount a dedicated disk there, or set kv_store_nvme_allow_root_disk"


def _root_dev() -> int:
    return os.stat("/").st_dev


def _unescape_mount(field: str) -> str:
    """A mountinfo path field: spaces, tabs, newlines and backslashes are octal escapes."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def fs_type(path: Path) -> str | None:
    """The filesystem type of the mount holding ``path`` (the longest mount point that is
    a prefix of it, from ``MOUNTINFO``), or None when the table cannot be read."""
    target = str(path.resolve())
    best, best_len = None, -1
    try:
        with open(MOUNTINFO) as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in lines:
        left, _, right = line.partition(" - ")
        fields, rest = left.split(), right.split()
        if len(fields) < 5 or not rest:
            continue
        point = _unescape_mount(fields[4])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) > best_len:
            best, best_len = rest[0], len(point)
    return best


def _existing(path: Path) -> Path:
    """``path``, or its nearest ancestor that exists: where a missing folder would be made."""
    p = path.absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def root_disk_reason(nvme_dir: Path) -> str | None:
    """Why ``nvme_dir`` is not a dedicated disk, or None when it may be one.

    Refused: the root filesystem's device (``st_dev`` of ``/``), or an ``overlay``,
    ``tmpfs`` or ``ramfs`` mount. A missing folder is judged by the folder it would be
    made in. A bind mount of a directory on the host's root disk looks like a dedicated
    disk from inside a container; the mount marker covers that case.
    """
    where = _existing(nvme_dir)
    try:
        same_dev = os.stat(where).st_dev == _root_dev()
    except OSError:
        return None  # the lock (which makes the folder) reports an unusable folder
    kind = fs_type(where)
    if same_dev:
        named = f" ({kind})" if kind else ""
        return f"NVMe tier off: {nvme_dir} is on the root filesystem{named}; {ROOT_DISK_HINT}"
    if kind in NOT_A_DISK_FS:
        return (
            f"NVMe tier off: {nvme_dir} is on a {kind} filesystem, not a dedicated disk; "
            f"{ROOT_DISK_HINT}"
        )
    return None


def _fmt_gb(n: float) -> str:
    text = f"{n / 1_000_000_000:.1f}"
    return text[:-2] if text.endswith(".0") else text


def nvme_cap(nvme_dir: Path, cap_bytes: int) -> tuple[int, str | None, str | None]:
    """(cap in force, note when it was clamped, error when no room is left).

    ``min(cap_bytes, store bytes on disk + free bytes - NVME_FS_MARGIN * size)`` of the
    filesystem holding ``nvme_dir``, floored at 0. Worked out when the tier comes on
    (start and retry), never per write. 0 means NVMe off, and the error says why.
    """
    if cap_bytes <= 0:
        return 0, None, None
    st = os.statvfs(nvme_dir)
    free = int(st.f_bavail * st.f_frsize)
    size = int(st.f_blocks * st.f_frsize)
    on_disk = 0
    for seg in nvme_dir.glob("*/??????.seg"):
        with contextlib.suppress(OSError):
            on_disk += seg.stat().st_size
    room = on_disk + free - int(NVME_FS_MARGIN * size)
    kept = f"the filesystem has {_fmt_gb(free)} GB free and {NVME_FS_MARGIN:.0%} is kept free"
    if room <= 0:
        return 0, None, f"NVMe tier off: {nvme_dir} is too full: {kept}"
    if room >= cap_bytes:
        return cap_bytes, None, None
    return room, f"cap {_fmt_gb(cap_bytes)} GB clamped to {_fmt_gb(room)} GB: {kept}", None


def nvme_refusal(nvme_dir: Path, marker: bool, allow_root_disk: bool) -> str | None:
    """Why the NVMe directory may not be used before its lock is taken, or None."""
    if not nvme_dir_ready(nvme_dir, marker):
        return _marker_reason(nvme_dir)
    if not allow_root_disk:
        return root_disk_reason(nvme_dir)
    return None


#: How often a RAM-only daemon tries again to get the NVMe directory's lock.
NVME_LOCK_RETRY_S = 30.0


def lock_nvme_dir(nvme_dir: Path, *, quiet: bool = False) -> int | None:
    """Take an exclusive, non-blocking ``flock`` on ``<nvme_dir>/.lock``; its fd, or None.

    The lock lives as long as the fd (the daemon process): a killed daemon releases it.
    None when another daemon holds it, for example the other api pod during a rolling
    update that shares the volume, or when the directory cannot be used at all.
    """
    try:
        nvme_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(nvme_dir / NVME_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        if not quiet:
            log.warning("cannot open the NVMe lock in %s: %s", nvme_dir, exc)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def try_take_nvme(
    store: Store, nvme_dir: Path, marker: bool = False, allow_root_disk: bool = False
) -> int | None:
    """One retry of the NVMe lock for a RAM-only store: the lock fd once NVMe is on, or None.

    The same rules as at start, quietly (the start logged the one WARNING): with
    ``marker`` the directory must hold the marker, it must not be on the root disk unless
    ``allow_root_disk`` (no lock, no directory made otherwise), and the cap is clamped to
    the disk's room. Each refusal's reason lands in ``stats.nvme_error``.

    Keep the fd open while the store runs. If the lock is taken but the tier cannot be
    turned on (the directory fails, or the disk has no room), the lock is given back and
    None returned.
    """
    reason = nvme_refusal(nvme_dir, marker, allow_root_disk)
    if reason is not None:
        store.set_nvme_error(reason)
        return None
    fd = lock_nvme_dir(nvme_dir, quiet=True)
    if fd is None:
        store.set_nvme_error(_locked_reason(nvme_dir))
        return None
    try:
        cap, note, error = nvme_cap(nvme_dir, store.cfg.nvme_cap_bytes)
    except OSError as exc:
        cap, note, error = 0, None, _no_room_reading(nvme_dir, exc)
    if error is not None or not store.enable_nvme(nvme_dir, cap, note):
        if error is not None:
            store.set_nvme_error(error)
        os.close(fd)
        return None
    log.info("NVMe tier on: got the lock on %s", nvme_dir)
    return fd


def _no_room_reading(nvme_dir: Path, exc: OSError) -> str:
    return f"NVMe tier off: cannot read the free space of {nvme_dir}: {exc}"


def _locked_reason(nvme_dir: Path) -> str:
    return f"NVMe tier off: {nvme_dir} is locked by another KV store daemon (or unusable)"


async def _retry_nvme(
    store: Store,
    nvme_dir: Path,
    fds: list[int],
    marker: bool = False,
    allow_root_disk: bool = False,
) -> None:
    """Retry the NVMe lock every ``NVME_LOCK_RETRY_S`` until NVMe is on (one flock a try)."""
    while True:
        await asyncio.sleep(NVME_LOCK_RETRY_S)
        take = asyncio.ensure_future(
            asyncio.to_thread(try_take_nvme, store, nvme_dir, marker, allow_root_disk)
        )
        try:
            fd = await asyncio.shield(take)
        except asyncio.CancelledError:
            # Stopped mid-take: the thread runs on and may get the lock. Wait for it so
            # the fd lands in ``fds`` (closed by the caller), never leaked.
            with contextlib.suppress(Exception):
                late = await take
                if late is not None:
                    fds.append(late)
            raise
        if fd is not None:
            fds.append(fd)
            return


def store_config(args: argparse.Namespace) -> tuple[StoreConfig, int | None]:
    """The store's config and the NVMe lock fd (keep it open while the store runs).

    Without the lock the daemon runs RAM-only: two daemons must never write one NVMe store.
    The NVMe directory is also refused (one WARNING, ``nvme_error`` in ``stats``) without
    its marker (``--nvme-marker``), on the root disk (unless ``--nvme-allow-root-disk``)
    and when the disk has no room left above ``NVME_FS_MARGIN``; the cap in force is
    clamped to that room. The daemon tries again every ``NVME_LOCK_RETRY_S``.
    """
    nvme = Path(args.nvme_dir) if args.nvme_dir else None
    lock_fd = None
    marker = bool(getattr(args, "nvme_marker", False))
    allow_root = bool(getattr(args, "nvme_allow_root_disk", False))
    cap = int(args.nvme_cap_gb * 1_000_000_000)
    error: str | None = None
    note: str | None = None
    if nvme is not None:
        error = nvme_refusal(nvme, marker, allow_root)
        if error is not None:
            log.warning("%s", error)
            nvme = None
    if nvme is not None:
        lock_fd = lock_nvme_dir(nvme)
        if lock_fd is None:
            error = _locked_reason(nvme)
            log.warning(
                "NVMe tier off: %s is locked by another KV store daemon (or unusable); "
                "running RAM-only and trying the lock again every %.0f s",
                nvme,
                NVME_LOCK_RETRY_S,
            )
            nvme = None
    if nvme is not None and lock_fd is not None:
        try:
            cap, note, error = nvme_cap(nvme, cap)
        except OSError as exc:
            error = _no_room_reading(nvme, exc)
        if error is not None:
            log.warning("%s", error)
            os.close(lock_fd)  # given back: the retry (or another daemon) may take it
            lock_fd, nvme, cap = None, None, int(args.nvme_cap_gb * 1_000_000_000)
            note = None
    cfg = StoreConfig(
        ram_path=Path(args.ram_path),
        ram_bytes=int(args.ram_gib * (1 << 30)),
        nvme_dir=nvme,
        nvme_cap_bytes=cap,
        engine_workers=max(0, int(getattr(args, "engine_workers", 0) or 0)),
        nvme_error=error,
        nvme_cap_note=note,
        nvme_cap_asked=int(args.nvme_cap_gb * 1_000_000_000),
    )
    return cfg, lock_fd


#: Seconds between two writes of the reuse counters to ``reuse.db``.
REUSE_PERSIST_S = float(_REUSE_PERSIST_S)


async def _persist_reuse_loop(store: Store) -> None:
    """Write the reuse counters every ``REUSE_PERSIST_S`` in a worker thread, once more at stop."""

    async def once() -> None:
        try:
            await asyncio.to_thread(store.persist_reuse)
        except Exception:
            log.exception("persisting the reuse counters failed")

    try:
        while True:
            await asyncio.sleep(REUSE_PERSIST_S)
            await once()
    except asyncio.CancelledError:
        await once()
        raise


async def _run(args: argparse.Namespace) -> None:
    # Armed before the store is built: committing the pool takes seconds, and a SIGTERM
    # in that window with the default action would leave the pool file in /dev/shm.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    cfg, lock_fd = store_config(args)
    late_fds: list[int] = []  # the NVMe lock, when it was taken after start
    retry: asyncio.Task[None] | None = None
    persist: asyncio.Task[None] | None = None
    try:
        store = await asyncio.to_thread(Store, cfg)
        try:
            daemon = Daemon(store, args.socket)
            try:
                if stop.is_set():
                    log.info("stopped while the RAM pool was being committed")
                    return
                await daemon.start()
                if args.nvme_dir and cfg.nvme_dir is None:
                    retry = asyncio.create_task(
                        _retry_nvme(
                            store,
                            Path(args.nvme_dir),
                            late_fds,
                            bool(getattr(args, "nvme_marker", False)),
                            bool(getattr(args, "nvme_allow_root_disk", False)),
                        )
                    )
                if args.nvme_dir:
                    persist = asyncio.create_task(_persist_reuse_loop(store))
                log.info("serving on %s", args.socket)
                await stop.wait()
            finally:
                if persist is not None:
                    persist.cancel()
                    await asyncio.gather(persist, return_exceptions=True)
                if retry is not None:
                    retry.cancel()
                    await asyncio.gather(retry, return_exceptions=True)
                await daemon.stop()
        finally:
            await asyncio.to_thread(store.close)  # unlinks the pool file
    finally:
        for fd in [lock_fd, *late_fds]:
            if fd is not None:
                os.close(fd)  # releases the NVMe lock


#: ``prctl`` option: the signal this process gets when its parent dies (linux/prctl.h).
PR_SET_PDEATHSIG = 1


def die_with_parent(parent_pid: int | None) -> bool:
    """On Linux, get SIGTERM when the api that started us dies; True when armed.

    Without it an api killed by the OOM killer leaves this daemon running, holding the
    unlinked pool's pages and the NVMe lock. Exits at once if the parent is already
    gone (it died before the signal was armed). A no-op elsewhere.
    """
    if parent_pid is None or not sys.platform.startswith("linux"):
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
        log.warning("cannot arm the parent-death signal: errno %d", ctypes.get_errno())
        return False
    if os.getppid() != parent_pid:
        raise SystemExit("the api that started this daemon is gone")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="lmw_kv.daemon", description="LM Warden KV store daemon")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--ram-path", required=True)
    ap.add_argument("--ram-gib", type=float, required=True)
    ap.add_argument("--nvme-dir")
    ap.add_argument("--nvme-cap-gb", type=float, default=0.0)
    ap.add_argument(
        "--nvme-marker",
        action="store_true",
        help=f"use --nvme-dir only while <dir>/{MARKER_NAME} exists (mount guard)",
    )
    ap.add_argument(
        "--nvme-allow-root-disk",
        action="store_true",
        help="use --nvme-dir even on the root filesystem, an overlay, tmpfs or ramfs",
    )
    ap.add_argument(
        "--engine-workers",
        type=int,
        default=0,
        help="engine workers whose host memory the pool leaves free",
    )
    ap.add_argument("--parent-pid", type=int, help="exit when this process (the api) dies")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    die_with_parent(args.parent_pid)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
