import hashlib
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import aiosqlite

from app.auth.bearer import generate_admin_token, generate_bearer_token

_SQLITE_UTC_FMT = "%Y-%m-%d %H:%M:%S"


def sqlite_utc_now() -> str:
    """Return the current UTC time as a SQLite-native naive UTC string."""
    return datetime.now(UTC).strftime(_SQLITE_UTC_FMT)


def sqlite_utc_in(delta: timedelta) -> str:
    """Return UTC now + delta as a SQLite-native naive UTC string."""
    return (datetime.now(UTC) + delta).strftime(_SQLITE_UTC_FMT)


# Column order MUST match the SELECT lists in find_by_plaintext / list_all /
# get below. Adding a column? Add it at the END of both this dataclass AND
# the SELECT clauses (so old positional tuples still unpack correctly), then
# update the create() INSERT if it needs a non-default value.
@dataclass
class TokenRow:
    id: str
    name: str
    prefix: str
    scope: str
    allowed_models: str | None
    rate_limit_rpm: int | None
    rate_limit_tpm: int | None
    revoked_at: str | None
    last_used_at: str | None
    created_at: str
    expires_at: str | None
    rotated_at: str | None
    rotated_from: str | None
    # UNUSED since per-token rate limits were removed (2026-09). The column
    # stays in api_tokens (SQLite cannot drop it cheaply) and the field stays
    # here, in its original position, so ``priority`` and ``paused_at`` keep
    # their positional indexes (see the rule above). Nothing reads or writes
    # it any more; do not expose it.
    rate_limit_tps: int | None = None
    # S5 (#104) — STRICT scheduler priority 0..9 (9 always served first;
    # starvation of priority-0 tokens is by design and documented in the UI
    # tooltip).
    priority: int = 5
    # Token details page (0033) -- when the operator paused this key; None =
    # not paused. require_bearer answers a paused key with 403 "token paused".
    paused_at: str | None = None
    # Admin tokens (0037) -- the user an admin token acts as; NULL on every
    # inference row. Appended last, per the column-order rule above.
    created_by: str | None = None


_SELECT_COLS = (
    "id, name, prefix, scope, allowed_models, rate_limit_rpm, rate_limit_tpm, "
    "revoked_at, last_used_at, created_at, expires_at, rotated_at, rotated_from, "
    "rate_limit_tps, priority, paused_at, created_by"
)


def hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class _Unset:
    """Sentinel marker for "field not provided" in TokenRepo.update().

    Distinct from None so a nullable field can be explicitly cleared without
    us treating it as 'leave alone'. Defined above ``TokenRepo``
    so the method annotations can reference the class without forward
    refs / TYPE_CHECKING gymnastics.
    """

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_UNSET = _Unset()


# When an admin token stopped working: the earlier of its revocation and its
# expiry, counting only those already past :now. NULL while it still works (a
# FUTURE revoked_at is a refresh's grace window). Used by TokenRepo.list_admin.
_DEAD_AT_SQL = (
    "CASE"
    " WHEN revoked_at IS NOT NULL AND revoked_at <= :now"
    "  AND expires_at IS NOT NULL AND expires_at <= :now"
    "  THEN MIN(revoked_at, expires_at)"
    " WHEN revoked_at IS NOT NULL AND revoked_at <= :now THEN revoked_at"
    " WHEN expires_at IS NOT NULL AND expires_at <= :now THEN expires_at"
    " END"
)


class TokenRepo:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def create(
        self,
        token_id: str,
        name: str,
        plaintext: str,
        scope: str = "inference",
        allowed_models: list[str] | None = None,
        expires_in_days: int = 365,
        priority: int = 5,
        created_by: str | None = None,
    ) -> None:
        """Insert a new API token row and commit.

        expires_in_days=0 (or any non-positive value) means 'never expires'.
        priority must be 0..9 (DB CHECK trigger enforces this, but we validate
        at the Pydantic layer too so users get a 422 instead of a 500).
        created_by is the issuing user of an admin token; None for inference.
        """
        prefix = plaintext[:8]
        if expires_in_days > 0:
            expires_at = sqlite_utc_in(timedelta(days=expires_in_days))
        else:
            expires_at = None
        await self.db.execute(
            "INSERT INTO api_tokens"
            "(id, name, prefix, hash, scope, allowed_models, expires_at, "
            " priority, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                token_id, name, prefix, hash_token(plaintext), scope,
                ",".join(allowed_models) if allowed_models else None,
                expires_at,
                priority,
                created_by,
            ),
        )
        await self.db.commit()

    async def _next_old_suffix(self, base_name: str, scope: str = "inference") -> int:
        """Return the next available ``N`` such that ``"{base_name} (old N)"``
        is unused.

        Issue #150 — when an operator rotates ``prod-bot`` we rename the old
        row to ``prod-bot (old 1)`` (or ``(old 2)``, ``(old 3)`` … if previous
        rotations already burnt the lower numbers) and mint a fresh row that
        keeps the original ``prod-bot`` name. The numbering is monotonic and
        never reuses gaps — a deleted ``(old 1)`` does NOT make ``N=1`` free
        again, because the operator's mental model is "the next slot", not
        "fill holes" (consider: a `(old 1)` deleted after audit could be
        confused with a current rotation if a future rotation reused the
        slot).

        Counted per scope: an admin ``ci`` and an inference ``ci`` number
        their ``(old N)`` independently.

        Implementation: pull every row whose name LIKE ``"{base} (old %)"``,
        parse the integer between ``(old`` and ``)`` with a strict regex
        (rejects non-int garbage and stray spacing — defence against a hand-
        edited DB), then return ``max(N) + 1`` with default ``1`` when no
        prior `(old N)` exists. Case-sensitive; SQLite default collation is
        BINARY for TEXT and the rest of the codebase treats names as
        case-sensitive so we don't normalise here.
        """
        # LIKE escape: '%' and '_' would be wildcards inside base_name. Use
        # ESCAPE so a name containing those characters still matches literally.
        like_pat = base_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cur = await self.db.execute(
            "SELECT name FROM api_tokens WHERE name LIKE ? ESCAPE '\\' AND scope = ?",
            (f"{like_pat} (old %)", scope),
        )
        rows = await cur.fetchall()
        suffix_re = re.compile(
            r"^" + re.escape(base_name) + r" \(old (\d+)\)$"
        )
        seen: list[int] = []
        for (name,) in rows:
            m = suffix_re.match(name)
            if m:
                seen.append(int(m.group(1)))
        return (max(seen) + 1) if seen else 1

    async def rotate(
        self,
        old_id: str,
        grace_hours: int = 24,
        expires_in_days: int | None = None,
    ) -> tuple[str, str, str]:
        """Rename the predecessor to ``"{name} (old N)"`` and mint a fresh
        successor that keeps the original ``name`` (#150).

        All writes — predecessor SELECT, predecessor RENAME, successor
        INSERT, predecessor UPDATE (rotated_at / revoked_at) — are part of
        a single transaction so a crash cannot leave rotation half-applied.

        Returns ``(new_id, new_plaintext, renamed_to)`` where ``renamed_to``
        is the predecessor's new ``"{original} (old N)"`` name so the route
        can surface it to the UI in one round-trip.

        Behaviour:
          * If the predecessor was already rotated (``rotated_at`` is not
            null) we raise ``ValueError("already rotated")`` — the route
            translates that into 409. Idempotent rotation would silently
            allocate ``(old 2)``, ``(old 3)`` … on accidental double-clicks
            and is footgun-y enough to forbid.
          * The successor INHERITS ``priority`` — operators do not want a
            rotation to silently change scheduler behaviour. To change it,
            PATCH the successor.
          * ``grace_hours`` schedules the predecessor's ``revoked_at`` (in
            the future when >0). Existing callers keep working through the
            window — see ``tests/integration/test_token_rotate_grace.py``.

        expires_in_days semantics:
          - None (default) → inherit the predecessor's expires_at (may be NULL).
          - 0              → successor never expires.
          - >0             → expires now + N days.
        """
        # Look up predecessor for inheritable fields + the current name.
        cur = await self.db.execute(
            "SELECT name, expires_at, priority, rotated_at, scope, created_by "
            "FROM api_tokens WHERE id = ?",
            (old_id,),
        )
        row = await cur.fetchone()
        if row is None:
            # rotate() is only called from routes after a list_all() existence
            # check, but guard anyway so future direct callers see a clean error.
            raise ValueError(f"token {old_id} not found")
        pred_name, pred_expires, pred_priority, pred_rotated_at, pred_scope, pred_owner = row

        if pred_rotated_at is not None:
            # Predecessor was already rotated. Rotating again would chain
            # `(old 1) → (old 2)` style renames into a meaningless ladder
            # (the row's secret has the same compromise risk regardless of
            # how many rotations it survives). Bounce to the route which
            # returns 409.
            raise ValueError("already rotated")

        # The successor is the same KIND of key, for the same owner: an admin
        # token refreshes into an admin token (spec 2026-09-19, decision 6).
        new_plaintext = (
            generate_admin_token() if pred_scope == "admin" else generate_bearer_token()
        )
        new_id = secrets.token_hex(16)
        new_prefix = new_plaintext[:8]

        renamed_to = (
            f"{pred_name} (old {await self._next_old_suffix(pred_name, pred_scope)})"
        )

        if expires_in_days is None:
            new_expires_at = pred_expires
        elif expires_in_days > 0:
            new_expires_at = sqlite_utc_in(timedelta(days=expires_in_days))
        else:
            new_expires_at = None

        # 1) Rename predecessor BEFORE the insert so a future UNIQUE
        # constraint on `name` (none today, but cheap to be defensive)
        # could not collide between predecessor + successor in the same
        # transaction. Today this is purely a code-clarity ordering.
        await self.db.execute(
            "UPDATE api_tokens SET name = ? WHERE id = ?",
            (renamed_to, old_id),
        )

        # 2) Insert successor with the ORIGINAL name + rotated_from pointer.
        await self.db.execute(
            "INSERT INTO api_tokens"
            "(id, name, prefix, hash, scope, rotated_from, expires_at, "
            " priority, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new_id, pred_name, new_prefix, hash_token(new_plaintext),
                pred_scope, old_id, new_expires_at,
                pred_priority, pred_owner,
            ),
        )

        # 3) Mark predecessor as rotated + schedule its revocation. Guarded
        # with `AND rotated_at IS NULL` (M7, final-review 2026-09-19): the
        # SELECT above and this UPDATE are not atomic, so a second rotate()
        # racing this one could pass the `pred_rotated_at is not None` check
        # too and reach here concurrently. Without the guard, both commit
        # and the predecessor ends up with two successors -- a forked
        # chain. With it, whichever UPDATE loses the race affects 0 rows;
        # we roll back its rename + INSERT (steps 1-2 above, never
        # committed) and raise the same "already rotated" ValueError the
        # top-of-function check raises, which the routes already map to a
        # 409.
        rotated_at = sqlite_utc_now()
        revoked_at = sqlite_utc_in(timedelta(hours=grace_hours))
        cur = await self.db.execute(
            "UPDATE api_tokens SET rotated_at = ?, revoked_at = ? "
            "WHERE id = ? AND rotated_at IS NULL",
            (rotated_at, revoked_at, old_id),
        )
        if cur.rowcount != 1:
            await self.db.rollback()
            raise ValueError("already rotated")

        await self.db.commit()
        return new_id, new_plaintext, renamed_to

    async def find_by_plaintext(self, plaintext: str) -> TokenRow | None:
        h = hash_token(plaintext)
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens WHERE hash = ?",
            (h,),
        )
        r = await cur.fetchone()
        return TokenRow(*r) if r else None

    async def get(self, token_id: str) -> TokenRow | None:
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens WHERE id = ?",
            (token_id,),
        )
        r = await cur.fetchone()
        return TokenRow(*r) if r else None

    async def list_all(self, *, scope: str = "inference") -> list[TokenRow]:
        """Every row of one scope, newest first. Inference by default: the
        admin tokens sharing this table (0037) are not API keys, and the one
        caller (the chat playground's duplicate sweep) must never touch them."""
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM api_tokens WHERE scope = ? "
            "ORDER BY created_at DESC",
            (scope,),
        )
        return [TokenRow(*r) for r in await cur.fetchall()]

    async def list_admin(self, *, now: str, since: str) -> list[TokenRow]:
        """Settings -> Admin tokens: the live admin tokens (active, or inside
        a refresh's grace window) first, then those that stopped working at or
        after ``since``; newest ``created_at`` first within each group.

        ``now`` and ``since`` are SQLite UTC strings. A token "stopped working"
        at the earlier of a past revoked_at and a past expires_at -- a future
        revoked_at is a grace window, still live.
        """
        cur = await self.db.execute(
            f"SELECT {_SELECT_COLS} FROM ("
            f" SELECT {_SELECT_COLS}, {_DEAD_AT_SQL} AS dead_at"
            " FROM api_tokens INDEXED BY idx_api_tokens_scope WHERE scope = 'admin'"
            ") WHERE dead_at IS NULL OR dead_at >= :since"
            " ORDER BY dead_at IS NOT NULL, created_at DESC, id DESC",
            {"now": now, "since": since},
        )
        return [TokenRow(*r) for r in await cur.fetchall()]

    async def revoke(self, token_id: str) -> None:
        await self.db.execute(
            "UPDATE api_tokens SET revoked_at = datetime('now') WHERE id = ?", (token_id,)
        )
        await self.db.commit()

    # How stale last_used_at may get before a request rewrites it.
    LAST_USED_GRANULARITY_S = 60

    async def touch_last_used(self, token_id: str) -> None:
        """Stamp ``last_used_at``, at most once a minute per key.

        This runs on every proxied request. Since 0035 indexes last_used_at
        (the list sorts by it), each rewrite dirties an index page as well as
        the row and pays a WAL append + fsync -- about 4 KB per request for a
        busy key, against nothing when the value is left alone. So a key used
        within the last LAST_USED_GRANULARITY_S seconds keeps its stamp: the
        column reads "used at most a minute before this", which is all the
        list and the token page ("2 min ago") show.
        """
        await self.db.execute(
            "UPDATE api_tokens SET last_used_at = datetime('now') WHERE id = ?"
            " AND (last_used_at IS NULL OR last_used_at < datetime('now', ?))",
            (token_id, f"-{self.LAST_USED_GRANULARITY_S} seconds"),
        )
        await self.db.commit()

    async def delete(self, token_id: str) -> bool:
        cur = await self.db.execute(
            "DELETE FROM api_tokens WHERE id = ?", (token_id,)
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def update(
        self,
        token_id: str,
        *,
        name: "str | _Unset" = _UNSET,
        priority: "int | _Unset" = _UNSET,
        paused: "bool | _Unset" = _UNSET,
    ) -> bool:
        """PATCH-style update for the operator-editable fields.

        Pass _UNSET (the default) to leave a field untouched. ``name`` and
        ``priority`` are NOT NULL in the schema; pass a value.

        ``paused=True`` stamps ``paused_at`` with the current UTC time only if
        it is not already set, so a repeated pause keeps the ORIGINAL time
        (the page says "paused since ..."). ``paused=False`` clears it. The
        rule that a dead key cannot be paused lives in the route, which has
        the row in hand; this method writes what it is told.

        Returns True if a row was updated, False if token_id was unknown. A
        call that sets nothing returns True without touching the database.
        """
        sets: list[str] = []
        params: list[object] = []
        if not isinstance(name, _Unset):
            sets.append("name = ?")
            params.append(name)
        if not isinstance(priority, _Unset):
            sets.append("priority = ?")
            params.append(priority)
        if not isinstance(paused, _Unset):
            if paused:
                sets.append("paused_at = COALESCE(paused_at, ?)")
                params.append(sqlite_utc_now())
            else:
                sets.append("paused_at = NULL")
        if not sets:
            return True  # noop PATCH — treat as success (idempotent)
        params.append(token_id)
        cur = await self.db.execute(
            f"UPDATE api_tokens SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def lineage(self, token_id: str) -> list[TokenRow]:
        """The rotation chain ``token_id`` belongs to, OLDEST first.

        Walks ``rotated_from`` backwards for the predecessors, then forwards
        through the row whose ``rotated_from`` is the current one for the
        successors -- rotate() refuses to rotate a row twice, so each row has
        at most one. Returns [] for an unknown id.

        A deleted predecessor ends the backward walk (``rotated_from`` is ON
        DELETE SET NULL, and with foreign keys off the lookup simply misses).
        ``seen`` bounds both walks so a hand-edited cycle terminates.
        """
        me = await self.get(token_id)
        if me is None:
            return []
        seen = {me.id}
        earlier: list[TokenRow] = []
        cur = me
        while cur.rotated_from is not None and cur.rotated_from not in seen:
            prev = await self.get(cur.rotated_from)
            if prev is None:
                break
            seen.add(prev.id)
            earlier.append(prev)
            cur = prev
        later: list[TokenRow] = []
        cur = me
        while True:
            c = await self.db.execute(
                f"SELECT {_SELECT_COLS} FROM api_tokens WHERE rotated_from = ? "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                (cur.id,),
            )
            r = await c.fetchone()
            if r is None or r[0] in seen:
                break
            nxt = TokenRow(*r)
            seen.add(nxt.id)
            later.append(nxt)
            cur = nxt
        return [*reversed(earlier), me, *later]


# ---------------------------------------------------------------------------
# GET /api/tokens: one sorted page, built for millions of rows.
#
# Two statements per page, whatever the page size: the page itself (rows +
# successor + 24h usage) and the counts. Nothing in Python walks the table.
# ---------------------------------------------------------------------------

# The list hides a key that was revoked WITHOUT being rotated; a rotated
# predecessor stays visible for the whole of its grace window and after it.
TOKEN_LIST_VISIBLE_SQL = "(t.revoked_at IS NULL OR t.rotated_at IS NOT NULL)"

# The status badge's rank, the same ladder as the UI's deriveStatus
# (token-row.tsx) and _enrich: Paused -> Revoked -> Expired -> Grace ->
# Expiring soon -> Active. "Revoked" is a rotated key whose grace is over (and
# a plainly revoked one, which the list hides) plus "Rotated (orphan)" -- the
# red rotated badges. The CASE tests run in deriveStatus's order, so a key that
# is both expired and past its grace ranks as Expired, the badge it shows. Every
# comparison uses the one ``:now`` the request captured.
TOKEN_STATUS_RANK_SQL = (
    "CASE"
    " WHEN t.paused_at IS NOT NULL THEN 0"
    " WHEN t.revoked_at IS NOT NULL AND t.rotated_at IS NULL THEN 1"
    " WHEN t.expires_at IS NOT NULL AND t.expires_at <= :now THEN 2"
    " WHEN t.rotated_at IS NOT NULL AND NOT EXISTS"
    "  (SELECT 1 FROM api_tokens s WHERE s.rotated_from = t.id) THEN 1"
    " WHEN t.rotated_at IS NOT NULL AND t.revoked_at <= :now THEN 1"
    " WHEN t.rotated_at IS NOT NULL THEN 3"
    " WHEN t.expires_at IS NOT NULL AND t.expires_at <= :in30 THEN 4"
    " ELSE 5 END"
)


@dataclass(frozen=True)
class _SortKey:
    expr: str  # over the api_tokens alias ``t`` (and ``u`` for usage_24h)
    collate: str = ""  # appended to the key in BOTH the inner and outer ORDER BY
    nullable: bool = False  # adds NULLS LAST -- never-used / never-expires sort last


# Every key but usage_24h and status is served by an index from 0035 walked
# in either direction, so a page reads only
# offset + limit index entries. usage_24h and status are computed per row and
# need a scan with a bounded top-N sorter.
TOKEN_SORT_KEYS: dict[str, _SortKey] = {
    "name": _SortKey("t.name", collate=" COLLATE NOCASE"),
    "prefix": _SortKey("t.prefix"),
    "created": _SortKey("t.created_at"),
    "expires": _SortKey("t.expires_at", nullable=True),
    "last_used": _SortKey("t.last_used_at", nullable=True),
    "priority": _SortKey("t.priority"),
    "usage_24h": _SortKey("COALESCE(u.tokens, 0)"),
    "status": _SortKey(TOKEN_STATUS_RANK_SQL),
}

# Only the usage_24h sort joins every token's 24h total into the ORDER BY
# query. idx_token_usage_minute_minute bounds it to the last day's buckets.
_USAGE_SORT_JOIN = (
    " LEFT JOIN (SELECT token_id, SUM(prompt_tokens) + SUM(completion_tokens) AS tokens"
    "  FROM token_usage_minute WHERE minute >= :since AND minute < :until"
    "  GROUP BY token_id) u ON u.token_id = t.id"
)


@dataclass
class TokenListEntry:
    row: TokenRow
    successor_id: str | None
    usage_24h: tuple[int, int, int]  # (requests, prompt_tokens, completion_tokens)


@dataclass
class TokenPage:
    entries: list[TokenListEntry]
    total: int  # visible tokens, all pages
    near_expiry: int  # visible tokens expiring within 30 days, not yet expired


def _order_by(key: _SortKey, k: str, id_col: str, desc: bool) -> str:
    d = " DESC" if desc else " ASC"
    nulls = " NULLS LAST" if key.nullable else ""
    return f"{k}{key.collate}{d}{nulls}, {id_col}{d}"


# Name search: a case-insensitive substring match. SQLite's LIKE already folds
# ASCII case; the pattern escapes its own wildcards (see like_contains) so a
# name search for "50%_off" is literal. A leading-% LIKE cannot seek a B-tree,
# so a search scans the visible rows -- see documents/API.md for the numbers
# and the FTS5 trigram option if that ever becomes too slow.
_NAME_SEARCH_SQL = " AND t.name LIKE :q ESCAPE '\\'"


def like_contains(needle: str) -> str:
    """A LIKE pattern matching ``needle`` anywhere, its own %, _ and \\ literal."""
    esc = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


# is_near_expiry, the ``near_expiry`` count and the ``near_expiry=1`` filter
# are one predicate: expires within 30 days and has not expired yet.
NEAR_EXPIRY_SQL = "t.expires_at > :now AND t.expires_at <= :in30"

# Admin tokens (scope 'admin', 0037) share api_tokens but never appear in the
# inference list, its counts, or any /api/tokens/{id} route (spec 2026-09-19,
# decision 2). The unary + keeps this term off every index: the page CTE must
# keep walking its 0035 sort index, not switch to 0037's idx_api_tokens_scope
# and sort the result.
INFERENCE_ONLY_SQL = "+t.scope = 'inference'"


def _visible_where(search: bool, expiring: bool = False) -> str:
    return (
        TOKEN_LIST_VISIBLE_SQL
        + f" AND {INFERENCE_ONLY_SQL}"
        + (_NAME_SEARCH_SQL if search else "")
        + (f" AND {NEAR_EXPIRY_SQL}" if expiring else "")
    )


def token_page_sql(
    sort: str, desc: bool, search: bool = False, expiring: bool = False,
) -> str:
    """The page statement for one sort order (exposed for EXPLAIN tests).

    The ``page`` CTE picks the ids -- ORDER BY, LIMIT and OFFSET over the
    visible set, touching nothing but the sort index for most keys. The outer
    query then decorates just those rows: the full column list, the successor
    (idx_api_tokens_rotated_from) and the 24h usage summed through
    token_usage_minute's (token_id, minute) primary key for the page's ids
    only. It re-sorts the page by the key the CTE carried out.
    """
    key = TOKEN_SORT_KEYS[sort]
    join = _USAGE_SORT_JOIN if sort == "usage_24h" else ""
    cols = ", ".join(f"t.{c.strip()}" for c in _SELECT_COLS.split(","))
    return (
        "WITH page AS ("
        f" SELECT t.id AS id, {key.expr} AS k FROM api_tokens t{join}"
        f" WHERE {_visible_where(search, expiring)}"
        # By the alias, so a computed key (the status CASE and its
        # EXISTS) is evaluated once per row, not again for the sort.
        f" ORDER BY {_order_by(key, 'k', 't.id', desc)}"
        " LIMIT :limit OFFSET :offset"
        ")"
        f" SELECT {cols},"
        " (SELECT s.id FROM api_tokens s WHERE s.rotated_from = t.id"
        "  ORDER BY s.created_at ASC, s.id ASC LIMIT 1),"
        " COALESCE(us.r, 0), COALESCE(us.p, 0), COALESCE(us.c, 0)"
        " FROM page JOIN api_tokens t ON t.id = page.id"
        " LEFT JOIN (SELECT token_id, SUM(requests) AS r, SUM(prompt_tokens) AS p,"
        "  SUM(completion_tokens) AS c FROM token_usage_minute"
        "  WHERE token_id IN (SELECT id FROM page) AND minute >= :since AND minute < :until"
        "  GROUP BY token_id) us ON us.token_id = t.id"
        f" ORDER BY {_order_by(key, 'page.k', 'page.id', desc)}"
    )


# The keys the list hides -- the negation of TOKEN_LIST_VISIBLE_SQL, written
# exactly as 0035's partial idx_api_tokens_hidden is so the planner uses it.
_HIDDEN_SQL = "t.revoked_at IS NOT NULL AND t.rotated_at IS NULL"
_HIDDEN_FROM = "api_tokens t INDEXED BY idx_api_tokens_hidden"

# The admin keys the visibility rule would show, subtracted from the unsearched
# all-rows arithmetic below. There are a handful; idx_api_tokens_scope (0037)
# seeks exactly them.
_ADMIN_VISIBLE_SQL = f"t.scope = 'admin' AND {TOKEN_LIST_VISIBLE_SQL}"
_ADMIN_FROM = "api_tokens t INDEXED BY idx_api_tokens_scope"


def token_counts_sql(search: bool = False, expiring: bool = False) -> str:
    """``(total, near_expiry)`` over the visible -- searched, filtered -- set.

    Unsearched, both are "all rows" minus "hidden rows": COUNT(*) walks the
    smallest index, the near-expiry range is a covering count on the expires
    index, and the hidden keys come from 0035's partial, covering
    idx_api_tokens_hidden -- no table row is read to test the visibility
    rule (a few ms at a million keys, against ~50 ms for a filtered scan). A
    search has to read every name anyway, so it counts the visible matches
    directly. With the near-expiry filter the total IS the near-expiry count.

    INDEXED BY because left alone the planner prefers 0007's
    idx_api_tokens_revoked, which is not covering and costs a table lookup
    per hidden key. The migration guarantees the index exists.

    Admin rows are subtracted the same way, through idx_api_tokens_scope: all
    rows − hidden rows (both scopes) − visible admin rows = visible
    inference rows.
    """
    window = NEAR_EXPIRY_SQL
    if search:
        where = _visible_where(True, expiring)
        return (
            f"SELECT (SELECT COUNT(*) FROM api_tokens t WHERE {where}),"
            f" (SELECT COUNT(*) FROM api_tokens t WHERE {where} AND {window})"
        )
    near = (
        f"(SELECT COUNT(*) FROM api_tokens t WHERE {window})"
        f" - (SELECT COUNT(*) FROM {_HIDDEN_FROM} WHERE {_HIDDEN_SQL} AND {window})"
        f" - (SELECT COUNT(*) FROM {_ADMIN_FROM} WHERE {_ADMIN_VISIBLE_SQL} AND {window})"
    )
    if expiring:
        return f"SELECT {near}, {near}"
    return (
        "SELECT (SELECT COUNT(*) FROM api_tokens)"
        f" - (SELECT COUNT(*) FROM {_HIDDEN_FROM} WHERE {_HIDDEN_SQL})"
        f" - (SELECT COUNT(*) FROM {_ADMIN_FROM} WHERE {_ADMIN_VISIBLE_SQL}),"
        f" {near}"
    )


async def list_token_page(
    db: aiosqlite.Connection,
    *,
    sort: str,
    desc: bool,
    limit: int,
    offset: int,
    now_str: str,
    in_30d_str: str,
    since_minute: int,
    until_minute: int,
    q: str = "",
    near_expiry_only: bool = False,
) -> TokenPage:
    """One page of the token list, plus the list-wide counts.

    ``sort`` is a TOKEN_SORT_KEYS key (the route validates it). ``id`` breaks
    every tie in the same direction, so a page boundary never splits or
    repeats rows. ``now_str`` / ``in_30d_str`` must be the instants the caller
    also hands to ``_enrich`` so the rank and the badge agree. Usage covers
    minutes ``[since_minute, until_minute)``. A non-empty ``q`` keeps only
    names containing it, case-insensitively; ``near_expiry_only`` keeps only
    keys expiring within 30 days and not yet expired. The counts honour both.
    """
    search = q != ""
    params: dict[str, object] = {
        "now": now_str,
        "in30": in_30d_str,
        "since": since_minute,
        "until": until_minute,
        "limit": limit,
        "offset": offset,
    }
    count_params: dict[str, object] = {"now": now_str, "in30": in_30d_str}
    if search:
        params["q"] = count_params["q"] = like_contains(q)
    cur = await db.execute(token_page_sql(sort, desc, search, near_expiry_only), params)
    entries: list[TokenListEntry] = []
    width = len(_SELECT_COLS.split(","))
    for r in await cur.fetchall():
        entries.append(
            TokenListEntry(
                row=TokenRow(*r[:width]),
                successor_id=r[width],
                usage_24h=(int(r[width + 1]), int(r[width + 2]), int(r[width + 3])),
            )
        )
    cur = await db.execute(token_counts_sql(search, near_expiry_only), count_params)
    counts = await cur.fetchone()
    total, near = (int(counts[0]), int(counts[1])) if counts else (0, 0)
    return TokenPage(entries=entries, total=total, near_expiry=near)


class TokenUsageRepo:
    """Per-token minute-bucket usage rollup (token_usage_minute).

    Counted alongside the existing /counters + /model_samples writes in the
    proxy success path. The rollup is the source for GET /api/tokens/{id}/usage.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def add(
        self,
        token_id: str,
        minute: int,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        commit: bool = True,
    ) -> None:
        # ON CONFLICT works here because (token_id, minute) is a real composite
        # PK with both columns NOT NULL — unlike the counters table that has
        # to dance around the NULL-token_id case.
        await self.db.execute(
            "INSERT INTO token_usage_minute"
            "(token_id, minute, requests, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, 1, ?, ?) "
            "ON CONFLICT(token_id, minute) DO UPDATE SET "
            "  requests = requests + 1, "
            "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "  completion_tokens = completion_tokens + excluded.completion_tokens",
            (token_id, minute, prompt_tokens, completion_tokens),
        )
        if commit:
            await self.db.commit()

    async def range(
        self,
        token_id: str,
        since_minute: int,
        until_minute: int,
    ) -> list[tuple[int, int, int, int]]:
        """Return (minute, requests, prompt_tokens, completion_tokens) rows
        for [since_minute, until_minute) in ascending order.

        Caller pre-computes the minute boundaries from the requested range
        (e.g. 24h, 1h) — this keeps the repo free of clock dependencies and
        the query plan trivially predictable via idx_token_usage_minute_minute.
        """
        cur = await self.db.execute(
            "SELECT minute, requests, prompt_tokens, completion_tokens "
            "FROM token_usage_minute "
            "WHERE token_id = ? AND minute >= ? AND minute < ? "
            "ORDER BY minute ASC",
            (token_id, since_minute, until_minute),
        )
        rows = await cur.fetchall()
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]

    async def totals(
        self,
        token_id: str,
        since_minute: int,
        until_minute: int,
    ) -> tuple[int, int, int]:
        """Return (requests, prompt_tokens, completion_tokens) summed over
        [since_minute, until_minute) for a single token.

        Used by the token list endpoint to populate the "Last 24h" column
        without forcing the UI to download every minute bucket.
        """
        cur = await self.db.execute(
            "SELECT COALESCE(SUM(requests), 0), "
            "       COALESCE(SUM(prompt_tokens), 0), "
            "       COALESCE(SUM(completion_tokens), 0) "
            "FROM token_usage_minute "
            "WHERE token_id = ? AND minute >= ? AND minute < ?",
            (token_id, since_minute, until_minute),
        )
        r = await cur.fetchone()
        return (int(r[0]), int(r[1]), int(r[2])) if r else (0, 0, 0)

    async def chain_bins(
        self,
        token_ids: Sequence[str],
        *,
        from_minute: int,
        to_minute: int,
        bin_minutes: int,
    ) -> list[tuple[int, int, int, int, int, int]]:
        """``(bin, requests, prompt_tokens, completion_tokens, peak_prompt,
        peak_completion)`` for ``token_ids`` over ``[from_minute, to_minute)``,
        in bins of ``bin_minutes`` keyed ``(minute / w) * w``, ascending.

        The inner ``GROUP BY minute`` adds up every key in the set for each
        minute FIRST, so the peaks are the set's busiest minute -- two rotated
        keys serving 10 and 20 prompt tokens in the same minute peak at 30,
        not at 20. Bins without rows are absent.
        """
        if not token_ids:
            return []
        marks = ",".join("?" for _ in token_ids)
        cur = await self.db.execute(
            "SELECT (minute / ?) * ? AS bin, SUM(r), SUM(p), SUM(c), MAX(p), MAX(c) "
            "FROM ("
            "  SELECT minute, SUM(requests) AS r, SUM(prompt_tokens) AS p, "
            "         SUM(completion_tokens) AS c "
            "  FROM token_usage_minute "
            f"  WHERE token_id IN ({marks}) AND minute >= ? AND minute < ? "
            "  GROUP BY minute"
            ") GROUP BY bin ORDER BY bin",
            (bin_minutes, bin_minutes, *token_ids, from_minute, to_minute),
        )
        return [
            (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
            for r in await cur.fetchall()
        ]


#: SQL behind ``TokenModelUsageRepo.earliest_minute``; a module constant so the
#: EXPLAIN QUERY PLAN test pins the exact production query.
EARLIEST_MODEL_MINUTE_SQL = "SELECT MIN(minute) FROM token_model_usage_minute"


class TokenModelUsageRepo:
    """Per-(key, model variant) minute-bucket rollup (token_model_usage_minute,
    0036).

    Written next to token_usage_minute in the proxy success path, with the same
    minute integer, only for requests that carried a key. Stores the variant id
    (app/runtime/variants.py) and, denormalised, its model id; the served name
    is resolved at read time and falls back to the stored id once the model is
    deleted. Source of the token page's "Usage by model" card and its per-model
    tokens chart (``GET /api/tokens/{id}/series``).
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def add(
        self,
        token_id: str,
        variant_id: str,
        model_id: str,
        minute: int,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        commit: bool = True,
    ) -> None:
        await self.db.execute(
            "INSERT INTO token_model_usage_minute"
            "(token_id, variant_id, model_id, minute, requests, prompt_tokens, "
            " completion_tokens) "
            "VALUES (?, ?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(token_id, variant_id, minute) DO UPDATE SET "
            "  requests = requests + 1, "
            "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "  completion_tokens = completion_tokens + excluded.completion_tokens",
            (token_id, variant_id, model_id, minute, prompt_tokens, completion_tokens),
        )
        if commit:
            await self.db.commit()

    async def by_variant(
        self,
        token_ids: Sequence[str],
        *,
        from_minute: int,
        to_minute: int,
    ) -> list[tuple[str, str, str, str | None, float | None, int, int, int]]:
        """``(model_id, model, variant_id, descriptor_json, first_seen,
        requests, prompt_tokens, completion_tokens)`` per (model, variant) for
        ``token_ids`` over ``[from_minute, to_minute)``, in no particular order.

        ``model`` is the served name from ``models``, else the name the variant
        was first seen under, else the stored model id. ``descriptor_json`` and
        ``first_seen`` are None for a variant with no ``model_variants`` row.
        """
        if not token_ids:
            return []
        marks = ",".join("?" for _ in token_ids)
        cur = await self.db.execute(
            "SELECT a.model_id, "
            "       COALESCE(m.served_model_name, v.served_model_name, a.model_id), "
            "       a.variant_id, v.descriptor, v.first_seen, a.r, a.p, a.c "
            "FROM ("
            "  SELECT model_id, variant_id, SUM(requests) AS r, "
            "         SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c "
            "  FROM token_model_usage_minute "
            f"  WHERE token_id IN ({marks}) AND minute >= ? AND minute < ? "
            "  GROUP BY model_id, variant_id"
            ") AS a "
            "LEFT JOIN models AS m ON m.id = a.model_id "
            "LEFT JOIN model_variants AS v ON v.id = a.variant_id",
            (*token_ids, from_minute, to_minute),
        )
        return [
            (
                str(r[0]), str(r[1]), str(r[2]),
                None if r[3] is None else str(r[3]),
                None if r[4] is None else float(r[4]),
                int(r[5]), int(r[6]), int(r[7]),
            )
            for r in await cur.fetchall()
        ]

    async def model_bins(
        self,
        token_ids: Sequence[str],
        *,
        from_minute: int,
        to_minute: int,
        bin_minutes: int,
    ) -> list[tuple[int, str, int, int]]:
        """``(bin, model_id, prompt_tokens, completion_tokens)`` for every
        (bin, model) pair with rows -- variants of one model summed -- bins
        keyed ``(minute / w) * w`` exactly like ``TokenUsageRepo.chain_bins``,
        ascending by bin."""
        if not token_ids:
            return []
        marks = ",".join("?" for _ in token_ids)
        cur = await self.db.execute(
            "SELECT (minute / ?) * ? AS bin, model_id, SUM(prompt_tokens), "
            "       SUM(completion_tokens) "
            "FROM token_model_usage_minute "
            f"WHERE token_id IN ({marks}) AND minute >= ? AND minute < ? "
            "GROUP BY bin, model_id ORDER BY bin, model_id",
            (bin_minutes, bin_minutes, *token_ids, from_minute, to_minute),
        )
        return [(int(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in await cur.fetchall()]

    async def earliest_minute(self) -> int | None:
        """The first minute with a row, store-wide -- where the per-model
        breakdown starts (0036 has no backfill). None on an empty table."""
        cur = await self.db.execute(EARLIEST_MODEL_MINUTE_SQL)
        row = await cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None
