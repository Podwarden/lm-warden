"""Forward-only SQLite schema migrations.

Every ``*.sql`` file in ``sql/`` is applied exactly once and recorded by
filename in ``schema_migrations``. Order is lexicographic over the filenames,
so the numeric prefix is load-bearing.

#231 — each migration's DDL and its ``schema_migrations`` row now commit
TOGETHER, inside one explicit transaction per file.

The obvious implementation (``db.executescript(sql)``) cannot give us that.
``executescript`` issues an implicit COMMIT *before* it runs the script, so
the bookkeeping INSERT that follows it necessarily lands in a second, separate
transaction. Anything that killed the process between the two left the DDL
half-applied while the migration still looked unapplied: the next boot replayed
it, hit "duplicate column name" / "table already exists", and the pod never came
up again. Recovery meant hand-editing the SQLite file on the /data volume —
the only failure mode in this tree with no in-band recovery path.

The trigger is not exotic. A pod evicted mid-migration does it, and so does
ENOSPC on /data — the same volume the engine logs share, and a failure this
project has already hit once (2026-06-15; see ``_rotate`` in
app/runtime/engine/local_subprocess.py).

So we split each file into statements ourselves and run them inside
``BEGIN IMMEDIATE ... COMMIT`` that also covers the INSERT. An interruption now
either commits the whole migration or none of it: SQLite's own journal rolls the
partial work back when the file is next opened, and the next boot replays the
migration against the schema it was written for.
"""

import contextlib
import sqlite3
from pathlib import Path

import aiosqlite

SQL_DIR = Path(__file__).parent / "sql"

# Statements that must never appear in a migration file, because they would
# break the per-file transaction that makes #231's atomicity guarantee hold.
#
# ``PRAGMA`` and ``VACUUM`` are the SQLite-level offenders: several pragmas
# (journal_mode, foreign_keys, page_size) are silent no-ops inside a
# transaction and VACUUM errors outright. The transaction-control keywords are
# ours: a migration that opens or closes its own transaction would commit the
# DDL out from under the ``schema_migrations`` INSERT and reintroduce exactly
# the split-commit window this module exists to close.
#
# No migration in sql/ contains any of these today (0018's and 0019's BEGIN /
# COMMIT recipes live inside ``--`` comments and are correctly skipped by the
# splitter). This guard exists so the next author who needs one gets a loud
# failure on a fresh test DB rather than a database that silently loses
# atomicity in production. If you genuinely need a pragma around a table
# rebuild, run it outside apply_migrations and say why here.
_TRANSACTION_HOSTILE = frozenset(
    {"pragma", "vacuum", "begin", "commit", "end", "rollback", "attach", "detach"}
)


def _leading_keyword(statement: str) -> str:
    """First real SQL keyword of ``statement``, lowercased.

    Skips leading whitespace and both comment forms, because the splitter keeps
    a statement's preceding comment block attached to it — every file in sql/
    opens with a paragraph of ``--`` prose, so the raw first token is almost
    never the keyword. Returns "" for a chunk that is only comments, which is
    how the caller recognises trailing prose after the last semicolon and skips
    executing it.
    """
    i, n = 0, len(statement)
    while i < n:
        if statement[i].isspace():
            i += 1
        elif statement.startswith("--", i):
            nl = statement.find("\n", i)
            i = n if nl == -1 else nl + 1
        elif statement.startswith("/*", i):
            end = statement.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            break
    j = i
    while j < n and (statement[j].isalpha() or statement[j] == "_"):
        j += 1
    return statement[i:j].lower()


def _split_statements(sql: str) -> list[str]:
    """Split a migration file into individually executable statements.

    Naive ``sql.split(";")`` is wrong for this directory in three separate
    ways, all of them present in files we ship:

    * ``0018_tokens_rate_priority.sql`` defines four triggers whose
      ``BEGIN SELECT RAISE(ABORT, ...); END;`` bodies contain semicolons. A
      naive split cuts each trigger in half and both halves are syntax errors.
    * ``0018`` and ``0019`` document their manual rollback recipe in ``--``
      comments containing full ``BEGIN; ... COMMIT;`` blocks. A naive split
      turns each comment line into its own "statement".
    * ``0002``/``0010``/``0020``/``0023`` carry string literals; a semicolon
      inside one would split mid-literal.

    Rather than write a SQL tokenizer, we borrow SQLite's own:
    ``sqlite3.complete_statement`` wraps ``sqlite3_complete()``, which is
    comment-aware, string-literal-aware, and — the part no hand-rolled splitter
    gets right — knows that a semicolon inside a CREATE TRIGGER body does not
    end the statement. Feeding it a character at a time and cutting at every
    semicolon it accepts yields exactly one statement per chunk, which is what
    ``Connection.execute`` requires (it rejects multi-statement strings).
    """
    statements: list[str] = []
    buf = ""
    for ch in sql:
        buf += ch
        if ch == ";" and sqlite3.complete_statement(buf):
            statements.append(buf)
            buf = ""
    # Trailing text after the final semicolon: usually nothing, or a closing
    # comment. Keep it only if it is an actual statement (a file whose last
    # statement forgot its semicolon), so we fail on the SQL rather than
    # silently dropping it.
    if _leading_keyword(buf):
        statements.append(buf)
    return statements


def _check_transaction_safe(filename: str, statements: list[str]) -> None:
    for stmt in statements:
        keyword = _leading_keyword(stmt)
        if keyword in _TRANSACTION_HOSTILE:
            raise RuntimeError(
                f"migration {filename} contains a '{keyword.upper()}' statement, which "
                "cannot run inside the per-migration transaction that makes migrations "
                "atomic (#231). Rewrite the migration without it, or handle it "
                "deliberately outside apply_migrations()."
            )


async def apply_migrations(db: aiosqlite.Connection) -> None:
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    await db.commit()

    cur = await db.execute("SELECT filename FROM schema_migrations")
    applied = {row[0] for row in await cur.fetchall()}

    files = sorted(p for p in SQL_DIR.glob("*.sql"))
    for path in files:
        if path.name in applied:
            continue
        statements = _split_statements(path.read_text(encoding="utf-8"))
        _check_transaction_safe(path.name, statements)

        # BEGIN IMMEDIATE, not a deferred BEGIN: take the write lock up front so
        # a busy /data (the stats sampler and pull poller hold their own
        # connections) fails on the busy_timeout set in database.py instead of
        # halfway through the DDL with SQLITE_BUSY on upgrade.
        await db.execute("BEGIN IMMEDIATE")
        try:
            for stmt in statements:
                await db.execute(stmt)
            await db.execute("INSERT INTO schema_migrations(filename) VALUES (?)", (path.name,))
        except BaseException:
            # BaseException, not Exception: a cancelled task is exactly the
            # in-process shape of the eviction this guards against, and leaving
            # the transaction open would strand the caller's connection —
            # main.py keeps using it after apply_migrations returns.
            with contextlib.suppress(Exception):
                await db.rollback()
            raise
        await db.commit()
