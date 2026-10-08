"""The inference list's scope filter keeps 0035's query plans: the page CTE
still walks its sort index (the scope term is unindexable: `+t.scope`), and
the unsearched counts subtract the admin rows by seeking idx_api_tokens_scope
(0037) -- no table scan. tests/unit/db/test_migration_0035_* re-checks every
sort's walk against the migrated schema, 0037 included."""

from pathlib import Path

import pytest

from app.db.repos import tokens as token_repo
from tests.unit.db.test_migration_0035_api_tokens_sort_indexes import _migrated, _plan


def test_the_page_filters_on_an_unindexable_scope_term() -> None:
    for search in (False, True):
        assert "+t.scope = 'inference'" in token_repo.token_page_sql("created", True, search)


@pytest.mark.parametrize("expiring", [False, True])
async def test_the_unsearched_counts_seek_admin_rows_by_scope(
    tmp_data_dir: Path, expiring: bool
) -> None:
    db_path = await _migrated(tmp_data_dir)
    plan = await _plan(db_path, token_repo.token_counts_sql(False, expiring))
    assert any("USING INDEX idx_api_tokens_scope (scope=?)" in line for line in plan), plan
    assert not any(line.startswith("SCAN t") and "COVERING" not in line for line in plan), plan
