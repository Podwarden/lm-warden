"""CalVer parsing and advisory ranges (spec 2026-11-20, update check)."""

from app.updates.version import advisory_applies, is_newer, parse_calver


def test_parse_calver_accepts_v_prefix_and_compares_numerically() -> None:
    assert parse_calver("v2026.11.20.1") == (2026, 11, 20, 1)
    assert parse_calver("2026.5.3.12") == (2026, 5, 3, 12)
    assert parse_calver("2026.11.20.1") > parse_calver("2026.11.5.9")  # type: ignore[operator]


def test_parse_calver_rejects_everything_else() -> None:
    for junk in (
        "dev",
        "",
        "abc1234",
        "2026.11.20",
        "2026.11.20.1-rc1",
        "v2026.11.20.1.2",
        None,
        5,
    ):
        assert parse_calver(junk) is None, junk


def test_is_newer_is_strict() -> None:
    assert is_newer("v2026.11.21.1", "v2026.11.20.3")
    assert not is_newer("v2026.11.20.3", "v2026.11.20.3")
    assert not is_newer("v2026.11.19.1", "v2026.11.20.3")
    assert not is_newer("junk", "v2026.11.20.3")
    assert not is_newer("v2026.11.21.1", "dev")


def test_advisory_applies_in_half_open_range() -> None:
    adv = {"id": "LMW-1", "introduced": "2026.10.01.1", "fixed_in": "2026.11.20.1"}
    assert advisory_applies(adv, "v2026.10.01.1")
    assert advisory_applies(adv, "v2026.11.19.4")
    assert not advisory_applies(adv, "v2026.11.20.1")
    assert not advisory_applies(adv, "v2026.09.30.9")


def test_advisory_without_introduced_applies_to_everything_older() -> None:
    adv = {"id": "LMW-2", "fixed_in": "2026.11.20.1"}
    assert advisory_applies(adv, "v2025.01.01.1")
    assert not advisory_applies(adv, "v2026.12.01.1")


def test_malformed_advisory_never_applies() -> None:
    assert not advisory_applies({"id": "x"}, "v2026.01.01.1")
    assert not advisory_applies({"fixed_in": "soon"}, "v2026.01.01.1")
    assert not advisory_applies({"fixed_in": "2026.11.20.1"}, "dev")
    assert not advisory_applies("nope", "v2026.01.01.1")  # type: ignore[arg-type]


def test_release_version_prefers_the_build_arg(monkeypatch, tmp_path) -> None:
    from app.updates import version as v

    monkeypatch.setattr(v, "_BUILD_VERSION", "v2026.11.20.2")
    assert v.release_version(search=[tmp_path]) == "v2026.11.20.2"


def test_a_source_build_takes_its_version_from_the_changelog(monkeypatch, tmp_path) -> None:
    from app.updates import version as v

    monkeypatch.setattr(v, "_BUILD_VERSION", "dev")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n- x\n\n## [v2026.11.20.1] — 2026-11-20\n\n## [v2026.11.05.1]\n"
    )
    assert v.release_version(search=[tmp_path]) == "v2026.11.20.1"


def test_no_changelog_means_dev(monkeypatch, tmp_path) -> None:
    from app.updates import version as v

    monkeypatch.setattr(v, "_BUILD_VERSION", "dev")
    assert v.release_version(search=[tmp_path]) == "dev"
