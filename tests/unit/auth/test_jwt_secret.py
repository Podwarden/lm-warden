# tests/unit/auth/test_jwt_secret.py
import pytest

from app.auth.jwt_secret import load_jwt_secret


def test_env_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_JWT_SECRET", "from-env")
    secret = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert secret == "from-env"
    assert not (tmp_path / "jwt_secret").exists()


def test_persists_on_first_boot(tmp_path, monkeypatch):
    monkeypatch.delenv("VW_JWT_SECRET", raising=False)
    secret_path = tmp_path / "jwt_secret"
    secret = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert len(secret) >= 64
    assert secret_path.exists()
    assert oct(secret_path.stat().st_mode)[-3:] == "600"
    secret2 = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert secret == secret2


def test_refuses_unwritable_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VW_JWT_SECRET", raising=False)
    bad = tmp_path / "nope"  # parent does not exist and we will not mkdir
    with pytest.raises(RuntimeError, match="cannot persist"):
        load_jwt_secret(db_path=bad / "child" / "vw.db")


def test_blank_env_falls_through_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VW_JWT_SECRET", "   ")
    secret = load_jwt_secret(db_path=tmp_path / "vw.db")
    assert len(secret) >= 64
    assert (tmp_path / "jwt_secret").exists()


def test_rejects_empty_secret_file(tmp_path, monkeypatch):
    monkeypatch.delenv("VW_JWT_SECRET", raising=False)
    (tmp_path / "jwt_secret").write_text("")
    with pytest.raises(RuntimeError, match="exists but is empty"):
        load_jwt_secret(db_path=tmp_path / "vw.db")
