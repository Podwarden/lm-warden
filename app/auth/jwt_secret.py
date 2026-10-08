# app/auth/jwt_secret.py
import os
import secrets
from pathlib import Path


def load_jwt_secret(db_path: Path) -> str:
    env_val = os.environ.get("VW_JWT_SECRET", "").strip()
    if env_val:
        return env_val
    secret_path = db_path.parent / "jwt_secret"
    if secret_path.exists():
        existing = secret_path.read_text().strip()
        if not existing:
            raise RuntimeError(
                f"JWT secret file {secret_path} exists but is empty; " "delete it to regenerate"
            )
        return existing
    if not db_path.parent.exists():
        raise RuntimeError(
            f"cannot persist JWT secret: data dir {db_path.parent} does not exist "
            "and VW_JWT_SECRET is unset"
        )
    secret = secrets.token_urlsafe(64)
    tmp = secret_path.parent / f".jwt_secret.{os.getpid()}.tmp"
    tmp.write_text(secret)
    tmp.chmod(0o600)
    tmp.rename(secret_path)
    return secret
