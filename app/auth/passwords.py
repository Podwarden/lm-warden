"""The rule for a NEW admin password (setup wizard, Settings password change).

At least 12 characters, at most 72 bytes. The floor is for a console that
may face the internet; the ceiling is bcrypt's, and a longer password is
refused rather than silently truncated. The rule applies when a password is
set, never at login: a password stored under the old 6-character minimum
keeps working until its owner changes it.

The UI checks the same bounds before submitting
(frontend/src/lib/password-policy.ts); keep the two in step.
"""

from __future__ import annotations

MIN_PASSWORD_CHARS = 12
MAX_PASSWORD_BYTES = 72


def new_password_problem(password: str) -> str | None:
    """Why ``password`` may not be set, or None when it may."""
    if len(password) < MIN_PASSWORD_CHARS:
        return f"password must be at least {MIN_PASSWORD_CHARS} characters"
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"password must be at most {MAX_PASSWORD_BYTES} bytes"
    return None
