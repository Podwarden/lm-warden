"""Checksum and structure validators that cut false positives."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections import Counter

_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def luhn(s: str) -> bool:
    d = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(d) <= 19 or len(set(d)) < 3:
        return False
    total = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0 and str(d[0]) in "23456"


def mod97(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    try:
        return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1
    except ValueError:
        return False


def github_crc(s: str) -> bool:
    body = s.split("_", 1)[1] if "_" in s else s
    if len(body) != 36:
        return False
    head, check = body[:30], body[30:]
    crc = binascii.crc32(head.encode()) & 0xFFFFFFFF
    n = 0
    for c in check:
        n = n * 62 + _B62.index(c)
    return n == crc


def jwt(s: str) -> bool:
    try:
        h = s.split(".")[0]
        hdr = json.loads(base64.urlsafe_b64decode(h + "=" * (-len(h) % 4)))
        return isinstance(hdr, dict) and "alg" in hdr
    except (ValueError, binascii.Error):
        return False


def _b58decode(s: str) -> bytes | None:
    n = 0
    for c in s:
        i = _B58.find(c)
        if i < 0:
            return None
        n = n * 58 + i
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return b"\x00" * (len(s) - len(s.lstrip("1"))) + raw


def base58check(s: str) -> bool:
    raw = _b58decode(s)
    if raw is None or len(raw) < 5:
        return False
    body, chk = raw[:-4], raw[-4:]
    return hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4] == chk


def btc_wif(s: str) -> bool:
    raw = _b58decode(s)
    return raw is not None and len(raw) in (37, 38) and raw[0] in (0x80, 0xEF) and base58check(s)


def bech32(s: str) -> bool:
    s = s.lower()
    if "1" not in s:
        return False
    hrp, data = s.rsplit("1", 1)
    if not data or any(c not in _BECH32 for c in data) or len(data) < 6:
        return False
    vals = [_BECH32.index(c) for c in data]
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp] + vals:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if (b >> i) & 1 else 0
    return chk in (1, 0x2BC830A3)


def b64_userpass(s: str) -> bool:
    """A Docker config `auth`: base64 of `user:password`."""
    try:
        raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=True).decode("utf-8")
    except (binascii.Error, ValueError):
        return False
    user, sep, pw = raw.partition(":")
    return bool(sep and user and len(pw) >= 4 and raw.isprintable())


VALIDATORS = {
    "luhn": luhn,
    "mod97": mod97,
    "github_crc": github_crc,
    "jwt": jwt,
    "base58check": base58check,
    "btc_wif": btc_wif,
    "bech32": bech32,
    "b64_userpass": b64_userpass,
}


# --- what a password looks like ---------------------------------------------------------------

#: A key that names a password: `password`, `passwd`, `pwd`, `passphrase`, or `pass` as one part
#: of a compound name (`SMTP_PASS`, `pass_word`) -- never inside a word (`bypass`) and never on
#: its own (`pass:` is a test result or a message in most code).
_PW_KEY = re.compile(
    r"(?i)(?:passw(?:or)?d|passphrase|pwd|[\w\-]+[_.\-]pass(?:$|[_.\-])|^pass[_.\-][\w\-]+)"
)
#: Values that are code, not a password: snake_case, camelCase / PascalCase words, dotted names.
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL = re.compile(r"^[a-zA-Z][a-z]+(?:[A-Z][a-z]+)+$")
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_.\-]*$")
#: Words that make a quoted string a UI label rather than a passphrase.
_UI_WORDS = (
    "password",
    "passphrase",
    "your ",
    "enter",
    "must",
    "invalid",
    "required",
    "please",
    "incorrect",
    "characters",
    "confirm",
    "forgot",
    "reset",
)


def _classes(v: str) -> int:
    return sum(
        (any(c.isalpha() for c in v), any(c.isdigit() for c in v), any(not c.isalnum() for c in v))
    )


def _parts(key: str) -> list[str]:
    """The words of a key's last segment: `db.SmtpPassword` -> ["smtp", "password"]."""
    seg = re.split(r"[.]", key)[-1]
    return [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", seg)]


def password_shape(key: str, value: str, sep: str = "=") -> bool:
    """An unquoted password value. A key in env style (`DB_PASSWORD=`) or YAML (`password: x`)
    holds a literal, so a letters-only value counts; in code (`password = x`) it is an
    identifier."""
    if not _PW_KEY.search(key) or len(value) < 8 or key.strip("'\"").lower() == "pwd":
        return False  # PWD is the working directory
    if _SNAKE.match(value) or value.startswith(("http://", "https://", "/", "~", "./", "=")):
        return False
    if _classes(value) >= 2:
        return True
    if _ENV_KEY.match(key):
        # An env file holds literals: letters-only counts unless it reads as an identifier.
        return len(value) >= 16 or not _CAMEL.match(value)
    if sep != ":":
        return False  # in code (`password = x`), a letters-only value is an identifier
    # YAML / data (`password: x`): only a random-looking lowercase word -- never a type, keyword or
    # ordinary capitalised word (`Optional`, `required`, `Kubernetes`, `passwordless`).
    return letters_password(value)


#: Words that sit after `password:` in code and config without being a password.
_CODE_WORDS = frozenset(
    "optional required disabled enabled none null true false string str int bool bytes secret "
    "secretstr text hidden masked redacted changeme placeholder default empty unset sensitive "
    "readonly writeonly any object number boolean".split()
)


def letters_password(value: str) -> bool:
    """A letters-only value that reads as a password rather than a word: lowercase, 10+ characters,
    not a code word, not a capitalised or camelCase word, and not naming a password itself."""
    low = value.lower()
    if value != low or len(value) < 10 or low in _CODE_WORDS:
        return False
    return not any(w in low for w in ("pass", "word", "secret", "token", "less"))


_STRONG_PW_KEY = re.compile(r"(?i)passw(?:or)?d|passphrase")


def passphrase_shape(key: str, value: str, sep: str = "=") -> bool:
    """A quoted password: may hold spaces, but not read like a UI label or a template. Spaces are
    only allowed under a key that plainly names a password (`password`, `passphrase`)."""
    if not _PW_KEY.search(key) or len(value) < 8:
        return False
    if " " in value and not _STRONG_PW_KEY.search(key):
        return False
    low = value.lower()
    if any(w in low for w in _UI_WORDS) or "{" in value or "%" in value or "$" in value:
        return False
    if " " not in value:
        # Quoted, so a literal: letters-only counts unless it reads as an identifier or a word.
        if _classes(value) >= 2 or _ENV_KEY.match(key):
            return password_shape(key, value, "=" if _ENV_KEY.match(key) else ":")
        return not _CAMEL.match(value) and letters_password(value.lower()) and value.islower()
    # A passphrase is a few words, not a sentence.
    words = value.split()
    return 12 <= len(value) <= 64 and 3 <= len(words) <= 8 and not any(c in value for c in ".,!?:;")


_SECRET_WORDS = {"secret", "token"}
_KEY_PAIRS = (("api", "key"), ("access", "key"), ("auth", "key"))


#: Trailing key words that name an environment or a version, not what the value is.
_ENV_SUFFIXES = {
    "prod",
    "production",
    "dev",
    "staging",
    "stage",
    "test",
    "live",
    "qa",
    "old",
    "new",
    "primary",
    "secondary",
    "b64",
    "base64",
    "v1",
    "v2",
    "v3",
    "1",
    "2",
    "3",
}
#: Leading key words that make a "token" something else: a paging cursor, a CSRF field.
_NOT_SECRET_PARTS = {
    "next",
    "page",
    "continuation",
    "cursor",
    "csrf",
    "xsrf",
    "last",
    "first",
    "significant",
    "lexer",
    "parser",
}


def secret_shape(key: str, value: str, sep: str = "=") -> bool:
    """A secret or token value under a key that names one as a word (`compileToken`, `API_KEY`,
    `INTERNAL_TOKEN_PROD`), not by accident of letters (`lastTokEnd`), not a cursor (`nextToken`),
    not another field (`TOKEN_FILENAME`), and not code (`a=b=c`, `x && y`)."""
    parts = _parts(key)
    while len(parts) > 1 and parts[-1] in _ENV_SUFFIXES:
        parts = parts[:-1]
    if not parts or _NOT_SECRET_PARTS & set(parts):
        return False
    last = parts[-1]
    named = last in _SECRET_WORDS or (
        last == "key" and any(p in parts for p in ("api", "access", "auth", "secret", "private"))
    )
    named = named or last in ("apikey", "accesskey", "authkey") or parts[-2:] == ["secret", "id"]
    if not named:
        return False
    body = value.rstrip("=")
    return not (
        "=" in body or "&&" in value or "||" in value or value.startswith(("http://", "https://"))
    )


def two_classes(s: str) -> bool:
    """At least two of letters, digits and symbols: a password, not a word like `active`."""
    return _classes(s) >= 2


VALIDATORS["two_classes"] = two_classes


def phone_digits(s: str) -> bool:
    return 10 <= sum(c.isdigit() for c in s) <= 15


VALIDATORS["phone_digits"] = phone_digits


SHAPES = {"password": password_shape, "passphrase": passphrase_shape, "secret": secret_shape}
