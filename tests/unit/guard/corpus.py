"""Synthetic secrets (generated here, so no real secret is committed) and look-alikes."""

from __future__ import annotations

import base64
import binascii
import json
import random
import string

_R = random.Random(20261029)
B62 = string.ascii_letters + string.digits


def rnd(alphabet: str, n: int) -> str:
    return "".join(_R.choice(alphabet) for _ in range(n))


def _b62(n: int, width: int) -> str:
    out = ""
    while n:
        n, r = divmod(n, 62)
        out = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[r] + out
    return out.rjust(width, "0")


def github_token(prefix: str = "ghp") -> str:
    body = rnd(B62, 30)
    crc = binascii.crc32(body.encode()) & 0xFFFFFFFF
    return f"{prefix}_{body}{_b62(crc, 6)}"


def luhn_card(prefix: str = "4", n: int = 16) -> str:
    digits = prefix + rnd(string.digits, n - len(prefix) - 1)
    total = 0
    for i, d in enumerate(reversed(digits)):
        x = int(d)
        if i % 2 == 0:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return digits + str((10 - total % 10) % 10)


def iban(country: str = "DE") -> str:
    bban = rnd(string.digits, 18)
    num = "".join(str(int(c, 36)) for c in bban + country + "00")
    check = 98 - int(num) % 97
    return f"{country}{check:02d}{bban}"


def jwt() -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{enc({'alg': 'HS256', 'typ': 'JWT'})}.{enc({'sub': rnd(B62, 8), 'exp': 1900000000})}.{rnd(B62 + '-_', 43)}"


def pem_private() -> str:
    body = "\n".join(rnd(B62 + "+/", 64) for _ in range(8))
    return f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"


def positives() -> dict[str, list[tuple[str, str]]]:
    """rule id -> [(carrier text, the secret value it must find)]"""
    aws_id = "AKIA" + rnd(string.ascii_uppercase + "234567", 16)
    aws_secret = rnd(B62 + "/+", 40)
    gh = github_token()
    pw = rnd(B62, 18)
    ant = "sk-ant-api03-" + rnd(B62 + "-_", 80)
    oa = "sk-proj-" + rnd(B62 + "-_", 120)
    lmw = "vw_" + rnd("abcdefghijklmnopqrstuvwxyz234567", 56)
    card = luhn_card()
    ib = iban()
    tok = jwt()
    pem = pem_private()
    slack = "xoxb-" + rnd(string.digits, 12) + "-" + rnd(string.digits, 12) + "-" + rnd(B62, 24)
    hf = "hf_" + rnd(string.ascii_letters, 34)
    bearer = rnd(B62 + "._-", 48)
    return {
        "aws-access-key": [(f"export AWS_ACCESS_KEY_ID={aws_id}\n", aws_id)],
        "aws-secret-key": [(f'aws_secret_access_key = "{aws_secret}"', aws_secret)],
        "github-token": [(f"token: {gh}", gh)],
        "anthropic-key": [(f"ANTHROPIC_API_KEY={ant}", ant)],
        "openai-key": [(f'{{"key": "{oa}"}}', oa)],
        "lm-warden-key": [(f"curl -H 'Authorization: Bearer {lmw}' http://x", lmw)],
        "slack-token": [(f"SLACK={slack}", slack)],
        "huggingface-token": [(f"hf token {hf}", hf)],
        "jwt": [(f"session={tok};", tok)],
        "pem-private-key": [(f"key file:\n{pem}\nend", pem)],
        "password-assignment": [(f"DB_PASSWORD={pw}\n", pw)],
        "url-credentials": [(f"DATABASE_URL=postgres://app:{pw}@db.internal:5432/app", pw)],
        "authorization-header": [(f"Authorization: Bearer {bearer}", bearer)],
        "payment-card": [
            (
                f"card {card[:4]} {card[4:8]} {card[8:12]} {card[12:]} exp 12/29",
                f"{card[:4]} {card[4:8]} {card[8:12]} {card[12:]}",
            )
        ],
        "iban": [(f"IBAN: {ib}", ib)],
    }


NEGATIVES = [
    '"integrity": "sha512-9xJ0Kv3Ra3B2iJ0f0F6NvNsUQ3SSWg2DpIjUuDlP6v8Tm5w8y1Lq2y0XHZ8M1b4yY8vH7k0Kq5o6Yh1u3wKq7Q=="',
    "commit 3f786850e387550fdab836ed7e6dc881de23001b\nAuthor: Jane <jane@example.com>",
    "sha256:4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b",
    "id: 123e4567-e89b-12d3-a456-426614174000",
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGq3yQ5u0Lx0nM0rVwXbZJcVb2p7m3W7Z0KQyQ8Yk6q jane@host",
    "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE"
    + "A" * 64
    + "\n-----END PUBLIC KEY-----",
    "-----BEGIN CERTIFICATE-----\nMIIB" + "B" * 120 + "\n-----END CERTIFICATE-----",
    "password = os.environ['DB_PASSWORD']",
    'api_key: "${OPENAI_API_KEY}"',
    "token = get_token()  # no literal",
    "password: <your-password-here>",
    "secret_key = settings.SECRET_KEY",
    "the number 1234 5678 9012 3456 is not a valid card",
    "version 4.12.3, build 20261029",
    "⟦LMW-SECRET-0123456789⟧ is a placeholder",
    # identifiers on the right of a credential-ish name (found in LM Warden's own code)
    'PLAYGROUND_TOKEN_NAME = "vw-playground"',
    "self._secret = PlaygroundSecret(token_id=tid)",
    "app.state.jwt_secret = load_jwt_secret(settings.data_dir)",
    "completion_tokens = stream_usage_completion",
    "max_num_batched_tokens=_max_num_batched_tokens(model),",
]


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58check(payload: bytes) -> str:
    import hashlib

    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\x00"))) + out


def crypto_positives() -> dict[str, list[tuple[str, str]]]:
    wif = b58check(b"\x80" + bytes(_R.randrange(256) for _ in range(32)) + b"\x01")
    xprv = b58check(bytes.fromhex("0488ade4") + bytes(_R.randrange(256) for _ in range(74)))
    hexkey = rnd("0123456789abcdef", 64)
    words = " ".join(
        _R.choice(
            [
                "abandon",
                "ability",
                "able",
                "about",
                "above",
                "absent",
                "absorb",
                "abstract",
                "absurd",
                "abuse",
            ]
        )
        for _ in range(12)
    )
    return {
        "btc-private-key": [(f"wallet import: {wif}", wif)],
        "extended-private-key": [(f"master key {xprv}", xprv)],
        "hex-private-key": [(f"PRIVATE_KEY=0x{hexkey}", hexkey)],
        "seed-phrase": [(f"mnemonic: {words}", words)],
    }


def address_positives() -> dict[str, list[tuple[str, str]]]:
    p2pkh = b58check(b"\x00" + bytes(_R.randrange(256) for _ in range(20)))
    eth = "0x" + rnd("0123456789abcdefABCDEF", 40)
    bech = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"  # BIP-173 test vector
    return {
        "btc-address": [(f"send to {p2pkh}", p2pkh)],
        "eth-address": [(f"to {eth}", eth)],
        "btc-bech32-address": [(f"addr {bech}", bech)],
    }


CRYPTO_NEGATIVES = [
    "sha256 of the file: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "commit 3f786850e387550fdab836ed7e6dc881de23001b fixes the seed loader",
    "see 0x52908400098527886E0F7030069857D2E4169EE7 in the docs",  # address: only flagged by the opt-in pack
    "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTK has a broken checksum (last char changed)",
]
