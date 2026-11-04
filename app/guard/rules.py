"""A detection rule and the built-in packs. A rule is keywords (a cheap pre-filter) + a regex run
only in a bounded window around each keyword hit + an optional validator / entropy floor. Rules
without keywords scan the whole text and must be simple, linear patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Characters a generic credential value never contains (code, templating, placeholders).
_VAL = r"[^\s'\"`,;()\[\]{}<>]"


@dataclass(frozen=True)
class Rule:
    id: str
    pack: str
    label: str
    regex: re.Pattern[str]
    keywords: tuple[str, ...] = ()
    group: int = 1
    validator: str | None = None
    entropy_min: float | None = None
    #: The value's public, non-secret prefix (shown in the log), e.g. "AKIA".
    public_prefix: str | None = None
    #: Generic rules get the value-level exclusions (env refs, identifiers, hashes).
    generic: bool = False
    #: Window around a keyword hit: chars before, chars after.
    back: int = 96
    window: int = 512
    action: str = "sanitize"
    #: Generic rules normally drop letters-only values (names, identifiers); some (a URL's
    #: password) are real secrets that are often only letters.
    allow_words: bool = False
    #: The regex group holding the key name, for `shape` and `exclude_keys`.
    key_group: int | None = None
    #: Key names (lowercase substrings) whose values are never secrets (endpoints, paging cursors).
    exclude_keys: tuple[str, ...] = ()
    #: A check on (key, value, separator) from validators.SHAPES, e.g. "password".
    shape: str | None = None
    #: The regex group holding the separator (`:` reads as YAML / JSON data, `=` as config or code).
    sep_group: int | None = None


def _r(p: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(p, flags)


CLOUD = "cloud-api-keys"
PRIV = "private-keys"
GEN = "generic-credentials"
CRYPTO = "crypto-keys"

# Order matters: on an identical span the earlier (more specific) rule wins.
BUILTIN: tuple[Rule, ...] = (
    Rule(
        "lm-warden-key",
        CLOUD,
        "LM Warden key",
        _r(r"\b(vwa?_[a-z2-7]{56})\b"),
        ("vw_", "vwa_"),
        public_prefix="vw_",
    ),
    Rule(
        "aws-access-key",
        CLOUD,
        "AWS access key",
        _r(r"\b((?:AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b"),
        ("akia", "asia", "abia", "acca"),
        public_prefix="AKIA",
    ),
    Rule(
        "aws-secret-key",
        CLOUD,
        "AWS secret key",
        _r(
            r"(?i)aws\w{0,30}(?:secret|key)\w{0,30}['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"
        ),
        ("aws",),
        entropy_min=3.5,
    ),
    Rule(
        "github-token",
        CLOUD,
        "GitHub token",
        _r(r"\b((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36})\b"),
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        validator="github_crc",
        public_prefix="ghp_",
    ),
    Rule(
        "github-fine-pat",
        CLOUD,
        "GitHub fine-grained token",
        _r(r"\b(github_pat_[A-Za-z0-9_]{82})\b"),
        ("github_pat_",),
        public_prefix="github_pat_",
    ),
    Rule(
        "gitlab-token",
        CLOUD,
        "GitLab token",
        _r(r"\b(glpat-[A-Za-z0-9_\-]{20})\b"),
        ("glpat-",),
        public_prefix="glpat-",
    ),
    Rule(
        "slack-token",
        CLOUD,
        "Slack token",
        _r(r"\b(xox[abprs]-[A-Za-z0-9-]{10,})\b"),
        ("xox",),
        public_prefix="xox",
    ),
    Rule(
        "slack-webhook",
        CLOUD,
        "Slack webhook",
        _r(r"(https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+)"),
        ("hooks.slack.com",),
    ),
    Rule(
        "stripe-key",
        CLOUD,
        "Stripe key",
        _r(r"\b((?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,})\b"),
        ("sk_live_", "rk_live_", "sk_test_", "rk_test_"),
    ),
    Rule(
        "anthropic-key",
        CLOUD,
        "Anthropic key",
        _r(r"\b(sk-ant-[A-Za-z0-9_\-]{32,})"),
        ("sk-ant-",),
        public_prefix="sk-ant-",
    ),
    Rule(
        "openai-key",
        CLOUD,
        "OpenAI key",
        _r(
            r"\b(sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}T3BlbkFJ[A-Za-z0-9_\-]{20,}|sk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{40,})"
        ),
        ("sk-",),
        public_prefix="sk-",
    ),
    Rule(
        "google-api-key",
        CLOUD,
        "Google API key",
        _r(r"\b(AIza[0-9A-Za-z_\-]{35})\b"),
        ("aiza",),
        public_prefix="AIza",
    ),
    Rule(
        "google-oauth-token",
        CLOUD,
        "Google OAuth access token",
        _r(r"\b(ya29\.[0-9A-Za-z_\-]{20,})"),
        ("ya29.",),
        public_prefix="ya29.",
    ),
    Rule(
        "vault-token",
        CLOUD,
        "HashiCorp Vault token",
        _r(r"\b(hv[sbr]\.[A-Za-z0-9_\-]{24,})"),
        ("hvs.", "hvb.", "hvr."),
        public_prefix="hvs.",
    ),
    Rule(
        "huggingface-token",
        CLOUD,
        "Hugging Face token",
        _r(r"\b(hf_[A-Za-z]{34})\b"),
        ("hf_",),
        public_prefix="hf_",
    ),
    Rule(
        "npm-token",
        CLOUD,
        "npm token",
        _r(r"\b(npm_[A-Za-z0-9]{36})\b"),
        ("npm_",),
        public_prefix="npm_",
    ),
    Rule(
        "pypi-token",
        CLOUD,
        "PyPI token",
        _r(r"\b(pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,})"),
        ("pypi-",),
        public_prefix="pypi-",
    ),
    Rule(
        "digitalocean-token",
        CLOUD,
        "DigitalOcean token",
        _r(r"\b(dop_v1_[a-f0-9]{64})\b"),
        ("dop_v1_",),
        public_prefix="dop_v1_",
    ),
    Rule(
        "sendgrid-key",
        CLOUD,
        "SendGrid key",
        _r(r"\b(SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43})\b"),
        ("sg.",),
        public_prefix="SG.",
    ),
    Rule(
        "telegram-bot-token",
        CLOUD,
        "Telegram bot token",
        _r(r"\b(\d{8,10}:AA[A-Za-z0-9_\-]{33})\b"),
        (":aa",),
    ),
    Rule(
        "azure-storage-key",
        CLOUD,
        "Azure storage key",
        _r(r"(?i)accountkey=([A-Za-z0-9+/]{86}==)"),
        ("accountkey=",),
    ),
    Rule(
        "jwt",
        CLOUD,
        "JSON Web Token",
        _r(r"\b(eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{16,})"),
        ("eyj",),
        validator="jwt",
        public_prefix="eyJ",
    ),
    Rule(
        "pem-private-key",
        PRIV,
        "Private key",
        _r(
            r"(-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----[A-Za-z0-9+/=\s:,\-\\]{16,16384}?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY(?: BLOCK)?-----)"
        ),
        ("-----begin",),
        back=0,
        window=17000,
    ),
    Rule(
        "putty-private-key",
        PRIV,
        "PuTTY private key",
        _r(r"(PuTTY-User-Key-File-\d: [\s\S]{0,8192}?Private-MAC: [0-9a-f]+)"),
        ("putty-user-key-file",),
        back=0,
        window=9000,
    ),
    Rule(
        "btc-private-key",
        CRYPTO,
        "Bitcoin private key (WIF)",
        _r(r"\b([5KLc9][1-9A-HJ-NP-Za-km-z]{50,51})\b"),
        validator="btc_wif",
    ),
    Rule(
        "extended-private-key",
        CRYPTO,
        "Extended private key",
        _r(r"\b([xtyz]prv[1-9A-HJ-NP-Za-km-z]{107,108})\b"),
        ("xprv", "tprv", "yprv", "zprv"),
        validator="base58check",
        public_prefix="xprv",
    ),
    Rule(
        "hex-private-key",
        CRYPTO,
        "Hex private key (Ethereum etc.)",
        _r(r"(?i)priv(?:ate)?[_\- ]?key\w*['\"]?\s*[:=]\s*['\"]?(?:0x)?([0-9a-f]{64})\b"),
        ("privkey", "private_key", "privatekey", "private-key", "private key", "priv_key"),
    ),
    Rule(
        "seed-phrase",
        CRYPTO,
        "Wallet seed phrase",
        _r(
            r"(?i)(?:mnemonic|seed[ _-]?phrase|seed|recovery[ _-]?phrase|secret[ _-]?phrase)\w*['\"]?\s*[:=]?\s*['\"]?((?:[a-z]{3,8} ){11,23}[a-z]{3,8})\b"
        ),
        ("mnemonic", "seed", "recovery phrase", "recovery_phrase", "secret phrase"),
    ),
    Rule(
        "url-credentials",
        GEN,
        "Password in a URL",
        _r(r"[a-zA-Z][a-zA-Z0-9+.\-]{1,20}://[^\s:/@'\"]{1,64}:([^\s@/'\"]{4,128})@[^\s/'\"]+"),
        ("://",),
        generic=True,
        allow_words=True,
    ),
    Rule(
        "docker-auth",
        GEN,
        "Registry login (Docker config)",
        _r(r"\"auth\"\s*:\s*\"([A-Za-z0-9+/]{8,}={0,2})\""),
        ('"auth"',),
        validator="b64_userpass",
    ),
    Rule(
        "cli-password",
        GEN,
        "Password on a command line",
        _r(r"(?:^|\s)(?:-u|--user)\s+['\"]?[^\s:'\"]{1,64}:([^\s'\"]{4,128})"),
        ("-u ", "--user"),
        generic=True,
    ),
    Rule(
        "cli-password",
        GEN,
        "Password on a command line",
        _r(r"\bmysql\w*\b[^\n|;]*?\s-p([^\s'\"\-][^\s'\"]{3,127})"),
        ("mysql",),
        generic=True,
    ),
    Rule(
        "cli-password",
        GEN,
        "Password on a command line",
        _r(r"--password(?:=|\s+)['\"]?([^\s'\"]{4,128})"),
        ("--password",),
        generic=True,
    ),
    Rule(
        "pgpass",
        GEN,
        "Password in .pgpass",
        _r(r"(?m)^[A-Za-z0-9.\-*]+:(?:\d{2,5}|\*):[^:\s]+:[^:\s]+:([^\s:]{4,128})$"),
        validator="two_classes",
        generic=True,
        allow_words=True,
    ),
    Rule(
        "cli-password",
        GEN,
        "Password in .netrc",
        _r(r"\bmachine\s+\S+\s+login\s+\S+\s+password\s+(\S{4,128})"),
        ("machine",),
        generic=True,
        allow_words=True,
    ),
    Rule(
        "authorization-header",
        GEN,
        "Authorization header",
        _r(
            r"(?i)(?:authorization\s*[:=]\s*['\"]?(?:bearer|basic|token)\s+|\bbearer\s+)([A-Za-z0-9._~+/\-]{16,}=*)"
        ),
        ("authorization", "bearer"),
        entropy_min=3.0,
        generic=True,
    ),
    Rule(
        "password-assignment",
        GEN,
        "Password in config",
        _r(
            rf"(?i)\b([\w.\-]*(?:pass|pwd)[\w.\-]*)['\"]?[ \t]*([:=])[ \t]*(?!['\"])({_VAL}{{8,200}})"
        ),
        ("pass", "pwd"),
        group=3,
        key_group=1,
        sep_group=2,
        shape="password",
        generic=True,
        allow_words=True,
    ),
    Rule(
        "password-assignment",
        GEN,
        "Password in config",
        _r(r"(?i)\b([\w.\-]*(?:pass|pwd)[\w.\-]*)['\"]?[ \t]*[:=][ \t]*(['\"])([^'\"\n]{8,200})\2"),
        ("pass", "pwd"),
        group=3,
        key_group=1,
        shape="passphrase",
        generic=True,
        allow_words=True,
    ),
    Rule(
        "secret-assignment",
        GEN,
        "Secret or token in config",
        _r(
            rf"(?i)\b([\w.\-]*(?:secret|token|api[_\-]?key|access[_\-]?key|auth[_\-]?key)[\w.\-]*)['\"]?[ \t]*[:=][ \t]*['\"]?({_VAL}{{12,200}})"
        ),
        ("secret", "token", "api_key", "apikey", "api-key", "access_key", "access-key", "auth_key"),
        group=2,
        key_group=1,
        shape="secret",
        entropy_min=3.5,
        generic=True,
    ),
    Rule(
        "payment-card",
        "payment-cards",
        "Payment card",
        _r(r"(?<![\d\-])((?:\d[ \-]?){12,18}\d)(?![\d\-])"),
        validator="luhn",
    ),
    Rule(
        "iban",
        "iban",
        "Bank account (IBAN)",
        _r(r"\b([A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?)\b"),
        validator="mod97",
    ),
    Rule(
        "email",
        "pii-contact",
        "E-mail address",
        _r(r"\b([A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,24})\b"),
        ("@",),
    ),
    Rule("phone", "pii-contact", "Phone number", _r(r"(?<![\w+])(\+[1-9]\d{9,14})\b"), ("+",)),
    Rule(
        "phone",
        "pii-contact",
        "Phone number",
        _r(r"(?<![\d\w])(\(\d{3}\)\s?\d{3}-\d{4}|\d{3}-\d{3}-\d{4})(?![\d\w])"),
        ("-",),
    ),
    Rule(
        "phone",
        "pii-contact",
        "Phone number",
        _r(r"(?<![\w+])(\+[1-9]\d{0,2}(?:[ \-]\d{2,4}){2,4})(?![\w\d])"),
        ("+",),
        validator="phone_digits",
    ),
    Rule(
        "btc-address",
        "crypto-addresses",
        "Bitcoin address",
        _r(r"\b([13][1-9A-HJ-NP-Za-km-z]{25,34})\b"),
        validator="base58check",
    ),
    Rule(
        "btc-bech32-address",
        "crypto-addresses",
        "Bitcoin address (bech32)",
        _r(r"\b(bc1[ac-hj-np-z02-9]{11,71})\b"),
        ("bc1",),
        validator="bech32",
        public_prefix="bc1",
    ),
    Rule(
        "eth-address",
        "crypto-addresses",
        "Ethereum address",
        _r(r"\b(0x[a-fA-F0-9]{40})\b"),
        ("0x",),
        public_prefix="0x",
    ),
)

PACKS: dict[str, str] = {
    CLOUD: "Cloud and API keys",
    PRIV: "Private keys",
    GEN: "Passwords in config",
    CRYPTO: "Crypto wallet keys",
    "payment-cards": "Payment cards",
    "iban": "Bank accounts (IBAN)",
    "pii-contact": "E-mail and phone",
    "crypto-addresses": "Crypto addresses",
}
DEFAULT_PACKS = (CLOUD, PRIV, GEN, CRYPTO)
