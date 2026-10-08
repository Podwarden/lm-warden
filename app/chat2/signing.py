from __future__ import annotations

import hashlib
import hmac
import time


def _sig(secret: str, user_id: int, attachment_id: str, exp: int) -> str:
    msg = f"{user_id}:{attachment_id}:{exp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:32]


def sign_attachment(secret: str, user_id: int, attachment_id: str, exp: int) -> str:
    return f"{user_id}.{exp}.{_sig(secret, user_id, attachment_id, exp)}"


def verify_attachment(secret: str, user_id: int, attachment_id: str, token: str, now: int) -> bool:
    try:
        uid_s, exp_s, sig = token.split(".", 2)
        uid, exp = int(uid_s), int(exp_s)
    except ValueError:
        return False
    if uid != user_id or exp < now:
        return False
    return hmac.compare_digest(sig, _sig(secret, user_id, attachment_id, exp))


def attachment_url(secret: str, user_id: int, attachment_id: str, ttl_s: int) -> str:
    exp = int(time.time()) + ttl_s
    return f"/api/chat2/attachments/{attachment_id}?t={sign_attachment(secret, user_id, attachment_id, exp)}"
