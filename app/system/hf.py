from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HfWhoAmI:
    username: str
    account_type: str


async def validate_hf_token(token: str) -> HfWhoAmI:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code == 401 or r.status_code == 403:
        raise ValueError("HuggingFace rejected token")
    r.raise_for_status()
    j = r.json()
    return HfWhoAmI(username=j.get("name", "?"), account_type=j.get("type", "user"))
