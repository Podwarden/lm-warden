import json
from dataclasses import asdict

import bcrypt
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.auth.passwords import new_password_problem
from app.db.database import open_db
from app.db.repos.setup import SetupRepo
from app.db.repos.users import UserRepo
from app.setup.state_machine import next_step
from app.system import gpu as gpu_mod
from app.system import hf as hf_mod


def _hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

router = APIRouter(prefix="/api/setup")


@router.get("/state")
async def get_state(request: Request):
    # Unauthenticated entry-gate probe. The /api/setup prefix is CSRF-exempt
    # (see app/auth/csrf.py _BYPASS_PREFIXES); GET is a safe method anyway.
    # A fresh install reports {"step": "welcome", "done": false}; the frontend
    # uses this to funnel first-run visitors to /setup/welcome instead of the
    # sign-in form.
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
    return {"step": state.step, "done": state.step == "done"}


@router.post("/welcome")
async def post_welcome(request: Request):
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step == "welcome":
            await repo.set_step(next_step("welcome"))
            return {"step": "gpus"}
    # Idempotent re-entry: browser Back / a reload can land a mid-wizard
    # visitor on the welcome page again (the client routes by URL, not by
    # server state). The welcome step collects no data, so re-posting it is
    # harmless — report the real current step so the client can resume
    # there, instead of erroring the wizard into an unrecoverable state.
    return {"step": state.step}


class GpusBody(BaseModel):
    allowed_gpu_indices: list[int]


class GpuInfoOut(BaseModel):
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int


@router.get("/gpus", response_model=list[GpuInfoOut])
async def get_gpus(request: Request):
    # Hide endpoint once setup is complete — don't leak GPU info post-setup.
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step == "done":
            raise HTTPException(404, "setup complete")
    gpus = await gpu_mod.query_gpus()
    return [asdict(g) for g in gpus]


@router.post("/gpus")
async def post_gpus(body: GpusBody, request: Request):
    settings = request.app.state.settings
    detected = await gpu_mod.query_gpus()
    valid_indices = {g.index for g in detected}
    if not body.allowed_gpu_indices:
        raise HTTPException(400, "must select at least one GPU")
    requested = set(body.allowed_gpu_indices)
    if not requested.issubset(valid_indices):
        bad = sorted(requested - valid_indices)
        raise HTTPException(
            400, f"GPU indices {bad} not present in container (have: {sorted(valid_indices)})"
        )
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "gpus":
            raise HTTPException(400, f"not at gpus step (current: {state.step})")
        await repo.merge_draft(allowed_gpu_indices=sorted(requested))
        await repo.set_step(next_step("gpus"))
    return {"step": "hf_token"}


class HfTokenBody(BaseModel):
    hf_token: str | None


@router.post("/hf_token")
async def post_hf_token(body: HfTokenBody, request: Request):
    settings = request.app.state.settings
    whoami_dict = None
    if body.hf_token:
        try:
            whoami = await hf_mod.validate_hf_token(body.hf_token)
            whoami_dict = {"username": whoami.username, "account_type": whoami.account_type}
        except ValueError as e:
            raise HTTPException(400, f"HuggingFace token rejected: {e}") from e
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "hf_token":
            raise HTTPException(400, f"not at hf_token step (current: {state.step})")
        if body.hf_token:
            await repo.merge_draft(hf_token=body.hf_token)
        await repo.set_step(next_step("hf_token"))
    return {"step": "admin", "whoami": whoami_dict}


class AdminBody(BaseModel):
    username: str
    password: str


@router.post("/admin")
async def post_admin(body: AdminBody, request: Request):
    # >= 12 characters, <= 72 bytes (bcrypt's limit; refused, not truncated).
    # See app/auth/passwords.py.
    problem = new_password_problem(body.password)
    if problem is not None:
        raise HTTPException(400, problem)
    if not body.username or not body.username.strip():
        raise HTTPException(400, "username required")

    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SetupRepo(db)
        state = await repo.get()
        if state.step != "admin":
            raise HTTPException(400, f"not at admin step (current: {state.step})")
        users = UserRepo(db)
        if await users.get_by_username(body.username):
            raise HTTPException(400, "username already exists")
        await users.create(body.username, _hash_password(body.password))

        # Replace plaintext token in draft with a flag
        draft = state.draft.copy()
        had_token = "hf_token" in draft
        token_value = draft.pop("hf_token", None)
        if had_token:
            draft["hf_token_present"] = True
            # Persist hf_token plaintext to a sealed file under data_dir for vLLM subprocess
            hf_token_file = settings.data_dir / "hf-token"
            hf_token_file.write_text(token_value)
            hf_token_file.chmod(0o600)

        await db.execute(
            "UPDATE setup_state SET step = 'done', draft = ?, updated_at = datetime('now') "
            "WHERE id = 1",
            (json.dumps(draft),),
        )
        await db.commit()
    return {"step": "done"}
