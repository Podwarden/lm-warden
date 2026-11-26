"""The Prompt Lens REST API (analysis Tasks 4.1, 4.2, 4.4).

``/api/lens/me/*`` reads the calling key's own traffic: it takes an inference key (``vw_...``)
or the key holder's sign-in (app/lens/auth.py); an admin credential gets 403, and a ``key=``
parameter is refused (422): the key is the credential's (A-1). Without the key's ``prompt_lens``
permission, or with the platform switch off, every route answers the one 404 of the UI routes.
``/api/lens/admin/*`` is the same for any key (``key=``) with an admin credential, plus
``GET /api/lens/admin/fleet`` (distributions; never reachable by a key, D-4).

Each route calls the same ``service.*`` / ``api_reads.*`` functions the page and the MCP tools
call, so the doors cannot drift. Answers are codes, counts, sizes, positions and keyed hashes
(A-10); no rank, percentile or other key's figure reaches ``/me`` (A-14). Calls are limited per
credential (app/lens/limits.py), logged as route, key hash, status and time only.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.auth.deps import require_jwt
from app.lens import api_reads, report, service, wontfix
from app.lens.auth import NOT_YOURS, LensCred, gate, lens_credential, logged
from app.lens.limits import limiter_of
from app.lens.routes_ui import (
    Compare,
    Fleet,
    Scorecard,
    SessionList,
    Trends,
    Versions,
    WontfixBody,
    WontfixMarks,
)

router = APIRouter(tags=["lens-api"])

ME = "/api/lens/me"
ADMIN = "/api/lens/admin"
_NO_KEY = "no key parameter: this route reads the credential's own key"
_KEY_NEEDED = "key= is required: name the key to read"
_NOT_FOUND = "no such request or session"
_NO_COMPARE = "no scored sessions on one of the versions"

_Id = Annotated[str, Path(min_length=1, max_length=64)]
_Model = Annotated[str | None, Query(max_length=200)]
_Key = Annotated[str | None, Query(max_length=64)]
_Limit = Annotated[int, Query(ge=1, le=200)]
_By = Literal["day", "week", "version"]
_Tz = Annotated[int, Query(ge=-840, le=840)]
_Version = Annotated[str, Query(min_length=1, max_length=72)]
_Rule = Annotated[str, Path(min_length=1, max_length=16)]


class LensRequestList(BaseModel):
    requests: list[dict[str, Any]]
    next_before: float | None = None


class LensRequest(BaseModel):
    """One request: figures, the layer 1 verdict and the findings that touch it."""

    request_id: str
    finished_at: float
    model: str
    relayed: bool
    finish_reason: str | None = None
    prompt_tokens: int
    cached_tokens: int | None = None
    completion_tokens: int
    #: The key's own reusable prefix (the admin sees the fleet-lens figure).
    reusable_tokens: int | None = None
    edge: str | None = None
    ttft_s: float | None = None
    #: "sys", "tools" or "node:<i>": the first part that broke reuse; null when none did.
    break_: str | None = Field(default=None, alias="break")
    cause: str | None = None
    #: With ``span_line``: the kind of value on the changed system prompt line (an index of
    #: ``app.lens.probes.SPAN_CLASSES``) and its 1-based line.
    span_class: str | int | None = None
    span_line: int | None = None
    lost_tokens: int | None = None
    counterfactual_tokens: int | None = None
    other_breaks: int | None = None
    version: str | None = None
    session_id: str | None = None
    turn: int | None = None
    findings: list[dict[str, Any]]
    names: dict[str, str]

    model_config = ConfigDict(populate_by_name=True)


class LensDiff(BaseModel):
    request_id: str
    against: str | None = None
    continues: bool
    first_change: dict[str, Any] | None = None
    tokens: dict[str, int] | None = None
    a: dict[str, Any] | None = None
    b: dict[str, Any]


class SessionCost(BaseModel):
    cost: dict[str, Any]
    turns: list[dict[str, Any]]


class SessionFindings(BaseModel):
    findings: list[dict[str, Any]]
    not_scored: dict[str, str | None]
    names: dict[str, str]


class LeanpromptScore(BaseModel):
    """leanprompt's measured score. Without enough calls only ``scored`` (false), ``reason``
    and ``calls`` are set, never a number (A-7)."""

    scored: bool
    reason: str | None = None
    calls: int
    kind: str | None = None
    leanprompt: int | None = None
    prefill_efficiency: float | None = None
    stability: float | None = None
    generation_waste: float | None = None
    continuing_calls: int | None = None
    source: str
    since: float | None = None
    model: str | None = None


class _Scope:
    """Whose data a route reads: a key holder's own key, or the admin's pick."""

    def __init__(self, cred: LensCred, key: str | None, *, need_key: bool) -> None:
        if cred.admin:
            if need_key and not key:
                raise HTTPException(422, _KEY_NEEDED)
            self.token_id: str | None = None
            self.key = key
        else:
            if key is not None:
                raise HTTPException(422, _NO_KEY)
            self.token_id = str(cred.token_id)
            self.key = self.token_id
        self.admin = cred.admin


async def _admin_cred(user: str = Depends(require_jwt)) -> LensCred:
    return LensCred("admin", None, user)


def _build(prefix: str, dep: Any, template: str) -> None:
    """Register every route under ``prefix`` with credential dependency ``dep``."""

    async def enter(
        request: Request, cred: LensCred, key: str | None, need_key: bool, diff: bool = False
    ) -> _Scope:
        scope = _Scope(cred, key, need_key=need_key)
        await gate(request, cred, diff=diff)
        return scope

    @router.get(f"{prefix}/requests", response_model=LensRequestList)
    async def requests_list(
        request: Request,
        key: _Key = None,
        since: float | None = None,
        before: float | None = None,
        model: _Model = None,
        limit: _Limit = 50,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """The caller's finished requests, newest first (``before`` pages on)."""
        s = await enter(request, cred, key, False)
        return await logged(
            request,
            cred,
            f"{template}/requests",
            api_reads.requests_list(
                request.app.state,
                token_id=s.token_id,
                key=s.key,
                since=since,
                before=before,
                model=model,
                limit=limit,
            ),
        )

    @router.get(f"{prefix}/requests/{{req_id}}", response_model=LensRequest)
    async def request_one(
        request: Request, req_id: _Id, key: _Key = None, cred: LensCred = Depends(dep)
    ) -> dict[str, Any]:
        """One request: cache figures, the first part that broke reuse and its findings."""
        s = await enter(request, cred, key, False)
        body = await logged(
            request,
            cred,
            f"{template}/requests/{{id}}",
            api_reads.request_detail(request.app.state, token_id=s.token_id, req_id=req_id),
        )
        if body is None:
            raise HTTPException(404, NOT_YOURS if not s.admin else _NOT_FOUND)
        return body

    @router.get(f"{prefix}/requests/{{req_id}}/diff", response_model=LensDiff)
    async def request_diff(
        request: Request,
        req_id: _Id,
        against: Annotated[str | None, Query(max_length=64)] = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """What changed against the parent turn (or ``against``): the first broken part, its
        cause and position, and the token deltas. Codes and counts, never text."""
        s = await enter(request, cred, key, False, diff=True)
        try:
            body = await logged(
                request,
                cred,
                f"{template}/requests/{{id}}/diff",
                api_reads.request_diff(
                    request.app.state, token_id=s.token_id, req_id=req_id, against=against
                ),
            )
        finally:
            if not s.admin:
                limiter_of(request.app).diff_done(cred.credential_id)
        if body is None:
            raise HTTPException(404, NOT_YOURS if not s.admin else _NOT_FOUND)
        return body

    @router.get(f"{prefix}/requests/{{req_id}}/findings")
    async def request_findings(
        request: Request, req_id: _Id, key: _Key = None, cred: LensCred = Depends(dep)
    ) -> dict[str, Any]:
        """The findings that touch one request, with the part each sits on."""
        s = await enter(request, cred, key, False)
        body = await logged(
            request,
            cred,
            f"{template}/requests/{{id}}/findings",
            service.request_findings_body(request.app.state, token_id=s.token_id, req_id=req_id),
        )
        if body is None:
            raise HTTPException(404, NOT_YOURS if not s.admin else _NOT_FOUND)
        return body

    @router.get(f"{prefix}/sessions", response_model=SessionList, response_model_exclude_unset=True)
    async def sessions(
        request: Request,
        since: float | None = None,
        model: _Model = None,
        key: _Key = None,
        limit: _Limit = 50,
        cursor: Annotated[str | None, Query(max_length=200)] = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """Scored sessions, newest first, with their scores."""
        s = await enter(request, cred, key, False)
        return await logged(
            request,
            cred,
            f"{template}/sessions",
            service.session_list(
                request.app.state,
                token_id=s.token_id,
                since=since,
                model=model,
                limit=limit,
                cursor=cursor,
                key=s.key if s.admin else None,
            ),
        )

    async def _card(
        request: Request, cred: LensCred, s: _Scope, ref: str, route: str
    ) -> dict[str, Any]:
        body = await logged(
            request,
            cred,
            route,
            service.scorecard(request.app.state, token_id=s.token_id, sess_ref=ref),
        )
        if body is None:
            raise HTTPException(404, NOT_YOURS if not s.admin else _NOT_FOUND)
        return body

    @router.get(
        f"{prefix}/sessions/{{ref}}/scorecard",
        response_model=Scorecard,
        response_model_exclude_unset=True,
    )
    async def scorecard(
        request: Request, ref: _Id, key: _Key = None, cred: LensCred = Depends(dep)
    ) -> dict[str, Any]:
        """One session's scores, inputs, ranges, confidence, top 3, findings and turn strip."""
        s = await enter(request, cred, key, False)
        return await _card(request, cred, s, ref, f"{template}/sessions/{{id}}/scorecard")

    @router.get(f"{prefix}/sessions/{{ref}}/cost", response_model=SessionCost)
    async def session_cost(
        request: Request, ref: _Id, key: _Key = None, cred: LensCred = Depends(dep)
    ) -> dict[str, Any]:
        """One session's cost (tokens, seconds, money with a price card) and per-turn bars."""
        s = await enter(request, cred, key, False)
        card = await _card(request, cred, s, ref, f"{template}/sessions/{{id}}/cost")
        return {"cost": card["cost"], "turns": card["turns"]}

    @router.get(f"{prefix}/sessions/{{ref}}/findings", response_model=SessionFindings)
    async def session_findings(
        request: Request,
        ref: _Id,
        layer: Annotated[Literal["E", "C", "R"] | None, Query()] = None,
        rule: Annotated[str | None, Query(max_length=16)] = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """A session's findings, optionally one layer or one rule."""
        s = await enter(request, cred, key, False)
        card = await _card(request, cred, s, ref, f"{template}/sessions/{{id}}/findings")
        found = [
            f
            for f in card["findings"]
            if (layer is None or f["layer"] == layer) and (rule is None or f["rule"] == rule)
        ]
        return {"findings": found, "not_scored": card["not_scored"], "names": card["names"]}

    @router.get(f"{prefix}/sessions/{{ref}}/report")
    async def session_report(
        request: Request,
        ref: _Id,
        format: Annotated[Literal["json", "html"], Query()] = "json",
        tool_names: Annotated[int, Query(ge=0, le=1)] = 0,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> Response:
        """Download a session report (HTML or JSON): numbers and codes with report-local labels,
        no key, session or request ids and no prompt or reply text. Tool names only with
        ``tool_names=1``."""
        s = await enter(request, cred, key, False)
        card = await _card(request, cred, s, ref, f"{template}/sessions/{{id}}/report")
        return report.response(card, fmt=format, tool_names=bool(tool_names), admin=s.admin)

    @router.get(f"{prefix}/trends", response_model=Trends)
    async def trends(
        request: Request,
        by: _By = "week",
        model: _Model = None,
        since: float | None = None,
        tz: _Tz = 0,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """Medians per day, week or prompt version, with the prompt versions seen."""
        s = await enter(request, cred, key, True)
        return await logged(
            request,
            cred,
            f"{template}/trends",
            service.trends_body(
                request.app.state,
                key=str(s.key),
                by=by,
                model=model,
                since=since,
                tz_offset_min=tz,
            ),
        )

    @router.get(f"{prefix}/versions", response_model=Versions)
    async def versions(
        request: Request,
        model: _Model = None,
        since: float | None = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """The detected prompt versions: first and last seen, sessions, medians."""
        s = await enter(request, cred, key, True)
        return await logged(
            request,
            cred,
            f"{template}/versions",
            service.versions_body(request.app.state, key=str(s.key), model=model, since=since),
        )

    @router.get(f"{prefix}/compare", response_model=Compare)
    async def compare(
        request: Request,
        a: _Version,
        b: _Version,
        model: _Model = None,
        since: float | None = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """Version ``b`` against version ``a``: the verdict, the differences, like for like."""
        s = await enter(request, cred, key, True)
        body = await logged(
            request,
            cred,
            f"{template}/compare",
            service.compare_body(
                request.app.state, key=str(s.key), a=a, b=b, model=model, since=since
            ),
        )
        if body is None:
            raise HTTPException(404, _NO_COMPARE)
        return body

    @router.get(f"{prefix}/score", response_model=LeanpromptScore, response_model_exclude_none=True)
    async def score(
        request: Request,
        since: float | None = None,
        model: _Model = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """leanprompt's measured score (``npx leanprompt score --from lmwarden``): prefill
        efficiency, stability, generation waste, the number and the calls counted. Under 50
        calls: ``{"scored": false, ...}`` and no number."""
        s = await enter(request, cred, key, True)
        return await logged(
            request,
            cred,
            f"{template}/score",
            api_reads.score(
                request.app.state, token_id=s.token_id, key=s.key, since=since, model=model
            ),
        )

    @router.get(f"{prefix}/wontfix", response_model=WontfixMarks)
    async def wontfix_list(
        request: Request, key: _Key = None, cred: LensCred = Depends(dep)
    ) -> dict[str, Any]:
        """The rules marked won't fix."""
        s = await enter(request, cred, key, True)
        marks = await logged(
            request,
            cred,
            f"{template}/wontfix",
            service.wontfix_marks(request.app.state, key=str(s.key)),
        )
        return {"marks": marks, "changed": False}

    async def _wontfix(
        request: Request,
        cred: LensCred,
        s: _Scope,
        action: str,
        rule: str,
        version: str | None,
    ) -> dict[str, Any]:
        try:
            wontfix.check_rule(rule)
        except wontfix.WontfixError as exc:
            raise HTTPException(422, str(exc)) from None
        if s.admin and not await service.key_known(request.app.state, str(s.key)):
            raise HTTPException(404, _NOT_FOUND)
        changed, marks = await logged(
            request,
            cred,
            f"{template}/wontfix/{{rule}}",
            service.wontfix_change(
                request.app.state,
                key=str(s.key),
                action=action,
                rule=rule,
                version=version,
                by="admin" if s.admin else "key",
            ),
        )
        if changed:
            wontfix.audit(
                request,
                key_id=str(s.key),
                username=cred.user if s.admin and cred.user else wontfix.KEY_HOLDER,
                action=wontfix.MARK_ACTION if action == "mark" else wontfix.UNMARK_ACTION,
                rule=rule,
                version=version,
            )
        return {"marks": marks, "changed": changed}

    @router.put(f"{prefix}/wontfix/{{rule}}", response_model=WontfixMarks)
    async def wontfix_mark(
        request: Request,
        rule: _Rule,
        body: WontfixBody,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """Mark a rule won't fix for every version or one: it stops counting in the scores."""
        s = await enter(request, cred, key, True)
        return await _wontfix(request, cred, s, "mark", rule, body.version)

    @router.delete(f"{prefix}/wontfix/{{rule}}", response_model=WontfixMarks)
    async def wontfix_unmark(
        request: Request,
        rule: _Rule,
        version: Annotated[str | None, Query(min_length=1, max_length=72)] = None,
        key: _Key = None,
        cred: LensCred = Depends(dep),
    ) -> dict[str, Any]:
        """Remove the mark (the one for ``version``, or the one for every version)."""
        s = await enter(request, cred, key, True)
        return await _wontfix(request, cred, s, "unmark", rule, version)


_build(ME, lens_credential, ME)
_build(ADMIN, _admin_cred, ADMIN)


@router.get(f"{ADMIN}/fleet", response_model=Fleet)
async def fleet(
    request: Request,
    model: _Model = None,
    since: float | None = None,
    cred: LensCred = Depends(_admin_cred),
) -> dict[str, Any]:
    """The distribution of stored scores over every key (admin only, D-4)."""
    return await service.fleet_body(request.app.state, model=model, since=since)
