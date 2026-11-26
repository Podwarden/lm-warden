"""Prompt Lens request findings for the request view (analysis Task 1.8, spec 16.1).

* ``GET /api/me/prompt-lens/requests/{req_id}/findings`` -- one of the signed-in
  key's OWN requests (``key_holder``: the key comes from the sign-in, never from
  the request, A-1). One 404 for Prompt Lens off, an unknown id, an evicted one
  and another key's; an admin session or admin token gets 403
  (``KEY_HOLDER_ROUTES``). At most ``READS_PER_MINUTE`` reads a minute per key.
* ``GET /api/admin/prompt-lens/requests/{req_id}/findings`` -- any key's request,
  also while Prompt Lens is off (the findings come from ``request_history``).
* ``GET /api/me/prompt-lens/sessions`` and
  ``GET /api/me/prompt-lens/sessions/{ref}/scorecard`` -- the signed-in key's own
  scored sessions and one session's scorecard (analysis Task 2.4, spec 12). Same
  rules: the key comes from the sign-in (a ``key`` parameter is ignored), one 404
  for off, unknown, evicted and another key's, 403 for an admin credential.
* ``GET /api/admin/prompt-lens/sessions`` (optional ``key=``) and
  ``GET /api/admin/prompt-lens/sessions/{ref}/scorecard`` -- any key's.

* ``GET /api/me/prompt-lens/trends``, ``/versions`` and ``/compare`` -- the key's
  own medians per day, week or prompt version, its versions, and one version
  against another (analysis Task 3.2). A ``key=`` parameter is REFUSED (422): the
  key is the sign-in's (A-1). ``GET /api/me/prompt-lens/wontfix``,
  ``PUT|DELETE /api/me/prompt-lens/wontfix/{rule}`` -- the key's won't-fix marks
  (Task 3.3). The admin's equivalents take a required ``key=``;
  ``GET /api/admin/prompt-lens/fleet`` is the admin's alone (D-4).

``req_id`` is the Prompt Lens id the request view holds (in memory or a kept
capture) or a stored request id. The answer is codes, counts, positions and
tool hashes (A-10); ``names`` joins the hashes the caller may name.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.auth.deps import require_jwt
from app.lens import report, service, wontfix
from app.proxy.prompt_lens_kept import ReadBook
from app.tokens.routes_me import key_holder, prompt_lens_view_available

router = APIRouter(tags=["prompt-lens"])

ME_PATH = "/api/me/prompt-lens/requests/{req_id}/findings"
ADMIN_PATH = "/api/admin/prompt-lens/requests/{req_id}/findings"

#: Findings reads one key may make a minute (each reads its session's rows).
READS_PER_MINUTE = 60
#: The one 404 of the key holder's route: off, unknown, evicted and another key's.
NOT_YOURS = "no request of yours with this id"
_NOT_FOUND = "no such request"


class FindingSaving(BaseModel):
    tokens: int
    prefill_s: float | None = None
    decode_s: float | None = None
    money: float | None = None


class FindingGuard(BaseModel):
    """R-09 only: the guard rule ids and the roles they matched in, from the guard's data."""

    rules: list[str]
    where: list[str]


class LensFinding(BaseModel):
    """One finding on the request: ``Finding.to_json()`` and where it sits.
    ``counts["info"] == 1`` marks a notice (never a fix); E-01's ``counts["class"]``
    indexes ``app.lens.probes.SPAN_CLASSES``; ``tool`` is a keyed hash."""

    rule: str
    turns: list[int]
    counts: dict[str, int]
    tool: str | None = None
    cause: str | None = None
    saving: FindingSaving
    confidence: Literal["high", "medium", "low"]
    muted: bool = False
    #: Who marked the rule won't fix (``key`` or ``admin``); set when ``muted``.
    muted_by: str | None = None
    #: "sys" | "tools" | "node:<i>" | "call:<k>", or null (the strip only).
    part: str | None = None
    guard: FindingGuard | None = None


class RequestFindings(BaseModel):
    #: The request's turn number in its session (1-based), null when unknown.
    turn: int | None = None
    findings: list[LensFinding]
    #: Tool hash -> plain name, for the hashes above the caller may name.
    names: dict[str, str]


def _reads_allowed(
    request: Request,
    tid: str,
    attr: str = "lens_findings_reads",
    per_minute: int = READS_PER_MINUTE,
) -> None:
    book = getattr(request.app.state, attr, None)
    if book is None:
        book = ReadBook(per_minute)
        setattr(request.app.state, attr, book)
    wait = book.wait(tid)
    if wait:
        raise HTTPException(
            429,
            "too many findings reads for this key; try again shortly",
            headers={"Retry-After": str(max(1, wait))},
        )


@router.get(ME_PATH, response_model=RequestFindings)
async def my_request_findings(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The findings on one of the signed-in key's own requests (own lens)."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, NOT_YOURS)
    _reads_allowed(request, tid)
    body = await service.request_findings_body(request.app.state, token_id=tid, req_id=req_id)
    if body is None:
        raise HTTPException(404, NOT_YOURS)
    return body


@router.get(ADMIN_PATH, response_model=RequestFindings)
async def admin_request_findings(
    request: Request,
    req_id: Annotated[str, Path(max_length=64)],
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """The findings on any key's request, as the admin sees them."""
    body = await service.request_findings_body(request.app.state, token_id=None, req_id=req_id)
    if body is None:
        raise HTTPException(404, _NOT_FOUND)
    return body


# -- scores and the scorecard (analysis Task 2.4, spec 10 and 12) -------------------------

ME_SESSIONS = "/api/me/prompt-lens/sessions"
ME_SCORECARD = "/api/me/prompt-lens/sessions/{ref}/scorecard"
ADMIN_SESSIONS = "/api/admin/prompt-lens/sessions"
ADMIN_SCORECARD = "/api/admin/prompt-lens/sessions/{ref}/scorecard"
ME_REPORT = "/api/me/prompt-lens/sessions/{ref}/report"
ADMIN_REPORT = "/api/admin/prompt-lens/sessions/{ref}/report"
#: Scorecards one key may open a minute (each analyses a whole session).
SCORECARDS_PER_MINUTE = 30


class ScoreOut(BaseModel):
    """One score, or the reason there is none (A-7): a score that was not
    computed has no ``value``, ``lo`` or ``hi`` at all, never a placeholder."""

    layer: Literal["E", "C", "R", "overall"]
    scored: bool
    turns: int
    reason: str | None = None
    value: int | None = None
    lo: int | None = None
    hi: int | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    #: The numbers the value is computed from; overall: ``{"from": ["E", ...]}``.
    inputs: dict[str, Any] | None = None
    #: Rules left out of this score (won't fix, or a term below its minimum).
    excluded_rules: list[str] | None = None


class ScorecardFinding(LensFinding):
    layer: Literal["E", "C", "R"]


class ScorecardSession(BaseModel):
    harness: str
    model: str
    route: str
    turns: int
    duration_s: float
    #: The prompt version's label ("v4", or the client's own name); null when unknown.
    version: str | None = None
    #: The stored version id, the value a version-limited won't-fix mark names.
    version_id: str | None = None
    version_since: float | None = None
    #: Admin scorecards only: the session's key.
    key_id: str | None = None


class TurnBar(BaseModel):
    n: int
    #: The stored request id of the turn, which the request view opens ("show turns").
    req_id: str
    reused: int
    recomputed: int
    new: int
    markers: list[str]


class ScorecardCard(BaseModel):
    provider: str
    model: str
    currency: str
    effective_from: str


class ScorecardCost(BaseModel):
    prompt_tokens: int
    reused_tokens: int
    recomputed_tokens: int
    fixable_lost_tokens: int
    card: ScorecardCard | None = None
    actual_input: float | None = None
    saving: float | None = None
    saving_pct: int | None = None


class Scorecard(BaseModel):
    session: ScorecardSession
    #: ``overall``, ``E``, ``C``, ``R``.
    scores: dict[str, ScoreOut]
    top: list[ScorecardFinding]
    #: R-09 (a secret in the prompt): pinned above the top 3, never ranked.
    notice: ScorecardFinding | None = None
    findings: list[ScorecardFinding]
    #: Layer -> why it was not scored.
    not_scored: dict[str, str | None]
    turns: list[TurnBar]
    cost: ScorecardCost
    score_version: int
    weights: dict[str, float]
    #: The model review (stage 5), null until then.
    review: dict[str, Any] | None = None
    names: dict[str, str]


class SessionListItem(BaseModel):
    session_id: str
    model: str
    route: str
    harness: str
    turns: int
    started: float
    last_at: float
    overall: int | None = None
    e: int | None = None
    c: int | None = None
    r: int | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    reasons: dict[str, str | None]
    score_version: int
    #: The API key's name; null when the key is gone. A key holder gets its
    #: own key's name only.
    key_name: str | None = None
    #: Admin list only.
    key_id: str | None = None


class SessionList(BaseModel):
    sessions: list[SessionListItem]
    next_cursor: str | None = None


_Ref = Annotated[str, Path(max_length=64)]
_Limit = Annotated[int, Query(ge=1, le=200)]
#: A filter that may repeat (``?model=a&model=b``): any of them.
_Many = Annotated[
    list[Annotated[str, StringConstraints(max_length=200)]] | None, Query(max_length=50)
]


@router.get(ME_SESSIONS, response_model=SessionList, response_model_exclude_unset=True)
async def my_sessions(
    request: Request,
    since: float | None = None,
    model: _Many = None,
    harness: _Many = None,
    limit: _Limit = 50,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The signed-in key's scored sessions, newest first. No ``key`` parameter:
    the key is the sign-in's (A-1)."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, NOT_YOURS)
    _reads_allowed(request, tid)
    return await service.session_list(
        request.app.state,
        token_id=tid,
        since=since,
        model=model,
        harness=harness,
        limit=limit,
        cursor=cursor,
    )


@router.get(ME_SCORECARD, response_model=Scorecard, response_model_exclude_unset=True)
async def my_scorecard(
    request: Request, ref: _Ref, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    """The scorecard of one of the signed-in key's own sessions."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, NOT_YOURS)
    _reads_allowed(request, tid, "lens_scorecard_reads", SCORECARDS_PER_MINUTE)
    body = await service.scorecard(request.app.state, token_id=tid, sess_ref=ref)
    if body is None:
        raise HTTPException(404, NOT_YOURS)
    return body


_Format = Annotated[Literal["json", "html"], Query()]
_NamesFlag = Annotated[int, Query(ge=0, le=1)]


@router.get(ME_REPORT)
async def my_report(
    request: Request,
    ref: _Ref,
    format: _Format = "json",
    tool_names: _NamesFlag = 0,
    tid: str = Depends(key_holder),
) -> Response:
    """Download one of the signed-in key's own session reports (HTML or JSON): numbers and
    codes with report-local labels, no key, session or request ids, no prompt or reply text;
    tool names only with ``tool_names=1`` (analysis Task 4.4)."""
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, NOT_YOURS)
    _reads_allowed(request, tid, "lens_scorecard_reads", SCORECARDS_PER_MINUTE)
    body = await service.scorecard(request.app.state, token_id=tid, sess_ref=ref)
    if body is None:
        raise HTTPException(404, NOT_YOURS)
    return report.response(body, fmt=format, tool_names=bool(tool_names), admin=False)


@router.get(ADMIN_REPORT)
async def admin_report(
    request: Request,
    ref: _Ref,
    format: _Format = "json",
    tool_names: _NamesFlag = 0,
    user: str = Depends(require_jwt),
) -> Response:
    """Download any key's session report; its first line says "admin view"."""
    body = await service.scorecard(request.app.state, token_id=None, sess_ref=ref)
    if body is None:
        raise HTTPException(404, _NOT_FOUND)
    return report.response(body, fmt=format, tool_names=bool(tool_names), admin=True)


@router.get(ADMIN_SESSIONS, response_model=SessionList, response_model_exclude_unset=True)
async def admin_sessions(
    request: Request,
    since: float | None = None,
    model: _Many = None,
    harness: _Many = None,
    key: Annotated[
        list[Annotated[str, StringConstraints(max_length=64)]] | None, Query(max_length=50)
    ] = None,
    limit: _Limit = 50,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Every key's scored sessions, or those of the keys named by ``key=`` (repeat it
    for several). ``model=`` and ``harness=`` repeat the same way."""
    return await service.session_list(
        request.app.state,
        token_id=None,
        since=since,
        model=model,
        harness=harness,
        limit=limit,
        cursor=cursor,
        key=key,
    )


@router.get(ADMIN_SCORECARD, response_model=Scorecard, response_model_exclude_unset=True)
async def admin_scorecard(
    request: Request, ref: _Ref, user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """The scorecard of any key's session."""
    body = await service.scorecard(request.app.state, token_id=None, sess_ref=ref)
    if body is None:
        raise HTTPException(404, _NOT_FOUND)
    return body


# -- trends, versions and won't fix (analysis Tasks 3.2, 3.3) ------------------------------

ME_TRENDS = "/api/me/prompt-lens/trends"
ME_VERSIONS = "/api/me/prompt-lens/versions"
ME_COMPARE = "/api/me/prompt-lens/compare"
ME_WONTFIX = "/api/me/prompt-lens/wontfix"
ME_WONTFIX_RULE = "/api/me/prompt-lens/wontfix/{rule}"
ADMIN_TRENDS = "/api/admin/prompt-lens/trends"
ADMIN_VERSIONS = "/api/admin/prompt-lens/versions"
ADMIN_COMPARE = "/api/admin/prompt-lens/compare"
ADMIN_WONTFIX = "/api/admin/prompt-lens/wontfix"
ADMIN_WONTFIX_RULE = "/api/admin/prompt-lens/wontfix/{rule}"
ADMIN_FLEET = "/api/admin/prompt-lens/fleet"
#: Trends, versions and comparisons one key may read a minute.
TRENDS_PER_MINUTE = 30
_NO_KEY = "no key parameter: this route reads the signed-in key"
_NO_COMPARE = "no scored sessions on one of the versions"
_BY = Literal["day", "week", "version"]
_Version = Annotated[str, Query(min_length=1, max_length=72)]
_Model = Annotated[str | None, Query(max_length=200)]
_Tz = Annotated[int, Query(ge=-840, le=840)]
_KeyId = Annotated[str, Query(min_length=1, max_length=64)]
_RuleId = Annotated[str, Path(min_length=1, max_length=16)]


class TrendPointOut(BaseModel):
    #: The local day, the local week's Monday, or the version id.
    bucket: str
    n: int
    #: ``overall``, ``E``, ``C``, ``R``; null where the bucket has under 3 scored sessions (A-7).
    medians: dict[str, int | None]
    #: 25th-75th percentile of the overall; null with no median.
    band: list[int] | None = None
    start: float
    label: str | None = None


class VersionOut(BaseModel):
    id: str
    label: str
    since: float
    last_at: float
    n: int
    medians: dict[str, int | None]
    #: Rule -> sessions of this version it fired in.
    rules: dict[str, int]
    #: Fired on the version before and not here.
    resolved: list[str]
    new: list[str]


class Trends(BaseModel):
    by: _BY
    model: str | None = None
    models: list[str]
    points: list[TrendPointOut]
    versions: list[VersionOut]
    weights: dict[str, float]
    score_version: int


class Versions(BaseModel):
    model: str | None = None
    versions: list[VersionOut]


class Compare(BaseModel):
    model: str | None = None
    a: str
    b: str
    a_label: str
    b_label: str
    n_a: int
    n_b: int
    #: Median difference (b minus a) per metric; null under 5 scored sessions a side.
    diffs: dict[str, int | None]
    verdict: Literal[
        "better_clear", "better_unclear", "no_change", "worse_clear", "worse_unclear", "not_enough"
    ]
    like_for_like: bool
    note: str | None = None
    resolved: list[str]
    new: list[str]
    p: float | None = None


class MarkOut(BaseModel):
    rule: str
    #: null: every prompt version.
    version: str | None = None
    by: str
    at: float


class WontfixMarks(BaseModel):
    marks: list[MarkOut]
    #: False when the request changed nothing (already marked, or nothing to remove).
    changed: bool = True


class WontfixBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: A prompt version id (as ``versions`` lists it), or null for every version.
    version: str | None = Field(default=None, min_length=1, max_length=72)


class Quartiles(BaseModel):
    n: int
    p10: int | None = None
    p25: int | None = None
    p50: int | None = None
    p75: int | None = None
    p90: int | None = None


class Fleet(BaseModel):
    model: str | None = None
    sessions: int
    keys: int
    overall: Quartiles
    E: Quartiles
    C: Quartiles
    R: Quartiles


def _refuse_key(request: Request) -> None:
    """A key holder's route never takes a key (A-1): it is the sign-in's."""
    if "key" in request.query_params:
        raise HTTPException(422, _NO_KEY)


async def _me(request: Request, tid: str, attr: str, per_minute: int = TRENDS_PER_MINUTE) -> None:
    _refuse_key(request)
    if not await prompt_lens_view_available(request, tid):
        raise HTTPException(404, NOT_YOURS)
    _reads_allowed(request, tid, attr, per_minute)


async def _check_rule(rule: str) -> None:
    try:
        wontfix.check_rule(rule)
    except wontfix.WontfixError as exc:
        raise HTTPException(422, str(exc)) from None


async def _change(
    request: Request, *, key: str, action: str, rule: str, version: str | None, by: str, user: str
) -> dict[str, Any]:
    await _check_rule(rule)
    changed, marks = await service.wontfix_change(
        request.app.state, key=key, action=action, rule=rule, version=version, by=by
    )
    if changed:
        wontfix.audit(
            request,
            key_id=key,
            username=user,
            action=wontfix.MARK_ACTION if action == "mark" else wontfix.UNMARK_ACTION,
            rule=rule,
            version=version,
        )
    return {"marks": marks, "changed": changed}


@router.get(ME_TRENDS, response_model=Trends)
async def my_trends(
    request: Request,
    by: _BY = "week",
    model: _Model = None,
    since: float | None = None,
    tz: _Tz = 0,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The signed-in key's medians per day, week or prompt version."""
    await _me(request, tid, "lens_trends_reads")
    return await service.trends_body(
        request.app.state, key=tid, by=by, model=model, since=since, tz_offset_min=tz
    )


@router.get(ME_VERSIONS, response_model=Versions)
async def my_versions(
    request: Request,
    model: _Model = None,
    since: float | None = None,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """The prompt versions of the signed-in key's sessions."""
    await _me(request, tid, "lens_trends_reads")
    return await service.versions_body(request.app.state, key=tid, model=model, since=since)


@router.get(ME_COMPARE, response_model=Compare)
async def my_compare(
    request: Request,
    a: _Version,
    b: _Version,
    model: _Model = None,
    since: float | None = None,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """Version ``b`` against version ``a`` of the signed-in key."""
    await _me(request, tid, "lens_trends_reads")
    body = await service.compare_body(
        request.app.state, key=tid, a=a, b=b, model=model, since=since
    )
    if body is None:
        raise HTTPException(404, _NO_COMPARE)
    return body


@router.get(ME_WONTFIX, response_model=WontfixMarks)
async def my_wontfix(request: Request, tid: str = Depends(key_holder)) -> dict[str, Any]:
    """The rules the signed-in key marked won't fix."""
    await _me(request, tid, "lens_trends_reads")
    return {"marks": await service.wontfix_marks(request.app.state, key=tid), "changed": False}


@router.put(ME_WONTFIX_RULE, response_model=WontfixMarks)
async def my_wontfix_mark(
    request: Request, rule: _RuleId, body: WontfixBody, tid: str = Depends(key_holder)
) -> dict[str, Any]:
    """Mark a rule won't fix for the signed-in key, for every version or one."""
    await _me(request, tid, "lens_trends_reads")
    return await _change(
        request,
        key=tid,
        action="mark",
        rule=rule,
        version=body.version,
        by="key",
        user=wontfix.KEY_HOLDER,
    )


@router.delete(ME_WONTFIX_RULE, response_model=WontfixMarks)
async def my_wontfix_unmark(
    request: Request,
    rule: _RuleId,
    version: Annotated[str | None, Query(min_length=1, max_length=72)] = None,
    tid: str = Depends(key_holder),
) -> dict[str, Any]:
    """Remove the mark (the one for ``version``, or the one for every version)."""
    await _me(request, tid, "lens_trends_reads")
    return await _change(
        request,
        key=tid,
        action="unmark",
        rule=rule,
        version=version,
        by="key",
        user=wontfix.KEY_HOLDER,
    )


async def _known_key(request: Request, key: str) -> None:
    if not await service.key_known(request.app.state, key):
        raise HTTPException(404, _NOT_FOUND)


@router.get(ADMIN_TRENDS, response_model=Trends)
async def admin_trends(
    request: Request,
    key: _KeyId,
    by: _BY = "week",
    model: _Model = None,
    since: float | None = None,
    tz: _Tz = 0,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """One key's medians per day, week or prompt version."""
    return await service.trends_body(
        request.app.state, key=key, by=by, model=model, since=since, tz_offset_min=tz
    )


@router.get(ADMIN_VERSIONS, response_model=Versions)
async def admin_versions(
    request: Request,
    key: _KeyId,
    model: _Model = None,
    since: float | None = None,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """One key's prompt versions."""
    return await service.versions_body(request.app.state, key=key, model=model, since=since)


@router.get(ADMIN_COMPARE, response_model=Compare)
async def admin_compare(
    request: Request,
    key: _KeyId,
    a: _Version,
    b: _Version,
    model: _Model = None,
    since: float | None = None,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Version ``b`` against version ``a`` of one key."""
    body = await service.compare_body(
        request.app.state, key=key, a=a, b=b, model=model, since=since
    )
    if body is None:
        raise HTTPException(404, _NO_COMPARE)
    return body


@router.get(ADMIN_WONTFIX, response_model=WontfixMarks)
async def admin_wontfix(
    request: Request, key: _KeyId, user: str = Depends(require_jwt)
) -> dict[str, Any]:
    """The rules one key marked won't fix."""
    return {"marks": await service.wontfix_marks(request.app.state, key=key), "changed": False}


@router.put(ADMIN_WONTFIX_RULE, response_model=WontfixMarks)
async def admin_wontfix_mark(
    request: Request,
    rule: _RuleId,
    key: _KeyId,
    body: WontfixBody,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Mark a rule won't fix for any key."""
    await _known_key(request, key)
    return await _change(
        request, key=key, action="mark", rule=rule, version=body.version, by="admin", user=user
    )


@router.delete(ADMIN_WONTFIX_RULE, response_model=WontfixMarks)
async def admin_wontfix_unmark(
    request: Request,
    rule: _RuleId,
    key: _KeyId,
    version: Annotated[str | None, Query(min_length=1, max_length=72)] = None,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """Remove a won't-fix mark of any key."""
    await _known_key(request, key)
    return await _change(
        request, key=key, action="unmark", rule=rule, version=version, by="admin", user=user
    )


@router.get(ADMIN_FLEET, response_model=Fleet)
async def admin_fleet(
    request: Request,
    model: _Model = None,
    since: float | None = None,
    user: str = Depends(require_jwt),
) -> dict[str, Any]:
    """The distribution of the stored scores over every key (admin only, D-4)."""
    return await service.fleet_body(request.app.state, model=model, since=since)
