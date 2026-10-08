"""Deciding whether a measured setting may be applied (design §6.4).

Applying is the only action in this feature that CHANGES the model rather than
observing it, and it is not cheap: the settings patch refuses to touch a loaded
model, so applying means unload -> persist -> load. The model stops serving for
the length of a load, and the measurement that justified the change describes
the configuration being replaced -- by construction, applying invalidates its
own evidence.

That is why it is OFFERED and never automatic, and why each refusal here exists:
applying anyway would leave the model in a state the measurement does not
describe.
"""

from __future__ import annotations

from dataclasses import dataclass


class ApplyRefused(Exception):
    """A refusal with a machine-readable reason for the API layer."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ApplyPlan:
    max_model_len: int
    previous: int | None


def plan_apply(
    *,
    run_status: str | None,
    run_fingerprint: str | None,
    current_fingerprint: str | None,
    recommended_config: object,
    current_max_model_len: int | None,
) -> ApplyPlan:
    """What applying this run's recommendation would do, or why it may not.

    Raises :class:`ApplyRefused` rather than returning a sentinel: every caller
    has to tell the operator WHY, and a None return would let one forget.
    """
    if run_status != "completed":
        raise ApplyRefused(
            "run_not_complete",
            "only a completed run can be applied; an unfinished one carries a "
            "bracket that was never confirmed",
        )

    # The fingerprint covers the GPUs, the engine build and the co-resident set.
    # A number measured under a different one describes a machine this is not.
    if not run_fingerprint or run_fingerprint != current_fingerprint:
        raise ApplyRefused(
            "fingerprint_changed",
            "this measurement was taken under a different hardware or engine "
            "configuration, so it does not describe this model as it stands",
        )

    value = None
    if isinstance(recommended_config, dict):
        value = recommended_config.get("max_model_len")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        # choose_recommendation returns None rather than guessing when nothing
        # met the bar. That silence is a real answer and must not become a
        # setting here.
        raise ApplyRefused(
            "no_recommendation",
            "this run did not produce a recommended max_model_len",
        )

    if current_max_model_len == value:
        raise ApplyRefused(
            "already_applied",
            f"max_model_len is already {value}; applying would stop the model "
            "serving for the length of a load to arrive back where it started",
        )

    return ApplyPlan(max_model_len=value, previous=current_max_model_len)
