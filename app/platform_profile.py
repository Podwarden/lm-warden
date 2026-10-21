"""Which platform this deployment runs on, as a PROFILE the registry can read.

The registry's backend set and default come from this profile, NOT from the
process's own OS identity: resolving by the running platform would make the
whole unit suite see a different registry on a developer's Mac, on the Mac
merge gate, and in CI (OVERSEER DEFAULT_BACKEND decision, planner iter 22,
Metal design s10). The
profile is an OPT-IN fact about the deployment: the Swift launcher exports
``LMW_PLATFORM_PROFILE=apple`` (a P7 follow-up), and every deployment that
does not -- which includes this repository's test suite -- stays ``linux``,
byte-identical to what the registry served before profiles existed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

PROFILES: tuple[str, ...] = ("linux", "apple")


def resolve_profile(env: Mapping[str, str]) -> str:
    """Map an environment mapping to a profile. PURE: the mapping in, the
    profile out. Unset or empty means ``"linux"``; any other value that is
    not an exact match of an accepted profile raises, naming the accepted
    values, so a typo (``"Apple"``, ``"linux "``) fails at import instead of
    quietly serving the linux set on a Mac."""
    value = env.get("LMW_PLATFORM_PROFILE", "")
    if value == "":
        return "linux"
    if value in PROFILES:
        return value
    raise ValueError(
        f"LMW_PLATFORM_PROFILE={value!r} is not an accepted profile; "
        f"accepted values: {', '.join(PROFILES)}"
    )


PROFILE: str = resolve_profile(os.environ)
