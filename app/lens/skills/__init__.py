"""The guide "skills" the MCP endpoint serves (``list_skills``, ``load_skill``, the resources
``skill://<name>``): markdown files kept beside this module and shipped in the image.

Each file starts with a small front matter block::

    ---
    name: prompt-lens
    description: one line, shown in list_skills and resources/list
    needs: lens        # optional: "lens" (Prompt Lens permission) or "engine" (engine facts)
    ---

A guide whose ``needs`` the credential lacks is neither listed nor loadable, so what the MCP
offers matches what the tools can do. Facts in the guides come from the sister projects' own
documents (kvsig, leanprompt) and from this application; they hold no host name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent
_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_NEEDS = frozenset({"lens", "engine", "any"})


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    #: "lens", "engine" or "any".
    needs: str


def _parse(path: Path) -> Skill:
    m = _FRONT.match(path.read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"{path.name}: no front matter")
    meta = dict(
        (k.strip(), v.strip())
        for k, _, v in (line.partition(":") for line in m.group(1).splitlines() if ":" in line)
    )
    name, desc = meta.get("name", ""), meta.get("description", "")
    needs = meta.get("needs", "any")
    if name != path.stem or not re.fullmatch(r"[a-z0-9-]{1,64}", name) or not desc:
        raise ValueError(f"{path.name}: name must equal the file name and a description is needed")
    if needs not in _NEEDS:
        raise ValueError(f"{path.name}: needs must be one of {sorted(_NEEDS)}")
    return Skill(name, desc, m.group(2).lstrip("\n"), needs)


@cache
def all_skills() -> tuple[Skill, ...]:
    """Every guide, by name."""
    return tuple(_parse(p) for p in sorted(_DIR.glob("*.md")))
