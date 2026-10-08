STEPS = ["welcome", "gpus", "hf_token", "admin", "done"]


def next_step(current: str) -> str:
    if current not in STEPS:
        raise ValueError(f"unknown step: {current!r}")
    if current == "done":
        return "done"
    idx = STEPS.index(current)
    return STEPS[idx + 1]
