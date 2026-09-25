from threading import Lock


class GpuConflict(RuntimeError):
    pass


class GpuOwnership:
    """In-memory exclusive GPU ownership: gpu_idx -> model_id.

    LM Warden runs at most ONE loaded model per GPU. That is a deliberate
    exclusivity rule, not a capacity check -- a conflict is refused here,
    before the engine is ever spawned, so no amount of lowering
    ``gpu_memory_utilization`` will get past it. (It would not help anyway:
    vLLM reserves its KV cache as a fraction of the WHOLE card, so the
    default 0.9 leaves nothing behind however small the weights are.)

    ``label`` carries the operator-facing name of the claiming model so a
    refusal can say WHICH model is in the way. The bare
    ``GPUs [0] already claimed`` that this used to raise named neither the
    occupant nor a remedy, and it surfaces verbatim as the model's
    ``last_error`` in the UI -- so a first-time user reads it as a capacity
    problem and starts tuning numbers that cannot change the outcome.
    """

    def __init__(self) -> None:
        self._owner: dict[int, str] = {}
        self._labels: dict[str, str] = {}
        self._lock = Lock()

    def claim(
        self, model_id: str, gpu_indices: list[int], *, label: str | None = None
    ) -> None:
        with self._lock:
            conflicts = [
                g for g in gpu_indices
                if g in self._owner and self._owner[g] != model_id
            ]
            if conflicts:
                raise GpuConflict(self._conflict_message(conflicts))
            for g in gpu_indices:
                self._owner[g] = model_id
            if label:
                self._labels[model_id] = label

    def _conflict_message(self, conflicts: list[int]) -> str:
        """Name the occupant and the way out. Caller holds the lock."""
        def occupant(gpu: int) -> str:
            owner = self._owner[gpu]
            return self._labels.get(owner, owner)

        if len(conflicts) == 1:
            gpu = conflicts[0]
            head = f"GPU {gpu} is already serving '{occupant(gpu)}'"
        else:
            listed = ", ".join(f"{g} ('{occupant(g)}')" for g in conflicts)
            head = f"GPUs {listed} are already serving other models"
        return (
            f"{head} — unload it first, or load this model on a free GPU. "
            "LM Warden runs one loaded model per GPU."
        )

    def release(self, model_id: str) -> None:
        with self._lock:
            self._owner = {g: m for g, m in self._owner.items() if m != model_id}
            self._labels.pop(model_id, None)

    def owner_of(self, gpu_idx: int) -> str | None:
        with self._lock:
            return self._owner.get(gpu_idx)

    def all_claims(self) -> dict[int, str]:
        with self._lock:
            return dict(self._owner)
