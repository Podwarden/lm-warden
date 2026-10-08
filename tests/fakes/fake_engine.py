import asyncio

from app.runtime.engine import EngineSpec


class FakeHandle:
    def __init__(self):
        self._evt = asyncio.Event()
        self._rc = None
        self.pid = 4242

    @property
    def returncode(self):
        return self._rc

    async def wait(self):
        await self._evt.wait()
        return self._rc

    def _finish(self, rc: int):
        self._rc = rc
        self._evt.set()


class FakeDriver:
    def __init__(self):
        self.spawned: list[EngineSpec] = []
        self.handles: dict[str, FakeHandle] = {}

    async def spawn(self, spec: EngineSpec) -> FakeHandle:
        self.spawned.append(spec)
        h = FakeHandle()
        self.handles[spec.model_id] = h
        return h

    async def terminate(self, handle: FakeHandle, *, grace_s: float) -> None:
        handle._finish(0)
