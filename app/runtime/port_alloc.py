from threading import Lock


class PortExhausted(RuntimeError):
    pass


class PortAllocator:
    def __init__(self, *, start: int = 10000, end: int = 10999) -> None:
        self._free: list[int] = list(range(start, end + 1))
        self._used: set[int] = set()
        self._lock = Lock()

    def allocate(self) -> int:
        with self._lock:
            if not self._free:
                raise PortExhausted("no free ports in subprocess range")
            p = self._free.pop(0)
            self._used.add(p)
            return p

    def release(self, port: int) -> None:
        with self._lock:
            if port in self._used:
                self._used.discard(port)
                self._free.insert(0, port)
