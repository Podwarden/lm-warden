import pytest

from app.runtime.port_alloc import PortAllocator, PortExhausted


def test_allocate_and_release():
    p = PortAllocator(start=10000, end=10003)
    a = p.allocate()
    b = p.allocate()
    assert a != b
    assert 10000 <= a <= 10003
    p.release(a)
    c = p.allocate()
    assert c == a  # reuses freed port


def test_exhaustion_raises():
    p = PortAllocator(start=10000, end=10001)
    p.allocate()
    p.allocate()
    with pytest.raises(PortExhausted):
        p.allocate()
