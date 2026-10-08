import re
import time

from app.chat2.clock import now_iso
from app.chat2.ids import new_id


def test_now_iso_is_fixed_width_utc() -> None:
    s = now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", s), s


def test_ids_are_time_ordered_and_unique() -> None:
    a = new_id()
    time.sleep(0.002)
    b = new_id()
    assert a != b and a < b and len(a) == 29
