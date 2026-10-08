from app.system.disk import disk_free_bytes


def test_disk_free_bytes_on_tmp(tmp_path):
    free = disk_free_bytes(tmp_path)
    assert free > 0
