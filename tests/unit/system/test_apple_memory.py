"""P6 (Metal design s8) — vm_stat / vm.swapusage / kern.memorystatus_level
parsers and the never-raising reader.

None means "not reported", never 0: every field a probe could not read stays
None and no path raises."""

_VM_STAT_16384 = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free: 1000.\n"
    "Pages active: 2000.\n"
    "Pages inactive: 3000.\n"
    "Pages wired down: 4000.\n"
    "Pages occupied by compressor: 5000.\n"
    "Pages speculative: 6000.\n"
)


def test_parse_vm_stat_page_size_16384():
    from app.system.apple.memory import parse_vm_stat

    result = parse_vm_stat(_VM_STAT_16384)
    assert result["free"] == 16384000
    assert result["active"] == 32768000
    assert result["inactive"] == 49152000
    assert result["wired"] == 65536000
    assert result["compressed"] == 81920000
    assert result["speculative"] == 98304000


def test_parse_vm_stat_page_size_4096():
    from app.system.apple.memory import parse_vm_stat

    text = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free: 1000.\n"
        "Pages active: 1.\n"
    )
    result = parse_vm_stat(text)
    assert result["free"] == 4096000
    assert result["active"] == 4096
    # Keys not present in the text stay None, not 0.
    assert result["inactive"] is None
    assert result["wired"] is None
    assert result["compressed"] is None
    assert result["speculative"] is None


def test_parse_vm_stat_missing_header_all_none():
    from app.system.apple.memory import parse_vm_stat

    text = (
        "Pages free: 1000.\n"
        "Pages active: 2000.\n"
        "Pages inactive: 3000.\n"
        "Pages wired down: 4000.\n"
        "Pages occupied by compressor: 5000.\n"
        "Pages speculative: 6000.\n"
    )
    result = parse_vm_stat(text)
    assert result == {
        "free": None,
        "active": None,
        "inactive": None,
        "wired": None,
        "compressed": None,
        "speculative": None,
    }


def test_parse_vm_stat_unparseable_header_all_none():
    from app.system.apple.memory import parse_vm_stat

    text = "Mach Virtual Memory Statistics: (page size of ?? bytes)\n" "Pages free: 1000.\n"
    result = parse_vm_stat(text)
    assert result["free"] is None


def test_parse_vm_stat_missing_key_only_that_field_none():
    from app.system.apple.memory import parse_vm_stat

    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 1000.\n"
        "Pages active: 2000.\n"
        "Pages wired down: 4000.\n"
    )
    result = parse_vm_stat(text)
    assert result["free"] == 16384000
    assert result["active"] == 32768000
    assert result["wired"] == 65536000
    assert result["inactive"] is None
    assert result["compressed"] is None
    assert result["speculative"] is None


def test_parse_vm_stat_malformed_count_field_none_others_set():
    from app.system.apple.memory import parse_vm_stat

    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: n/a.\n"
        "Pages active: 2000.\n"
    )
    result = parse_vm_stat(text)
    assert result["free"] is None
    assert result["active"] == 32768000


def test_parse_vm_stat_empty_text():
    from app.system.apple.memory import parse_vm_stat

    result = parse_vm_stat("")
    assert result["free"] is None
    assert result["active"] is None
    assert result["inactive"] is None
    assert result["wired"] is None
    assert result["compressed"] is None
    assert result["speculative"] is None


def test_parse_swapusage_mixed_units():
    from app.system.apple.memory import parse_swapusage

    result = parse_swapusage(
        "        total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)"
    )
    assert result["total_bytes"] == 2048 * 2**20
    assert result["used_bytes"] == (1024 * 2 + 1) * 2**19
    assert result["free_bytes"] == (1023 * 2 + 1) * 2**19


def test_parse_swapusage_giga_unit():
    from app.system.apple.memory import parse_swapusage

    result = parse_swapusage("        total = 2.00G  used = 1.50G  free = 0.50G")
    assert result["total_bytes"] == 2 * 2**30
    assert result["used_bytes"] == 3 * 2**29
    assert result["free_bytes"] == 2**29


def test_parse_swapusage_unsuffixed_number_is_bytes():
    from app.system.apple.memory import parse_swapusage

    result = parse_swapusage("        total = 0  used = 0  free = 0")
    assert result["total_bytes"] == 0
    assert result["used_bytes"] == 0
    assert result["free_bytes"] == 0


def test_parse_swapusage_garbage_field_none():
    from app.system.apple.memory import parse_swapusage

    result = parse_swapusage("        total = bogus  used = 1.00G  free = 1.00G")
    assert result["total_bytes"] is None
    assert result["used_bytes"] == 2**30
    assert result["free_bytes"] == 2**30


def test_parse_swapusage_garbage_text_all_none():
    from app.system.apple.memory import parse_swapusage

    result = parse_swapusage("no swap configured")
    assert result == {"total_bytes": None, "used_bytes": None, "free_bytes": None}


def test_parse_memorystatus_level_values():
    from app.system.apple.memory import parse_memorystatus_level

    assert parse_memorystatus_level("0") == 0
    assert parse_memorystatus_level("0\n") == 0
    assert parse_memorystatus_level("  100  ") == 100


def test_parse_memorystatus_level_invalid():
    from app.system.apple.memory import parse_memorystatus_level

    assert parse_memorystatus_level("abc") is None
    assert parse_memorystatus_level("") is None
    assert parse_memorystatus_level("-1") is None
    assert parse_memorystatus_level("1.5") is None


def test_read_memory_pressure_all_fields():
    from app.system.apple.memory import MemoryPressure, read_memory_pressure

    def run(argv):
        if argv == ["vm_stat"]:
            return _VM_STAT_16384
        if argv == ["sysctl", "-n", "vm.swapusage"]:
            return "        total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)"
        if argv == ["sysctl", "-n", "kern.memorystatus_level"]:
            return "100\n"
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_memory_pressure(run=run)
    assert isinstance(result, MemoryPressure)
    assert result.free == 16384000
    assert result.active == 32768000
    assert result.inactive == 49152000
    assert result.wired == 65536000
    assert result.compressed == 81920000
    assert result.speculative == 98304000
    assert result.swap_total_bytes == 2048 * 2**20
    assert result.swap_used_bytes == (1024 * 2 + 1) * 2**19
    assert result.swap_free_bytes == (1023 * 2 + 1) * 2**19
    assert result.pressure_level == 100


def test_read_memory_pressure_runner_none():
    from app.system.apple.memory import read_memory_pressure

    result = read_memory_pressure(run=lambda argv: None)
    assert result.free is None
    assert result.active is None
    assert result.inactive is None
    assert result.wired is None
    assert result.compressed is None
    assert result.speculative is None
    assert result.swap_total_bytes is None
    assert result.swap_used_bytes is None
    assert result.swap_free_bytes is None
    assert result.pressure_level is None


def test_read_memory_pressure_runner_raises():
    from app.system.apple.memory import read_memory_pressure

    def boom(argv):
        raise OSError("vm_stat not found")

    result = read_memory_pressure(run=boom)
    assert result.free is None
    assert result.swap_total_bytes is None
    assert result.pressure_level is None


def test_read_memory_pressure_runner_raises_non_oserror():
    from app.system.apple.memory import read_memory_pressure

    # Not only OSError: a timeout, a decode error or a bug in the runner must
    # also degrade to None fields, never reach the caller.
    for exc in (RuntimeError("timed out"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")):

        def run(argv, exc=exc):
            if argv == ["sysctl", "-n", "kern.memorystatus_level"]:
                return "43\n"
            raise exc

        result = read_memory_pressure(run=run)
        assert result.free is None
        assert result.swap_total_bytes is None
        assert result.pressure_level == 43


def test_read_memory_pressure_partial_failure():
    from app.system.apple.memory import read_memory_pressure

    def run(argv):
        if argv == ["vm_stat"]:
            return _VM_STAT_16384
        if argv == ["sysctl", "-n", "vm.swapusage"]:
            return None
        if argv == ["sysctl", "-n", "kern.memorystatus_level"]:
            return "abc\n"
        raise AssertionError(f"unexpected argv: {argv}")

    result = read_memory_pressure(run=run)
    assert result.free == 16384000
    assert result.swap_total_bytes is None
    assert result.swap_used_bytes is None
    assert result.swap_free_bytes is None
    assert result.pressure_level is None


def test_parse_vm_stat_real_macos_layout_and_exact_bytes():
    from app.system.apple.memory import parse_vm_stat

    # Captured from `vm_stat` on an M1 (column-aligned, trailing '.', extra rows).
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                4024.\n"
        "Pages active:                            216331.\n"
        "Pages inactive:                          215180.\n"
        "Pages speculative:                          240.\n"
        "Pages throttled:                              0.\n"
        "Pages wired down:                        162961.\n"
        "Pages purgeable:                            100.\n"
        "Pages stored in compressor:              900000.\n"
        "Pages occupied by compressor:            123456.\n"
    )
    result = parse_vm_stat(text)
    assert result == {
        "free": 4024 * 16384,
        "active": 216331 * 16384,
        "inactive": 215180 * 16384,
        "wired": 162961 * 16384,
        "compressed": 123456 * 16384,
        "speculative": 240 * 16384,
    }
    # Exact integer math: a count past float's 2**53 keeps every digit.
    big = 2**60 + 1
    huge = parse_vm_stat(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" f"Pages free: {big}.\n"
    )
    assert huge["free"] == big * 16384


def test_parse_vm_stat_fractional_or_zero_page_size_is_none():
    from app.system.apple.memory import parse_vm_stat

    header = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    assert parse_vm_stat(header + "Pages free: 1000.5\n")["free"] is None
    assert parse_vm_stat(header + "Pages free: 1.2.\n")["free"] is None
    zero = parse_vm_stat("Mach Virtual Memory Statistics: (page size of 0 bytes)\nPages free: 7.\n")
    assert zero["free"] is None


def test_huge_values_degrade_to_none_never_raise():
    from app.system.apple.memory import parse_swapusage, read_memory_pressure

    huge = "9" * 400
    swap = parse_swapusage(f"total = {huge}.00M  used = 1.00M  free = 1.00M")
    assert swap["used_bytes"] == 2**20
    assert swap["total_bytes"] is not None and swap["total_bytes"] > 0

    def run(argv):
        if argv == ["vm_stat"]:
            return (
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                f"Pages free: {huge}.\n"
            )
        if argv == ["sysctl", "-n", "vm.swapusage"]:
            return f"total = {huge}.00G  used = {huge}M  free = 0.00M"
        return "43\n"

    result = read_memory_pressure(run=run)  # must not raise
    assert result.free == int(huge) * 16384
    assert result.swap_free_bytes == 0
    assert result.pressure_level == 43
