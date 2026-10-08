"""#276 regression: install.sh must never request the CDI "all" pseudo-device.

Reproduces the exact failure from GitLab #276: 8 physical NVIDIA GPUs, one
(index 6) fails driver init (RmInitAdapter). `nvidia-smi` correctly reports
only the 7 healthy GPUs (indices 0-5, 7), but the toolkit's CDI spec still
names all 8 device nodes, including the broken one, under
`nvidia.com/gpu=all`. The old install.sh requested that pseudo-device
whenever `--gpus all` resolved to every GPU nvidia-smi saw (GPU_MODE=all),
which handed the dead GPU's node to the container and made CUDA fail to
initialise for every GPU, healthy ones included.

This runs the real install.sh, with fake `docker`/`nvidia-smi`/`nvidia-ctk`
on PATH standing in for the host, and inspects the actual
docker-compose.override.yml it writes -- no docker daemon or GPU required.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# nvidia-smi indices for a host with 8 physical GPUs where index 6 failed
# driver init: nvidia-smi (correctly) never lists it.
HEALTHY_INDICES = ["0", "1", "2", "3", "4", "5", "7"]
FAILED_INDEX = "6"


def _write_fake_bin(bin_dir: Path, name: str, script: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_host(tmp_path):
    """A fake PATH standing in for a host with a partially-failed GPU."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()

    nvidia_smi_lines = "\n".join(
        f"{idx}, NVIDIA GeForce RTX 5090, 32768" for idx in HEALTHY_INDICES
    )
    _write_fake_bin(
        bin_dir,
        "nvidia-smi",
        f"""
case "$*" in
  "--query-gpu=index,name,memory.total --format=csv,noheader,nounits")
    cat <<'EOF'
{nvidia_smi_lines}
EOF
    ;;
  *) exit 0 ;;
esac
""",
    )

    # The CDI spec is generated from the PCI/driver view, not nvidia-smi, so
    # it still names all 8 device nodes -- the failed one included -- plus
    # the "all" pseudo-device.
    cdi_names = "\n".join(f"nvidia.com/gpu={i}" for i in range(8)) + "\nnvidia.com/gpu=all"
    _write_fake_bin(
        bin_dir,
        "nvidia-ctk",
        f"""
case "$*" in
  "cdi list") cat <<'EOF'
{cdi_names}
EOF
    ;;
  *) exit 0 ;;
esac
""",
    )

    docker_root = tmp_path / "docker-root"
    docker_root.mkdir()
    _write_fake_bin(
        bin_dir,
        "docker",
        f"""
case "$*" in
  "info --format {{{{json .CDISpecDirs}}}}") echo '["/etc/cdi"]' ;;
  "info --format {{{{.DockerRootDir}}}}") echo "{docker_root}" ;;
  "info --format {{{{.CgroupDriver}}}}") echo "cgroupfs" ;;
  info) exit 0 ;;
  "compose version --short") echo "2.29.0" ;;
  "compose config -q") exit 0 ;;
  images*) echo "" ;;
  *) exit 0 ;;
esac
""",
    )

    # Real dmesg/lspci must not leak host state into this hermetic test;
    # stub them to report nothing GPU-related.
    _write_fake_bin(bin_dir, "lspci", "exit 0")
    _write_fake_bin(bin_dir, "dmesg", "exit 0")

    return bin_dir


def _run_install(fake_host: Path, install_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{fake_host}:{env['PATH']}"
    return subprocess.run(
        [
            "sh",
            str(REPO_ROOT / "install.sh"),
            "--dir",
            str(install_dir),
            "--gpus",
            "all",
            "--yes",
            "--no-pull",
            "--no-start",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cdi_device_ids_name_only_the_healthy_gpus(fake_host, tmp_path):
    install_dir = tmp_path / "install"
    result = _run_install(fake_host, install_dir)
    assert result.returncode == 0, result.stderr

    override = (install_dir / "docker-compose.override.yml").read_text()

    # The bug: "all" hands over every device node the CDI spec knows about,
    # including the failed GPU's -- which breaks CUDA init for every GPU.
    assert "nvidia.com/gpu=all" not in override, (
        "install.sh requested the CDI 'all' pseudo-device, which includes "
        "GPUs nvidia-smi did not detect (override written:\n" + override + ")"
    )
    # The failed GPU's node must never be named, individually or via "all".
    assert f"nvidia.com/gpu={FAILED_INDEX}" not in override

    for idx in HEALTHY_INDICES:
        assert f'"nvidia.com/gpu={idx}"' in override, override


def test_gpu_selected_count_matches_healthy_gpus_only(fake_host, tmp_path):
    install_dir = tmp_path / "install"
    result = _run_install(fake_host, install_dir)
    assert result.returncode == 0, result.stderr

    env_file = (install_dir / ".env").read_text()
    assert "VW_CONTAINER_GPU_COUNT=7" in env_file, env_file
