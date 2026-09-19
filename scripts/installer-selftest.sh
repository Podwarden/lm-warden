#!/bin/sh
# Self-test for install.sh that needs no Docker daemon and no GPU.
#
# `docker` and `nvidia-smi` are replaced by stubs on PATH whose behaviour is
# driven by STUB_* variables, so every preflight branch and every .env /
# override outcome can be asserted on any machine -- a laptop, a shell
# runner, busybox in a container. What it cannot prove is that Compose
# accepts the generated files or that the images boot; the installer:stack-up
# CI job does that with the real thing.
#
# Usage: scripts/installer-selftest.sh            (uses the sh on PATH)
#        SHELL_UNDER_TEST=dash scripts/installer-selftest.sh

# shellcheck disable=SC2015  # `check && pass || fail` is the assertion idiom here; pass() cannot fail
# shellcheck disable=SC2016  # the shell-under-test expands ${BASH_VERSION}, not this one
# shellcheck disable=SC2012  # ls -l on one file we created ourselves, for its mode string

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SH=${SHELL_UNDER_TEST:-sh}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/vw-selftest.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

STUBS="$WORK/stubs"; mkdir -p "$STUBS"
CALLS="$WORK/calls.log"; export CALLS

# ---- stubs --------------------------------------------------------------------

cat > "$STUBS/docker" <<'EOF'
#!/bin/sh
# docker stub. STUB_DAEMON=down makes `docker info` fail; STUB_COMPOSE_VERSION
# is what `compose version --short` prints; STUB_NVIDIA_RUNTIME=1 lists the
# nvidia runtime in `docker info --format`; STUB_DOCKER_ROOT is the data root
# `docker info --format {{.DockerRootDir}}` reports (set it empty to model a
# daemon that reports none); STUB_HAVE_IMAGE=1 makes `docker images -q` find
# an api image already in the store; STUB_DOCKER_CDI=1 lists CDI spec dirs
# (CDI enabled); STUB_CGROUP is the cgroup driver (cgroupfs). Every call is
# appended to $CALLS.
printf '%s\n' "docker $*" >> "$CALLS"
case "$1" in
  info)
    [ "${STUB_DAEMON:-up}" = up ] || exit 1
    if [ "${2:-}" = "--format" ]; then
      case "${3:-}" in
        *DockerRootDir*) echo "${STUB_DOCKER_ROOT-/var/lib/docker}" ;;
        *CDISpecDirs*)
          if [ "${STUB_DOCKER_CDI:-0}" = 1 ]; then echo '["/etc/cdi","/var/run/cdi"]'; else echo '[]'; fi ;;
        *CgroupDriver*) echo "${STUB_CGROUP:-cgroupfs}" ;;
        *)
          if [ "${STUB_NVIDIA_RUNTIME:-0}" = 1 ]; then echo '{"io.containerd.runc.v2":{},"nvidia":{"path":"nvidia-container-runtime"},"runc":{}}'
          else echo '{"io.containerd.runc.v2":{},"runc":{}}'; fi ;;
      esac
    fi
    exit 0 ;;
  images)
    [ "${STUB_HAVE_IMAGE:-0}" = 1 ] && echo 6cf7aca99527
    exit 0 ;;
  compose)
    case "$2" in
      version) echo "${STUB_COMPOSE_VERSION:-2.29.1}"; exit 0 ;;
      config|pull|up|down) exit "${STUB_COMPOSE_RC:-0}" ;;
    esac ;;
esac
exit 0
EOF
cat > "$STUBS/nvidia-smi" <<'EOF'
#!/bin/sh
# nvidia-smi stub: STUB_GPUS is the number of GPUs to report (0 = fail like a
# host without a driver).
n=${STUB_GPUS:-0}
[ "$n" -gt 0 ] || exit 9
i=0
while [ "$i" -lt "$n" ]; do echo "$i, NVIDIA Stub GPU $i, 24576"; i=$((i+1)); done
EOF
cat > "$STUBS/df" <<'EOF'
#!/bin/sh
# df stub, POSIX -P layout. STUB_DF_AVAIL_KB is the Available column (default
# ~110 GB, roomy); STUB_DF=fail makes df fail the way it does on a path that
# does not exist, so the installer has to cope with no number at all.
[ "${STUB_DF:-ok}" = ok ] || { echo "df: cannot read file system information" >&2; exit 1; }
avail=${STUB_DF_AVAIL_KB:-107500000}
echo "Filesystem     1024-blocks      Used Available Capacity Mounted on"
echo "/dev/vda2        209612800 $((209612800 - avail)) $avail  50% /"
EOF
# nvidia-ctk is absent unless a case sets STUB_CTK=1 (the toolkit is then
# "installed"): run() puts $STUBS/ctk on PATH only for those cases.
mkdir -p "$STUBS/ctk"
cat > "$STUBS/ctk/nvidia-ctk" <<'EOF'
#!/bin/sh
# nvidia-ctk stub. STUB_CDI is a comma list of nvidia.com/gpu device names the
# CDI spec provides (e.g. all,0,1); empty = no spec.
printf '%s\n' "nvidia-ctk $*" >> "$CALLS"
case "$1 ${2:-}" in
  "cdi list")
    echo "INFO Found ${STUB_CDI:+some} CDI devices" >&2
    for n in $(printf '%s' "${STUB_CDI:-}" | tr ',' ' '); do echo "nvidia.com/gpu=$n"; done ;;
esac
exit 0
EOF
chmod +x "$STUBS/docker" "$STUBS/nvidia-smi" "$STUBS/df" "$STUBS/ctk/nvidia-ctk"

# A minimal source tree: the real files, as install.sh will see them in a
# checkout, plus a changelog naming a release so VERSION pinning is exercised.
SRC="$WORK/src"; mkdir -p "$SRC/deploy/caddy" "$SRC/make"
cp "$ROOT/install.sh" "$ROOT/docker-compose.yml" "$ROOT/.env.example" "$SRC/"
cp "$ROOT/deploy/caddy/Caddyfile" "$SRC/deploy/caddy/"
cp "$ROOT/make/operator.mk" "$SRC/make/"
printf '# Changelog\n\n## [Unreleased]\n\n## [v2026.01.02.3] - 2026-01-02\n' > "$SRC/CHANGELOG.md"

# ---- harness --------------------------------------------------------------------

n_pass=0; n_fail=0
pass() { n_pass=$((n_pass+1)); printf '  ok   %s\n' "$1"; }
fail() { n_fail=$((n_fail+1)); printf '  FAIL %s\n' "$1"; [ -z "${2:-}" ] || sed 's/^/       | /' "$2"; }
assert_rc() { # name expected actual [logfile]
  if [ "$2" = "$3" ]; then pass "$1"; else fail "$1 (exit $3, expected $2)" "${4:-}"; fi
}
assert_grep() { # name pattern file
  if grep -qE -- "$2" "$3" 2>/dev/null; then pass "$1"; else fail "$1 (no match for '$2' in $3)" "$3"; fi
}
assert_nogrep() {
  if grep -qE -- "$2" "$3" 2>/dev/null; then fail "$1 (unexpected match for '$2' in $3)" "$3"; else pass "$1"; fi
}
env_val() { sed -n "s/^$1=//p" "$2" | head -1 | sed 's/[[:space:]]*#.*$//'; }

# run NAME [env assignments...] -- args... : runs install.sh in a fresh copy
# of the tree with the stubs first on PATH. Sets RC, OUT (log), D (dir).
run() {
  NAME=$1; shift
  D="$WORK/case-$NAME"; rm -rf "$D"; cp -R "$SRC" "$D"
  OUT="$WORK/out-$NAME.log"; : > "$CALLS"
  envs=""
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs="$envs $1"; shift; done
  [ $# -eq 0 ] || shift
  set +e
  case " $envs " in *" STUB_CTK=1 "*) SP="$STUBS/ctk:$STUBS" ;; *) SP="$STUBS" ;; esac
  # shellcheck disable=SC2086
  (cd "$D" && env PATH="$SP:$PATH" $envs "$SH" ./install.sh "$@") >"$OUT" 2>&1
  RC=$?
  set -e
}

printf 'install.sh self-test under %s\n' "$("$SH" -c 'echo "${BASH_VERSION:-${KSH_VERSION:-sh}}"')"

# ---- argument handling -------------------------------------------------------------

run help -- --help
assert_rc "--help exits 0" 0 "$RC" "$OUT"
assert_grep "--help documents --gpus" '--gpus VALUE' "$OUT"
assert_grep "--help documents GPU_TOOLKIT_INSTALL" 'GPU_TOOLKIT_INSTALL=yes.no' "$OUT"
assert_grep "--help documents GPU_REQUEST" 'GPU_REQUEST=auto.cdi.nvidia' "$OUT"

run badflag -- --bogus
assert_rc "unknown flag exits 1" 1 "$RC"
assert_grep "unknown flag is named" 'Unknown argument: --bogus' "$OUT"

run badgpus -- --gpus 0-1
assert_rc "malformed --gpus exits 1" 1 "$RC"

run badport -- --port 70000 --gpus none --yes
assert_rc "out-of-range --port exits 1" 1 "$RC"

run dirnoval -- --dir
assert_rc "--dir without a value exits 1" 1 "$RC"

# ---- preflight failures -----------------------------------------------------------------

# A PATH with everything the host has except docker: a mirror directory of
# symlinks, because `command -v docker` must genuinely fail and on a Linux
# host the real binary sits in /usr/bin next to everything else.
D="$WORK/case-nodocker"; rm -rf "$D"; cp -R "$SRC" "$D"; OUT="$WORK/out-nodocker.log"
NODOCKER="$WORK/nodocker"; mkdir -p "$NODOCKER"
for pd in $(printf '%s' "$PATH" | tr ':' ' '); do
  [ -d "$pd" ] || continue
  for f in "$pd"/*; do
    b=$(basename "$f")
    [ "$b" = docker ] && continue
    [ -e "$NODOCKER/$b" ] || ln -s "$f" "$NODOCKER/$b" 2>/dev/null || true
  done
done
set +e; (cd "$D" && env PATH="$NODOCKER" "$SH" ./install.sh --gpus none --yes) >"$OUT" 2>&1; RC=$?; set -e
assert_rc "no docker binary exits 1" 1 "$RC" "$OUT"
assert_grep "no docker: says so" 'Docker is not installed' "$OUT"
[ ! -f "$D/.env" ] && pass "no docker: nothing written" || fail "no docker: .env was written"

run daemondown STUB_DAEMON=down -- --gpus none --yes
assert_rc "daemon unreachable exits 1" 1 "$RC"
assert_grep "daemon unreachable: hint about docker group" 'usermod -aG docker|systemctl status docker' "$OUT"

run oldcompose STUB_COMPOSE_VERSION=v2.20.3 -- --gpus none --yes
assert_rc "compose 2.20 exits 1" 1 "$RC"
assert_grep "compose 2.20: names the floor" '2\.24 or newer' "$OUT"

run compose3 STUB_COMPOSE_VERSION=3.0.0 STUB_GPUS=0 -- --gpus none --yes
assert_rc "compose 3.0 passes the version gate" 0 "$RC" "$OUT"

run nogpu_default STUB_GPUS=0 -- --yes
assert_rc "no GPU and no --gpus exits 1 (non-interactive)" 1 "$RC"
assert_grep "no GPU: suggests --gpus none" '--gpus none' "$OUT"

run gpuall_nogpu STUB_GPUS=0 -- --gpus all --yes
assert_rc "--gpus all with no GPU exits 1" 1 "$RC"

run gpuidx_bad STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 -- --gpus 0,5 --yes
assert_rc "--gpus with a nonexistent index exits 1" 1 "$RC"
assert_grep "bad index is named" 'GPU index 5 does not exist' "$OUT"

# ---- the check that matters: driver present, Docker runtime missing ---------------------

run noruntime STUB_GPUS=2 STUB_NVIDIA_RUNTIME=0 GPU_TOOLKIT_INSTALL=no -- --gpus all --yes
assert_rc "GPU present but no nvidia runtime: exit 2 (written, not startable)" 2 "$RC" "$OUT"
assert_grep "no runtime: explains driver != docker" 'Docker cannot pass them to containers' "$OUT"
assert_grep "no runtime: quotes the compose error" 'could not select device driver' "$OUT"
assert_grep "no runtime: files were still written" 'Files are in place' "$OUT"
[ -f "$D/docker-compose.override.yml" ] && pass "no runtime: override written for later" || fail "no runtime: override missing"

run check_noruntime STUB_GPUS=1 STUB_NVIDIA_RUNTIME=0 GPU_TOOLKIT_INSTALL=no -- --check
assert_rc "--check with no runtime exits 1" 1 "$RC"
[ ! -f "$D/.env" ] && pass "--check writes nothing" || fail "--check wrote .env"

run check_ok STUB_GPUS=1 STUB_NVIDIA_RUNTIME=1 -- --check
assert_rc "--check on a good host exits 0" 0 "$RC" "$OUT"
assert_grep "--check reports the runtime" 'Docker exposes the nvidia runtime' "$OUT"
assert_grep "--check reports the free disk with its number" 'Disk OK: 110 GB free on /var/lib/docker' "$OUT"

# ---- free disk on the Docker data root (#249) ------------------------------------------------

# 35 GB: under the 40 GB the stack wants, over the 20 GB floor. Warned, carried on.
run disk_low STUB_GPUS=0 STUB_DF_AVAIL_KB=34180000 -- --gpus none --yes
assert_rc "35 GB free: exits 0" 0 "$RC" "$OUT"
assert_grep "35 GB free: names the number and the data root" 'Only 35 GB free on /var/lib/docker' "$OUT"
assert_grep "35 GB free: names what the stack wants" 'about 40 GB' "$OUT"
assert_grep "35 GB free: names the image size as stored" '~29 GB as Docker stores it' "$OUT"
assert_grep "35 GB free: says it is carrying on" 'Carrying on' "$OUT"
assert_grep "35 GB free: still pulls" '^docker compose pull' "$CALLS"

# 15 GB and no api image in the store: a first pull cannot finish. Refused, nothing written.
run disk_floor STUB_GPUS=0 STUB_DF_AVAIL_KB=14650000 -- --gpus none --yes
assert_rc "15 GB free, first pull: exits 1" 1 "$RC" "$OUT"
assert_grep "15 GB free: says a first pull cannot finish" 'Only 15 GB free on /var/lib/docker.*first pull cannot finish' "$OUT"
assert_grep "15 GB free: says how to fix it" 'data-root in /etc/docker/daemon.json' "$OUT"
assert_grep "15 GB free: says nothing was written" 'Nothing was written' "$OUT"
[ ! -f "$D/.env" ] && pass "15 GB free: no .env written" || fail "15 GB free: .env was written"
assert_nogrep "15 GB free: no pull attempted" '^docker compose pull' "$CALLS"

# Same disk, but a release of the api image is already there: a pull shares its layers.
run disk_floor_haveimage STUB_GPUS=0 STUB_DF_AVAIL_KB=14650000 STUB_HAVE_IMAGE=1 -- --gpus none --yes
assert_rc "15 GB free with an api image present: exits 0" 0 "$RC" "$OUT"
assert_grep "15 GB free with image: explains the shared layers" 'already in Docker.s store' "$OUT"
[ -f "$D/.env" ] && pass "15 GB free with image: install proceeds" || fail "15 GB free with image: no .env"

# Same disk, --no-pull: nothing is pulled, so nothing to refuse; the note names make load-images.
run disk_floor_nopull STUB_GPUS=0 STUB_DF_AVAIL_KB=14650000 -- --gpus none --yes --no-pull
assert_rc "15 GB free with --no-pull: exits 0" 0 "$RC" "$OUT"
assert_grep "15 GB free with --no-pull: mentions load-images" 'make load-images' "$OUT"

# --check shares the floor: it is the preflight an operator runs to find this out.
run check_disk_floor STUB_GPUS=1 STUB_NVIDIA_RUNTIME=1 STUB_DF_AVAIL_KB=14650000 -- --check
assert_rc "--check at 15 GB free exits 1" 1 "$RC" "$OUT"
assert_grep "--check at 15 GB: names the disk as the failure" 'Preflight failed: not enough free disk' "$OUT"

run check_disk_low STUB_GPUS=1 STUB_NVIDIA_RUNTIME=1 STUB_DF_AVAIL_KB=34180000 -- --check
assert_rc "--check at 35 GB free exits 0 (warned)" 0 "$RC" "$OUT"
assert_grep "--check at 35 GB: still says Preflight OK" 'Preflight OK' "$OUT"

# The check itself cannot run: say so, carry on. Never a verdict from a df it could not parse.
run disk_df_fails STUB_GPUS=0 STUB_DF=fail -- --gpus none --yes
assert_rc "df fails: exits 0" 0 "$RC" "$OUT"
assert_grep "df fails: says the check was skipped" 'Could not measure free space on /var/lib/docker.*skipping' "$OUT"
assert_grep "df fails: still pulls" '^docker compose pull' "$CALLS"

run disk_no_root STUB_GPUS=0 STUB_DOCKER_ROOT= -- --gpus none --yes
assert_rc "no data root from docker info: exits 0" 0 "$RC" "$OUT"
assert_grep "no data root: says the check was skipped" 'Could not find the Docker data root.*skipping' "$OUT"

# ---- happy paths --------------------------------------------------------------------------

run none STUB_GPUS=0 -- --gpus none --yes
assert_rc "--gpus none --yes on a GPU-less host exits 0" 0 "$RC" "$OUT"
secret=$(env_val VW_COOKIE_SECRET "$D/.env")
[ "${#secret}" -eq 64 ] && pass "VW_COOKIE_SECRET generated (64 hex chars)" || fail "VW_COOKIE_SECRET is '$secret'"
echo "$secret" | grep -qE '^[0-9a-f]{64}$' && pass "VW_COOKIE_SECRET is hex" || fail "VW_COOKIE_SECRET not hex"
[ "$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")" = 0 ] && pass "VW_CONTAINER_GPU_COUNT=0 for --gpus none" || fail "VW_CONTAINER_GPU_COUNT=$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")"
[ "$(env_val VERSION "$D/.env")" = v2026.01.02.3 ] && pass "VERSION pinned to the changelog release" || fail "VERSION=$(env_val VERSION "$D/.env")"
assert_grep "override clears the GPU reservation" 'devices: !override \[\]' "$D/docker-compose.override.yml"
assert_grep "override hides GPUs from the runtime" 'NVIDIA_VISIBLE_DEVICES: "none"' "$D/docker-compose.override.yml"
assert_grep "override pins the api image to VERSION" 'vllm-warden:\$\{VERSION:-latest\}' "$D/docker-compose.override.yml"
assert_grep "override pins the ui image to VERSION" 'vllm-warden-ui:\$\{VERSION:-latest\}' "$D/docker-compose.override.yml"
assert_grep "override guards the secret with :?" 'VW_COOKIE_SECRET:\?' "$D/docker-compose.override.yml"
assert_grep "override overrides the front-door port" 'ports: !override' "$D/docker-compose.override.yml"
assert_grep "images were pulled" '^docker compose pull' "$CALLS"
assert_nogrep "stack not started without --start" '^docker compose up' "$CALLS"
assert_grep "compose config was validated" '^docker compose config -q' "$CALLS"
perm=$(ls -l "$D/.env" | cut -c1-10)
[ "$perm" = "-rw-------" ] && pass ".env is mode 600" || fail ".env mode is $perm"

run all STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 -- --gpus all --yes --start
assert_rc "--gpus all with 2 GPUs and a runtime exits 0" 0 "$RC" "$OUT"
[ "$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")" = 2 ] && pass "VW_CONTAINER_GPU_COUNT=2 for --gpus all" || fail "VW_CONTAINER_GPU_COUNT=$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")"
assert_grep "override reserves count: all" 'count: all' "$D/docker-compose.override.yml"
assert_grep "override exposes all GPUs" 'NVIDIA_VISIBLE_DEVICES: "all"' "$D/docker-compose.override.yml"
assert_grep "--start brings the stack up" '^docker compose up -d --no-build' "$CALLS"
assert_grep "summary prints the UI URL" 'http://localhost:8080/ui/' "$OUT"
assert_grep "override log prints the values, count first" 'Wrote docker-compose.override.yml \(release v2026.01.02.3, GPUs: 2 \(indices 0,1\) via nvidia, port 8080\)' "$OUT"
assert_nogrep "override log has no unexpanded variables" '\$\{VERSION\}|\$\{WARDEN_PORT\}' "$OUT"

run one STUB_GPUS=1 STUB_NVIDIA_RUNTIME=1 -- --gpus all --yes
assert_grep "one GPU reads as a count, not as index 0" 'GPUs: 1 \(index 0\)' "$OUT"

run subset STUB_GPUS=3 STUB_NVIDIA_RUNTIME=1 -- --gpus 2,0 --yes
assert_rc "--gpus 2,0 of 3 exits 0" 0 "$RC" "$OUT"
assert_grep "override pins device_ids" 'device_ids: \["2", "0"\]|device_ids: \["2","0"\]' "$D/docker-compose.override.yml"
assert_grep "override exposes only the subset" 'NVIDIA_VISIBLE_DEVICES: "2,0"' "$D/docker-compose.override.yml"
assert_grep "subset log says how many of how many" 'GPUs: 2 of 3 \(indices 2,0\)' "$OUT"
[ "$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")" = 2 ] && pass "VW_CONTAINER_GPU_COUNT=2 for a 2-GPU subset" || fail "VW_CONTAINER_GPU_COUNT=$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")"

run fullsubset STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 -- --gpus 0,1 --yes
assert_grep "listing every GPU collapses to count: all" 'count: all' "$D/docker-compose.override.yml"

# ---- GPUs through CDI (#254) -------------------------------------------------------------
# The legacy nvidia hook grants the GPUs behind the cgroup manager's back, and a
# systemd reload revokes them from the running container. CDI does not have
# that problem, so it is the default whenever Docker and the toolkit offer it.

run cdi_all STUB_GPUS=2 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0,1 STUB_CGROUP=systemd VW_HOST_ROOT="$WORK/host-cdi_all" -- --gpus all --yes
assert_rc "CDI: --gpus all exits 0 without the nvidia runtime" 0 "$RC" "$OUT"
assert_grep "CDI: override uses driver: cdi" 'driver: cdi' "$D/docker-compose.override.yml"
assert_grep "CDI: override requests nvidia.com/gpu=all" 'device_ids: \["nvidia.com/gpu=all"\]' "$D/docker-compose.override.yml"
assert_nogrep "CDI: no legacy nvidia driver left" 'driver: nvidia' "$D/docker-compose.override.yml"
assert_grep "CDI: the legacy hook is told to inject nothing" 'NVIDIA_VISIBLE_DEVICES: "void"' "$D/docker-compose.override.yml"
assert_grep "CDI: the log says why" 'through CDI' "$OUT"
assert_grep "CDI: override log names the request" 'GPUs: 2 \(indices 0,1\) via cdi' "$OUT"
assert_nogrep "CDI: no nvidia-runtime complaint" 'Docker cannot pass them' "$OUT"
assert_grep "systemd host: udev rule installed" 'create-dev-char-symlinks --create-all' "$WORK/host-cdi_all/etc/udev/rules.d/71-nvidia-dev-char.rules"
assert_grep "systemd host: symlinks created now" '^nvidia-ctk system create-dev-char-symlinks --create-all' "$CALLS"
[ "$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")" = 2 ] && pass "CDI: VW_CONTAINER_GPU_COUNT=2" || fail "CDI: VW_CONTAINER_GPU_COUNT=$(env_val VW_CONTAINER_GPU_COUNT "$D/.env")"

run cdi_subset STUB_GPUS=3 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0,1,2 -- --gpus 2,0 --yes
assert_rc "CDI subset exits 0" 0 "$RC" "$OUT"
assert_grep "CDI subset: one device per index, in order" 'device_ids: \["nvidia.com/gpu=2", "nvidia.com/gpu=0"\]' "$D/docker-compose.override.yml"

run cdi_cgroupfs STUB_GPUS=2 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0,1 VW_HOST_ROOT="$WORK/host-cdi_cgroupfs" -- --gpus all --yes
assert_grep "cgroupfs host: still CDI" 'driver: cdi' "$D/docker-compose.override.yml"
[ ! -e "$WORK/host-cdi_cgroupfs/etc/udev/rules.d/71-nvidia-dev-char.rules" ] && pass "cgroupfs host: no udev rule" || fail "cgroupfs host: udev rule written"

run cdi_hostprep_no STUB_GPUS=2 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0,1 STUB_CGROUP=systemd GPU_HOST_PREP=no VW_HOST_ROOT="$WORK/host-cdi_no" -- --gpus all --yes
[ ! -e "$WORK/host-cdi_no/etc/udev/rules.d/71-nvidia-dev-char.rules" ] && pass "GPU_HOST_PREP=no: no udev rule" || fail "GPU_HOST_PREP=no: udev rule written"
assert_nogrep "GPU_HOST_PREP=no: no symlinks created" 'create-dev-char-symlinks' "$CALLS"

run cdi_missing_index STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=0 -- --gpus 0,1 --yes
assert_grep "CDI spec lacks a device: falls back to the hook" 'driver: nvidia' "$D/docker-compose.override.yml"
assert_grep "CDI spec lacks a device: says which" 'no device for: all' "$OUT"

run cdi_disabled_systemd STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 STUB_CTK=1 STUB_CDI=all,0,1 STUB_CGROUP=systemd -- --gpus all --yes
assert_rc "CDI off: legacy hook still installs" 0 "$RC" "$OUT"
assert_grep "CDI off: legacy driver" 'driver: nvidia' "$D/docker-compose.override.yml"
assert_grep "CDI off: says Docker has CDI disabled" 'Docker has CDI disabled' "$OUT"
assert_grep "CDI off on systemd: warns about the reload" 'Failed to initialize' "$OUT"

run cdi_off_cgroupfs STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 -- --gpus all --yes
assert_nogrep "cgroupfs host on the hook: no systemd warning" 'Failed to initialize' "$OUT"

run force_nvidia STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0,1 GPU_REQUEST=nvidia -- --gpus all --yes
assert_grep "GPU_REQUEST=nvidia wins over CDI" 'driver: nvidia' "$D/docker-compose.override.yml"

run force_cdi_unavailable STUB_GPUS=2 STUB_NVIDIA_RUNTIME=1 GPU_REQUEST=cdi -- --gpus all --yes
assert_rc "GPU_REQUEST=cdi without CDI: exit 2 (written, not startable)" 2 "$RC" "$OUT"
assert_grep "GPU_REQUEST=cdi without CDI: explains" 'GPU_REQUEST=cdi, but' "$OUT"

run bad_request STUB_GPUS=2 GPU_REQUEST=hook -- --gpus all --yes
assert_rc "unknown GPU_REQUEST exits 1" 1 "$RC" "$OUT"
assert_grep "unknown GPU_REQUEST is named" "GPU_REQUEST must be auto, cdi or nvidia \(got 'hook'\)" "$OUT"

run check_cdi STUB_GPUS=1 STUB_CTK=1 STUB_DOCKER_CDI=1 STUB_CDI=all,0 STUB_CGROUP=systemd VW_HOST_ROOT="$WORK/host-check_cdi" -- --check
assert_rc "--check passes on CDI without the nvidia runtime" 0 "$RC" "$OUT"
[ ! -e "$WORK/host-check_cdi/etc/udev/rules.d/71-nvidia-dev-char.rules" ] && pass "--check writes no udev rule" || fail "--check wrote the udev rule"
assert_nogrep "--check creates no symlinks" 'create-dev-char-symlinks' "$CALLS"

run flags STUB_GPUS=0 -- --gpus none --yes --origin 'https://llm.example.com,http://10.0.0.5:8080' --port 9090 --version v2026.02.03.1 --no-pull
assert_rc "origin/port/version/no-pull exits 0" 0 "$RC" "$OUT"
[ "$(env_val VW_FRONTEND_ORIGIN "$D/.env")" = 'https://llm.example.com,http://10.0.0.5:8080' ] && pass "--origin written verbatim" || fail "VW_FRONTEND_ORIGIN=$(env_val VW_FRONTEND_ORIGIN "$D/.env")"
[ "$(env_val WARDEN_PORT "$D/.env")" = 9090 ] && pass "--port written" || fail "WARDEN_PORT=$(env_val WARDEN_PORT "$D/.env")"
[ "$(env_val VERSION "$D/.env")" = v2026.02.03.1 ] && pass "--version overrides the changelog pin" || fail "VERSION=$(env_val VERSION "$D/.env")"
assert_nogrep "--no-pull skips docker compose pull" '^docker compose pull' "$CALLS"
assert_grep "summary shows the chosen port" 'localhost:9090' "$OUT"
[ "$(grep -c '^VW_FRONTEND_ORIGIN=' "$D/.env")" = 1 ] && pass "env_set replaces in place (one VW_FRONTEND_ORIGIN line)" || fail "duplicate VW_FRONTEND_ORIGIN lines"

run nosecrets STUB_GPUS=0 -- --gpus none --yes --no-generate-secrets
assert_rc "--no-generate-secrets exits 2 (written, not startable)" 2 "$RC" "$OUT"
[ -z "$(env_val VW_COOKIE_SECRET "$D/.env")" ] && pass "secret left blank" || fail "secret was generated anyway"
assert_grep "says which variable to fill" 'VW_COOKIE_SECRET is empty' "$OUT"
assert_grep "config still validated with a placeholder" '^docker compose config -q' "$CALLS"

# re-run keeps .env
run rerun STUB_GPUS=0 -- --gpus none --yes
first=$(env_val VW_COOKIE_SECRET "$D/.env")
sed 's/^VW_JWT_SECRET=.*/VW_JWT_SECRET=keep-me/' "$D/.env" > "$D/.env.tmp" && mv "$D/.env.tmp" "$D/.env"
set +e; (cd "$D" && env PATH="$STUBS:$PATH" STUB_GPUS=0 "$SH" ./install.sh --gpus none --yes --port 8181) >"$OUT" 2>&1; RC=$?; set -e
assert_rc "re-run exits 0" 0 "$RC" "$OUT"
[ "$(env_val VW_COOKIE_SECRET "$D/.env")" = "$first" ] && pass "re-run keeps the existing secret" || fail "re-run rotated the secret"
[ "$(env_val VW_JWT_SECRET "$D/.env")" = keep-me ] && pass "re-run keeps operator edits" || fail "re-run lost VW_JWT_SECRET"
[ "$(env_val WARDEN_PORT "$D/.env")" = 8181 ] && pass "re-run applies new flags" || fail "WARDEN_PORT=$(env_val WARDEN_PORT "$D/.env")"
assert_grep "re-run says it kept .env" 'Existing .env kept' "$OUT"

# blank secret in an old .env gets filled on re-run
run refill STUB_GPUS=0 -- --gpus none --yes
sed 's/^VW_COOKIE_SECRET=.*/VW_COOKIE_SECRET= # REQUIRED | generate:hex32/' "$D/.env" > "$D/.env.tmp" && mv "$D/.env.tmp" "$D/.env"
set +e; (cd "$D" && env PATH="$STUBS:$PATH" STUB_GPUS=0 "$SH" ./install.sh --gpus none --yes) >"$OUT" 2>&1; RC=$?; set -e
assert_rc "re-run with a blanked secret exits 0" 0 "$RC" "$OUT"
s=$(env_val VW_COOKIE_SECRET "$D/.env"); [ "${#s}" -eq 64 ] && pass "blank secret refilled on re-run" || fail "blank secret not refilled"

# ---- --dir: install into a separate directory -----------------------------------------------

run dir STUB_GPUS=0 -- --gpus none --yes --dir "$WORK/target/opt-vllm-warden"
assert_rc "--dir into a new directory exits 0" 0 "$RC" "$OUT"
T="$WORK/target/opt-vllm-warden"
for f in docker-compose.yml deploy/caddy/Caddyfile .env.example Makefile install.sh .env docker-compose.override.yml CHANGELOG.md; do
  [ -f "$T/$f" ] && pass "--dir: $f present" || fail "--dir: $f missing"
done
cmp -s "$T/Makefile" "$ROOT/make/operator.mk" && pass "--dir: Makefile is make/operator.mk verbatim" || fail "--dir: Makefile differs from make/operator.mk"
[ -x "$T/install.sh" ] && pass "--dir: install.sh is executable" || fail "--dir: install.sh not executable"
[ ! -f "$D/.env" ] && pass "--dir: source checkout left untouched" || fail "--dir: .env written into the source checkout"

# ---- piped from curl: script on stdin, no /dev/tty needed with --yes ----------------------------

D="$WORK/case-piped"; rm -rf "$D"; cp -R "$SRC" "$D"; OUT="$WORK/out-piped.log"; : > "$CALLS"
set +e; (cd "$D" && env PATH="$STUBS:$PATH" STUB_GPUS=0 "$SH" -s -- --gpus none --yes --no-pull < ./install.sh) >"$OUT" 2>&1; RC=$?; set -e
assert_rc "piped (sh -s -- ... < install.sh) exits 0" 0 "$RC" "$OUT"
[ -f "$D/.env" ] && pass "piped: .env written into the cwd checkout" || fail "piped: no .env"

# ---- make/operator.mk: parses and lists targets -------------------------------------------------

if command -v make >/dev/null 2>&1; then
  D="$WORK/case-dir"; OUT="$WORK/out-make.log"
  set +e; (cd "$WORK/target/opt-vllm-warden" && env PATH="$STUBS:$PATH" make -n start stop restart logs status pull config uninstall save-images load-images export-hf-cache import-hf-cache help) >"$OUT" 2>&1; RC=$?; set -e
  assert_rc "operator Makefile: every target dry-runs" 0 "$RC" "$OUT"
  set +e; (cd "$WORK/target/opt-vllm-warden" && make help) >"$OUT" 2>&1; RC=$?; set -e
  assert_grep "operator Makefile: help lists start" 'make start' "$OUT"
  assert_grep "operator Makefile: help lists uninstall" 'make uninstall' "$OUT"
else
  printf '  skip make not available\n'
fi

printf '\n%d passed, %d failed\n' "$n_pass" "$n_fail"
[ "$n_fail" -eq 0 ]
