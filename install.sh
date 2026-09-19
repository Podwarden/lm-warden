#!/bin/sh
# LLM Warden installer.
#
# Installs the stack described by docker-compose.yml onto this host using the
# published release images, and writes the two files that are specific to
# this machine: .env (secrets, port, release) and docker-compose.override.yml
# (images, GPU passthrough, health checks). Nothing here talks to any server
# other than the container registry you pull images from, and with --no-pull
# not even that -- see documents/INSTALL.md "Offline / air-gapped install".
#
# Usage, from a checkout:
#   ./install.sh                       interactive: picks GPUs, asks to start
#   ./install.sh --gpus all --yes      unattended, all GPUs, no prompts
#   ./install.sh --gpus none --yes     CPU-only control plane (evaluation, CI)
#   ./install.sh --check               host preflight only, writes nothing
#
# Or without a checkout (downloads the source tree into --dir first):
#   curl -fsSL https://raw.githubusercontent.com/Podwarden/vllm-warden/main/install.sh | sh -s -- --dir /opt/vllm-warden
#
# The script is POSIX sh (dash, busybox ash and bash all run it) and reads
# every prompt from /dev/tty, so it behaves the same piped from curl as it
# does from a file. Re-running it is safe: an existing .env is kept and only
# blank secrets are filled in; the override file is regenerated every time.
#
# Exit codes: 0 installed and startable, 1 a preflight or argument problem
# (nothing or little written), 2 files written but the stack cannot start yet
# (the message says what is missing -- typically the NVIDIA runtime).

set -eu

VW_APP="vllm-warden"
VW_SOURCE_URL="${VW_SOURCE_URL:-https://github.com/Podwarden/vllm-warden/archive/refs/heads/main.tar.gz}"
VW_REGISTRY="${VW_REGISTRY:-registry.podwarden.com/podwarden/apps}"
COMPOSE_MIN_MAJOR=2
COMPOSE_MIN_MINOR=24
# Free space the stack needs on the Docker data root, in GB (10^9 bytes, the
# unit docker itself reports in). Measured on v2026.09.04.2: the api image is
# 9.2 GB compressed on the wire and unpacks to 19.7 GB of layers; with the
# containerd image store (the default on a fresh Docker Engine 29) the
# compressed layers are kept next to the unpacked ones, so the image costs
# ~29 GB as Docker stores it. documents/INSTALL.md "What it will cost you in
# disk" carries the same numbers.
DISK_IMAGE_GB=29     # the api image alone, as Docker stores it
DISK_STACK_GB=40     # the stack with room to run, before any model is pulled
DISK_FLOOR_GB=20     # under this the api image cannot even be unpacked
: "${GPU_TOOLKIT_INSTALL:=}"
: "${TERM:=}"

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

if [ -t 2 ] && [ "$TERM" != "dumb" ]; then
  C_GREEN=$(printf '\033[0;32m'); C_YELLOW=$(printf '\033[1;33m'); C_RED=$(printf '\033[0;31m')
  C_BOLD=$(printf '\033[1m'); C_DIM=$(printf '\033[2m'); C_OFF=$(printf '\033[0m')
else
  C_GREEN=""; C_YELLOW=""; C_RED=""; C_BOLD=""; C_DIM=""; C_OFF=""
fi
log()  { printf '%s[%s]%s %s\n' "$C_GREEN" "$VW_APP" "$C_OFF" "$*" >&2; }
warn() { printf '%s[%s]%s %s\n' "$C_YELLOW" "$VW_APP" "$C_OFF" "$*" >&2; }
err()  { printf '%s[%s]%s %s\n' "$C_RED" "$VW_APP" "$C_OFF" "$*" >&2; }
die()  { err "$@"; exit 1; }

usage() {
  cat <<EOF
LLM Warden installer

Usage: install.sh [options]

  --dir PATH              Install directory. Default: this checkout when run
                          from one, else /opt/${VW_APP} (root) or
                          \$HOME/${VW_APP}.
  --version TAG           Release to run (image tag, e.g. v2026.09.03.5, or
                          latest). Default: the newest release named in
                          CHANGELOG.md, else latest.
  --gpus VALUE            GPUs to pass to the engine: all | none | 0,1,...
                          (nvidia-smi indices). Default: ask on a terminal,
                          all otherwise. none installs a CPU-only control
                          plane that cannot load models.
  --origin URL[,URL]      Public URL(s) of the UI -> VW_FRONTEND_ORIGIN.
  --port N                Host port of the front door -> WARDEN_PORT (8080).
  --no-generate-secrets   Leave VW_COOKIE_SECRET for you to fill in.
  --no-pull               Do not pull images (air-gapped: docker load first).
  --start / --no-start    Start the stack when done / never. Default: ask on
                          a terminal, do not start otherwise.
  -y, --yes               Never prompt; take the defaults above.
  --check                 Run the host preflight only (Docker, Compose, free
                          disk, GPUs, nvidia runtime). Writes nothing.
  -h, --help              This text.

Environment:
  GPU_TOOLKIT_INSTALL=yes|no   Install the NVIDIA Container Toolkit without
                               asking when Docker lacks the nvidia runtime
                               (yes), or never (no). Unset: ask on a terminal.
                               "yes" RESTARTS THE DOCKER DAEMON, bouncing every
                               container on this host, not only LLM Warden's.
  GPU_REQUEST=auto|cdi|nvidia  How the engine asks Docker for its GPUs. auto
                               (default) uses CDI when Docker has it enabled and
                               the NVIDIA CDI spec names the GPUs, else the
                               legacy nvidia hook -- whose GPU grant a systemd
                               reload can silently revoke.
  GPU_HOST_PREP=no             Do not create the NVIDIA /dev/char symlinks or
                               install the udev rule that keeps them (systemd
                               cgroup hosts only; restarts nothing).
  VW_SOURCE_URL                Tarball to download when not run from a checkout.
  VW_REGISTRY                  Image registry prefix (${VW_REGISTRY}).
EOF
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------

DIR=""; GPUS=""; ORIGIN=""; PORT=""; VERSION_FLAG=""
GEN_SECRETS=1; PULL=1; START=""; ASSUME_YES=0; CHECK_ONLY=0

need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value (see --help)"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)      need_arg "$@"; DIR=$2; shift 2 ;;
    --dir=*)    DIR=${1#*=}; shift ;;
    --version)  need_arg "$@"; VERSION_FLAG=$2; shift 2 ;;
    --version=*) VERSION_FLAG=${1#*=}; shift ;;
    --gpus)     need_arg "$@"; GPUS=$2; shift 2 ;;
    --gpus=*)   GPUS=${1#*=}; shift ;;
    --origin)   need_arg "$@"; ORIGIN=$2; shift 2 ;;
    --origin=*) ORIGIN=${1#*=}; shift ;;
    --port)     need_arg "$@"; PORT=$2; shift 2 ;;
    --port=*)   PORT=${1#*=}; shift ;;
    --no-generate-secrets) GEN_SECRETS=0; shift ;;
    --no-pull)  PULL=0; shift ;;
    --start)    START=1; shift ;;
    --no-start) START=0; shift ;;
    -y|--yes)   ASSUME_YES=1; shift ;;
    --check)    CHECK_ONLY=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) err "Unknown argument: $1"; usage >&2; exit 1 ;;
  esac
done

case "$GPUS" in
  ""|all|none) ;;
  *) echo "$GPUS" | grep -Eq '^[0-9]+(,[0-9]+)*$' || die "--gpus must be all, none or a comma-separated list of indices (got '$GPUS')" ;;
esac
if [ -n "$PORT" ]; then
  echo "$PORT" | grep -Eq '^[0-9]+$' && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "--port must be 1-65535 (got '$PORT')"
fi

# ---------------------------------------------------------------------------
# prompts -- always via /dev/tty so `curl | sh` can still ask
# ---------------------------------------------------------------------------

# 0 when a person can answer a question right now.
interactive() {
  [ "$ASSUME_YES" -eq 0 ] || return 1
  [ -r /dev/tty ] || return 1
  (exec 9</dev/tty) 2>/dev/null
}

# ask_yn "question" default(y|n) -> 0 for yes
ask_yn() {
  _q=$1; _d=$2
  if [ "$_d" = y ]; then _h="[Y/n]"; else _h="[y/N]"; fi
  printf '%s %s ' "$_q" "$_h" >&2
  if ! read -r _a </dev/tty; then _a=""; fi
  case "$_a" in
    "") [ "$_d" = y ] ;;
    y|Y|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# where is the source tree, where does the install go
# ---------------------------------------------------------------------------

SRC=""
if [ -f "$0" ]; then
  _sd=$(cd "$(dirname "$0")" && pwd)
else
  _sd=$PWD
fi
if [ -f "$_sd/docker-compose.yml" ] && [ -f "$_sd/deploy/caddy/Caddyfile" ] && [ -f "$_sd/.env.example" ]; then
  SRC=$_sd
fi

if [ -z "$DIR" ]; then
  if [ -n "$SRC" ]; then
    DIR=$SRC
  elif [ "$(id -u)" = "0" ]; then
    DIR="/opt/$VW_APP"
  else
    DIR="$HOME/$VW_APP"
  fi
fi

# ---------------------------------------------------------------------------
# preflight: docker, compose >= 2.24, free disk, GPUs, nvidia runtime
# ---------------------------------------------------------------------------

preflight_docker() {
  command -v docker >/dev/null 2>&1 || {
    err "Docker is not installed."
    err "  https://docs.docker.com/engine/install/   (or: curl -fsSL https://get.docker.com | sh)"
    return 1
  }
  if ! docker info >/dev/null 2>&1; then
    err "Docker is installed but the daemon is not reachable."
    if [ "$(id -u)" != "0" ]; then
      err "  Either run as root, or add yourself to the docker group:"
      err "    sudo usermod -aG docker $(id -un) && newgrp docker"
    else
      err "  Is it running?  systemctl status docker"
    fi
    return 1
  fi
  _cv=$(docker compose version --short 2>/dev/null) || {
    err "Docker Compose v2 is not available (docker compose version failed)."
    err "  Install the compose plugin: https://docs.docker.com/compose/install/linux/"
    return 1
  }
  _cv=${_cv#v}
  _maj=${_cv%%.*}; _rest=${_cv#*.}; _min=${_rest%%.*}
  case "$_maj$_min" in *[!0-9]*|"") err "Could not parse Docker Compose version '$_cv'."; return 1 ;; esac
  if [ "$_maj" -lt "$COMPOSE_MIN_MAJOR" ] || { [ "$_maj" -eq "$COMPOSE_MIN_MAJOR" ] && [ "$_min" -lt "$COMPOSE_MIN_MINOR" ]; }; then
    err "Docker Compose $_cv is too old: ${COMPOSE_MIN_MAJOR}.${COMPOSE_MIN_MINOR} or newer is required."
    err "  docker-compose.override.yml uses the !override tag to replace the GPU"
    err "  reservation instead of appending to it; older Compose would merge the"
    err "  two lists and ask for more GPUs than the host has."
    return 1
  fi
  log "Docker OK, Compose $_cv."
  return 0
}

# 0 when some release of the api image is already in Docker's store. Pulling
# another release on top of it shares the base layers, which are nearly all
# of the ~29 GB, so the free-space floor below only applies to a first pull.
api_image_present() {
  [ -n "$(docker images -q "$VW_REGISTRY/vllm-warden" 2>/dev/null)" ]
}

# Free space where the images will land: the Docker data root, not / --
# /var/lib/docker is routinely its own filesystem. Every other preflight
# passes on a 30 GB disk, and then `docker compose pull` dies partway through
# with "no space left on device" (#249), so this one says the number up
# front. Sets DISK_ROOT, DISK_FREE_GB and DISK_STATE:
#   ok       at least DISK_STACK_GB free
#   low      less than that: warned, carrying on
#   floor    less than DISK_FLOOR_GB and the api image still has to be
#            pulled: the caller refuses (or asks, on a terminal)
#   unknown  the check itself could not run: said so, carrying on
# Like the toolkit preflight it never fails on its own account: a df it
# could not parse is a warning, not a verdict.
preflight_disk() {
  DISK_ROOT=""; DISK_FREE_GB=""; DISK_STATE=unknown
  DISK_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null) || DISK_ROOT=""
  if [ -z "$DISK_ROOT" ]; then
    warn "Could not find the Docker data root (docker info); skipping the free-space check."
    warn "  The stack needs about ${DISK_STACK_GB} GB there before any model is pulled."
    return 0
  fi
  # -P: one line per filesystem in the POSIX column order; -k: 1024-byte
  # blocks. Column 4 is what is available to us, which on ext4 is less than
  # what is free.
  _kb=$(df -Pk "$DISK_ROOT" 2>/dev/null | awk 'NR == 2 { print $4 }')
  case "$_kb" in
    ""|*[!0-9]*)
      warn "Could not measure free space on $DISK_ROOT (df); skipping the free-space check."
      warn "  The stack needs about ${DISK_STACK_GB} GB there before any model is pulled."
      return 0 ;;
  esac
  # 1 GB = 976,562.5 blocks of 1024 bytes.
  DISK_FREE_GB=$((_kb / 976563))
  if [ "$_kb" -ge $((DISK_STACK_GB * 976563)) ]; then
    DISK_STATE=ok
    log "Disk OK: ${DISK_FREE_GB} GB free on $DISK_ROOT (Docker data root)."
    return 0
  fi
  DISK_STATE=low
  if [ "$_kb" -lt $((DISK_FLOOR_GB * 976563)) ] && [ "$PULL" -eq 1 ] && ! api_image_present; then
    DISK_STATE=floor
    err "Only ${DISK_FREE_GB} GB free on $DISK_ROOT (Docker data root): a first pull cannot finish."
  else
    warn "Only ${DISK_FREE_GB} GB free on $DISK_ROOT (Docker data root); the stack wants about ${DISK_STACK_GB} GB there."
  fi
  warn "  The api image alone takes ~${DISK_IMAGE_GB} GB as Docker stores it (9.2 GB compressed on the wire,"
  warn "  19.7 GB unpacked, and the containerd image store keeps both), and models come on top:"
  warn "  a 7B AWQ checkpoint is ~5 GB, and the HuggingFace cache grows with every pull."
  if [ "$DISK_STATE" = floor ]; then
    warn "  Grow the disk, or move the Docker data root (data-root in /etc/docker/daemon.json), and re-run."
  elif api_image_present; then
    warn "  A release of the api image is already in Docker's store, so a pull shares its base"
    warn "  layers and needs far less than a first one; carrying on."
  elif [ "$PULL" -eq 0 ]; then
    warn "  --no-pull: loading the images yourself (make load-images) needs the same room; carrying on."
  else
    warn "  The pull may end with: no space left on device. Carrying on."
  fi
  return 0
}

# One line per GPU: "index|name|memory MiB". Empty when nvidia-smi is absent
# or fails (no driver).
gpu_detect() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' 'NF >= 2 && $1 ~ /^[0-9]+$/ { printf "%s|%s|%s\n", $1, $2, $3 }' || true
}

gpu_count_lines() { printf '%s\n' "$1" | sed '/^$/d' | wc -l | tr -d ' '; }

gpu_print_list() {
  printf '%s\n' "$1" | sed '/^$/d' | while IFS='|' read -r _i _n _m; do
    printf '    [%s] %s (%s MiB)\n' "$_i" "$_n" "$_m" >&2
  done
}

# A working nvidia-smi proves the DRIVER is there. It says nothing about
# whether Docker can hand a GPU to a container: that needs the NVIDIA
# Container Toolkit registered with dockerd as the "nvidia" runtime. An
# install that skips this check looks clean and then dies at the first
# `docker compose up` with
#   could not select device driver "nvidia" with capabilities: [[gpu]]
nvidia_runtime_ready() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

toolkit_installed() {
  command -v nvidia-ctk >/dev/null 2>&1 || command -v nvidia-container-runtime >/dev/null 2>&1
}

pkg_mgr() {
  if command -v apt-get >/dev/null 2>&1; then echo apt
  elif command -v dnf >/dev/null 2>&1; then echo dnf
  elif command -v yum >/dev/null 2>&1; then echo yum
  elif command -v zypper >/dev/null 2>&1; then echo zypper
  else echo ""
  fi
}

as_root() {
  if [ "$(id -u)" = "0" ]; then "$@"
  elif command -v sudo >/dev/null 2>&1; then sudo "$@"
  else err "Need root for: $*"; return 1
  fi
}

install_nvidia_toolkit() {
  _pm=$(pkg_mgr)
  [ -n "$_pm" ] || { warn "No supported package manager (apt/dnf/yum/zypper); cannot install the toolkit automatically."; return 1; }
  log "Installing the NVIDIA Container Toolkit via $_pm..."
  case "$_pm" in
    apt)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | as_root gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg || return 1
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | as_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null || return 1
      as_root apt-get update >/dev/null || return 1
      as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit || return 1 ;;
    dnf|yum)
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | as_root tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null || return 1
      as_root "$_pm" install -y nvidia-container-toolkit || return 1 ;;
    zypper)
      as_root zypper --non-interactive ar -f \
        https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo >/dev/null 2>&1 || true
      as_root zypper --non-interactive --gpg-auto-import-keys install -y nvidia-container-toolkit || return 1 ;;
  esac
}

configure_nvidia_runtime() {
  command -v nvidia-ctk >/dev/null 2>&1 || { warn "nvidia-ctk not found; cannot register the runtime with Docker."; return 1; }
  as_root nvidia-ctk runtime configure --runtime=docker || return 1
  log "Restarting Docker so it picks up the nvidia runtime..."
  if command -v systemctl >/dev/null 2>&1; then as_root systemctl restart docker || return 1
  elif command -v service >/dev/null 2>&1; then as_root service docker restart || return 1
  else warn "Could not restart Docker automatically; restart it yourself, then: make start"; return 1
  fi
  _i=0
  while [ "$_i" -lt 30 ]; do
    docker info >/dev/null 2>&1 && return 0
    _i=$((_i + 1)); sleep 1
  done
  warn "Docker did not come back within 30 s of the restart."
  return 1
}

# 0 when Docker can serve nvidia devices, possibly after installing the
# toolkit. 1 when it cannot; the caller decides how loudly to fail.
ensure_nvidia_runtime() {
  nvidia_runtime_ready && { log "NVIDIA Container Toolkit OK: Docker exposes the nvidia runtime."; return 0; }
  warn "NVIDIA GPUs are present, but Docker cannot pass them to containers:"
  if toolkit_installed; then
    warn "  the NVIDIA Container Toolkit is installed but not registered with Docker."
  else
    warn "  the NVIDIA Container Toolkit is not installed."
  fi
  warn "  Starting the stack as-is fails with:"
  warn "    could not select device driver \"nvidia\" with capabilities: [[gpu]]"

  # Registering the runtime means writing /etc/docker/daemon.json and
  # RESTARTING dockerd, which restarts every container on this host -- not
  # just ours. On a box that already runs other services that is a real
  # outage, and it must never be a surprise. Said before the prompt so the
  # answer is informed, and before the unattended run so the log records it.
  warn "  Doing that RESTARTS the Docker daemon: every container on this host"
  warn "  stops and starts again, including ones that have nothing to do with"
  warn "  LLM Warden. Containers with a restart policy come back; anything"
  warn "  started without one does not."

  _do=$GPU_TOOLKIT_INSTALL
  if [ -z "$_do" ]; then
    if interactive && ask_yn "  Install it and restart Docker now?" y; then _do=yes; else _do=no; fi
  fi
  if [ "$_do" != "yes" ]; then
    warn "Skipping. Install it yourself and re-run ./install.sh (or: make preflight):"
    warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    warn "  Unattended: GPU_TOOLKIT_INSTALL=yes ./install.sh ...  (restarts Docker)"
    return 1
  fi
  if toolkit_installed || install_nvidia_toolkit; then
    if configure_nvidia_runtime && nvidia_runtime_ready; then
      log "NVIDIA Container Toolkit is ready; Docker can now use the GPUs."
      return 0
    fi
  fi
  warn "Could not make the nvidia runtime available automatically."
  warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  return 1
}

# ---------------------------------------------------------------------------
# How the api service asks Docker for its GPUs (#254)
# ---------------------------------------------------------------------------
#
#   cdi     driver: cdi, device_ids nvidia.com/gpu=all | nvidia.com/gpu=<idx>.
#           Docker adds the device nodes itself, so its cgroup manager -- systemd
#           on most distributions -- knows about them.
#   nvidia  driver: nvidia, the NVIDIA Container Toolkit's legacy hook. The hook
#           grants the GPUs behind the cgroup manager's back. With the systemd
#           cgroup driver the next `systemctl daemon-reload` (unattended
#           upgrades run one) re-applies a device list without them: every
#           process in the container that opens a GPU from then on gets
#           "Failed to initialize NVML: Unknown Error". The engine already
#           running keeps serving, the VRAM gauges read 0, and the next model
#           load fails -- silently, until someone restarts the container.
#
# CDI is used whenever Docker has it enabled and the toolkit's CDI spec names
# every selected GPU; GPU_REQUEST=cdi|nvidia forces one.

# Docker 25+ reports the directories it reads CDI specs from; an empty list
# means CDI is off (it is on by default from Docker 28).
docker_cdi_enabled() {
  _dirs=$(docker info --format '{{json .CDISpecDirs}}' 2>/dev/null) || return 1
  case "$_dirs" in ""|null|"[]") return 1 ;; esac
  return 0
}

# nvidia.com/gpu=... device names in the toolkit's CDI spec, one per line.
# The toolkit (1.18+) regenerates the spec itself (nvidia-cdi-refresh).
cdi_gpu_names() {
  command -v nvidia-ctk >/dev/null 2>&1 || return 0
  nvidia-ctk cdi list 2>/dev/null | grep -E '^nvidia\.com/gpu=' || true
}

docker_cgroup_driver() { docker info --format '{{.CgroupDriver}}' 2>/dev/null || true; }

# Sets GPU_REQUEST to cdi or nvidia for the selection in GPU_MODE/GPU_SELECTED.
# Returns 1 only when GPU_REQUEST=cdi was asked for and CDI cannot serve it.
choose_gpu_request() {
  _want=${GPU_REQUEST:-auto}
  case "$_want" in
    auto|cdi|nvidia) ;;
    *) die "GPU_REQUEST must be auto, cdi or nvidia (got '$_want')." ;;
  esac
  GPU_REQUEST=nvidia
  [ "$_want" != nvidia ] || { log "GPU_REQUEST=nvidia: requesting the GPUs through the legacy nvidia hook."; return 0; }

  _why=""
  if ! docker_cdi_enabled; then
    _why="Docker has CDI disabled (Docker 28+ enables it by default)"
  else
    _names=$(cdi_gpu_names)
    if [ -z "$_names" ]; then
      _why="no NVIDIA CDI spec was found (nvidia-ctk cdi list shows no nvidia.com/gpu devices)"
    else
      _missing=""
      if [ "$GPU_MODE" = all ]; then
        printf '%s\n' "$_names" | grep -qx 'nvidia.com/gpu=all' || _missing=" all"
      else
        for _i in $(printf '%s' "$GPU_SELECTED" | tr ',' ' '); do
          printf '%s\n' "$_names" | grep -qx "nvidia.com/gpu=$_i" || _missing="$_missing $_i"
        done
      fi
      [ -z "$_missing" ] || _why="the NVIDIA CDI spec has no device for:$_missing (regenerate it: nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml)"
    fi
  fi

  if [ -z "$_why" ]; then
    GPU_REQUEST=cdi
    log "Requesting the GPUs through CDI (nvidia.com/gpu); the grant survives systemd reloads."
    return 0
  fi
  if [ "$_want" = cdi ]; then
    err "GPU_REQUEST=cdi, but $_why."
    return 1
  fi
  log "Requesting the GPUs through the nvidia hook: $_why."
  if [ "$(docker_cgroup_driver)" = systemd ]; then
    warn "Docker uses the systemd cgroup driver. With the nvidia hook, the next"
    warn "  systemctl daemon-reload (unattended upgrades run one) takes the GPUs away"
    warn "  from the running container: nvidia-smi then fails with 'Failed to initialize"
    warn "  NVML: Unknown Error', the VRAM gauges read 0 and new model loads fail until"
    warn "  the container restarts. Enable CDI and re-run ./install.sh to avoid it:"
    warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html"
  fi
  return 0
}

# systemd grants a container's devices through /dev/char/<major>:<minor>
# symlinks, and the NVIDIA driver creates none for its device nodes. Without
# them even a CDI grant can be dropped on the next reload. NVIDIA's documented
# fix: create them now, and install a udev rule that recreates them whenever
# the driver binds (i.e. every boot). Restarts nothing. GPU_HOST_PREP=no skips
# it; VW_HOST_ROOT redirects the rule file (tests only).
NVIDIA_DEV_CHAR_RULE=/etc/udev/rules.d/71-nvidia-dev-char.rules
ensure_nvidia_dev_char() {
  [ "${GPU_HOST_PREP:-yes}" != no ] || { log "GPU_HOST_PREP=no: leaving /dev/char symlinks and udev rules alone."; return 0; }
  [ "$(docker_cgroup_driver)" = systemd ] || return 0
  command -v nvidia-ctk >/dev/null 2>&1 || { warn "nvidia-ctk not found; cannot create the NVIDIA /dev/char symlinks systemd needs."; return 0; }
  _rule="${VW_HOST_ROOT:-}$NVIDIA_DEV_CHAR_RULE"
  if [ ! -f "$_rule" ]; then
    _body='# Installed by the LLM Warden installer (#254). systemd grants container
# devices through /dev/char/<major>:<minor> symlinks, which the NVIDIA driver
# does not create; without them a systemd reload can revoke GPU access from
# running containers. Recreate them whenever the driver binds.
ACTION=="add", DEVPATH=="/bus/pci/drivers/nvidia", RUN+="/usr/bin/nvidia-ctk system create-dev-char-symlinks --create-all"'
    if [ -n "${VW_HOST_ROOT:-}" ]; then
      mkdir -p "$(dirname "$_rule")" && printf '%s\n' "$_body" > "$_rule"
    else
      printf '%s\n' "$_body" | as_root tee "$_rule" >/dev/null
    fi || { warn "Could not write $NVIDIA_DEV_CHAR_RULE; the /dev/char symlinks will not come back after a reboot."; return 0; }
    log "Installed $NVIDIA_DEV_CHAR_RULE (recreates the NVIDIA /dev/char symlinks at boot)."
  fi
  if [ -n "${VW_HOST_ROOT:-}" ]; then
    nvidia-ctk system create-dev-char-symlinks --create-all >/dev/null 2>&1
  else
    as_root nvidia-ctk system create-dev-char-symlinks --create-all >/dev/null 2>&1
  fi || warn "nvidia-ctk system create-dev-char-symlinks failed; run it as root before starting."
  return 0
}

# Resolves GPUS ("" | all | none | list) against the detected GPUs. Sets:
#   GPU_SELECTED  comma-separated indices, or "" for none
#   GPU_TOTAL     number of GPUs detected
#   GPU_MODE      all | subset | none
resolve_gpus() {
  GPU_LINES=$(gpu_detect)
  GPU_TOTAL=$(gpu_count_lines "$GPU_LINES")
  GPU_ALL_IDX=$(printf '%s\n' "$GPU_LINES" | sed '/^$/d' | cut -d'|' -f1 | paste -sd, -)

  if [ "$GPU_TOTAL" -gt 0 ]; then
    log "Detected $GPU_TOTAL NVIDIA GPU(s):"
    gpu_print_list "$GPU_LINES"
  fi

  case "$GPUS" in
    none)
      GPU_SELECTED=""; GPU_MODE=none
      log "GPU passthrough disabled (--gpus none): CPU-only control plane, models cannot be loaded." ;;
    all)
      [ "$GPU_TOTAL" -gt 0 ] || die "--gpus all, but no NVIDIA GPU was detected (nvidia-smi missing or failing). Install the driver, or use --gpus none."
      GPU_SELECTED=$GPU_ALL_IDX; GPU_MODE=all ;;
    "")
      if [ "$GPU_TOTAL" -eq 0 ]; then
        warn "No NVIDIA GPU detected (nvidia-smi missing or failing)."
        if interactive && ask_yn "  Continue with a CPU-only control plane? Models cannot be loaded without a GPU." n; then
          GPU_SELECTED=""; GPU_MODE=none
        else
          die "LLM Warden needs an NVIDIA GPU. Install the driver and re-run, or pass --gpus none for a CPU-only control plane (evaluation, CI)."
        fi
      elif [ "$GPU_TOTAL" -gt 1 ] && interactive; then
        printf '  GPUs to pass to the engine, e.g. 0,1 (blank = all): ' >&2
        if ! read -r _pick </dev/tty; then _pick=""; fi
        _pick=$(printf '%s' "$_pick" | tr -d ' ')
        if [ -z "$_pick" ] || [ "$_pick" = all ]; then
          GPU_SELECTED=$GPU_ALL_IDX; GPU_MODE=all
        else
          echo "$_pick" | grep -Eq '^[0-9]+(,[0-9]+)*$' || die "Expected indices like 0,1 (got '$_pick')."
          GPUS=$_pick; resolve_gpus_subset
        fi
      else
        GPU_SELECTED=$GPU_ALL_IDX; GPU_MODE=all
        [ "$GPU_TOTAL" -le 1 ] || log "No terminal to ask on: passing through all $GPU_TOTAL GPUs (restrict with --gpus)."
      fi ;;
    *) resolve_gpus_subset ;;
  esac

  if [ "$GPU_MODE" = all ]; then
    log "Passing through all $GPU_TOTAL GPU(s)."
  elif [ "$GPU_MODE" = subset ]; then
    log "Passing through GPU(s) $GPU_SELECTED of $GPU_TOTAL."
  fi
}

resolve_gpus_subset() {
  [ "$GPU_TOTAL" -gt 0 ] || die "--gpus $GPUS, but no NVIDIA GPU was detected (nvidia-smi missing or failing)."
  _sel=""; _n=0
  for _i in $(echo "$GPUS" | tr ',' ' '); do
    printf '%s\n' "$GPU_LINES" | grep -q "^$_i|" || die "GPU index $_i does not exist on this host (have: $GPU_ALL_IDX)."
    case ",$_sel," in *",$_i,"*) continue ;; esac
    _sel="${_sel:+$_sel,}$_i"; _n=$((_n + 1))
  done
  GPU_SELECTED=$_sel
  if [ "$_n" -eq "$GPU_TOTAL" ]; then GPU_MODE=all; else GPU_MODE=subset; fi
}

gpu_selected_count() {
  [ -n "$GPU_SELECTED" ] || { echo 0; return; }
  echo "$GPU_SELECTED" | tr ',' '\n' | wc -l | tr -d ' '
}

# The selection for a log line, count first: "1 (index 0)", "2 of 3 (indices
# 2,0)", "none (CPU-only)". A bare index list reads as a count -- "GPUs: 0"
# on a one-GPU host looks like no GPU at all.
gpu_describe() {
  _n=$(gpu_selected_count)
  case "$GPU_MODE" in
    none) echo "none (CPU-only)" ;;
    all)  if [ "$_n" -eq 1 ]; then echo "1 (index $GPU_SELECTED)"; else echo "$_n (indices $GPU_SELECTED)"; fi ;;
    *)    if [ "$_n" -eq 1 ]; then echo "1 of $GPU_TOTAL (index $GPU_SELECTED)"; else echo "$_n of $GPU_TOTAL (indices $GPU_SELECTED)"; fi ;;
  esac
}

# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

# env_set KEY VALUE: replace the active KEY= line in .env (keeping its place
# in the file) or append one. The value is passed through the environment,
# never through a regex, so URLs and secrets with special characters are
# written verbatim.
env_set() {
  K="$1" V="$2" awk '
    BEGIN { k = ENVIRON["K"]; v = ENVIRON["V"]; done = 0 }
    !done && index($0, k "=") == 1 { print k "=" v; done = 1; next }
    { print }
    END { if (!done) print k "=" v }
  ' .env > .env.tmp && mv .env.tmp .env && chmod 600 .env
}

# env_get KEY: the value of the active KEY= line, minus a trailing comment.
env_get() {
  sed -n "s/^$1=//p" .env | head -1 | sed 's/[[:space:]]*#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//'
}

rand_hex() { # rand_hex BYTES
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex "$1"
  else
    head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

# Fill every `VAR=  # ... generate:<strategy>` line whose value is still
# empty. The marker grammar is shared with the PodWarden Hub catalog so the
# same .env.example drives both installers.
generate_secrets() {
  _count=0
  _pairs=$(sed -n 's/^\([A-Z_][A-Z0-9_]*\)=[[:space:]]*#.*generate:\([a-z0-9]*\).*/\1 \2/p' .env)
  for _pair in $(printf '%s\n' "$_pairs" | tr ' ' ':'); do
    _var=${_pair%%:*}; _how=${_pair#*:}
    case "$_how" in
      hex16) _val=$(rand_hex 16) ;;
      hex32) _val=$(rand_hex 32) ;;
      hex64) _val=$(rand_hex 64) ;;
      *) warn "Unknown generate strategy '$_how' for $_var; using 32 random bytes as hex."; _val=$(rand_hex 32) ;;
    esac
    env_set "$_var" "$_val"
    _count=$((_count + 1))
  done
  [ "$_count" -eq 0 ] || log "Generated $_count secret(s) in .env."
}

# Newest release named in the changelog (## [vYYYY.MM.DD.N]); empty if none.
changelog_release() {
  for _f in CHANGELOG.md changelog.md; do
    [ -f "$_f" ] || continue
    grep -m1 -oE '^## \[v[0-9][^]]*\]' "$_f" | sed 's/^## \[//; s/\]$//'
    return 0
  done
}

# ---------------------------------------------------------------------------
# docker-compose.override.yml
# ---------------------------------------------------------------------------

write_override() {
  _f=docker-compose.override.yml
  _sel=$(gpu_selected_count)
  {
    echo "# Generated by install.sh -- re-run ./install.sh to change anything here."
    echo "# Hand edits are overwritten on the next run."
    echo "#"
    echo "# This file turns the stack contract in docker-compose.yml into a runnable"
    echo "# install of the published release images. It adds nothing that is not a"
    echo "# choice for this particular host: which release, which GPUs, which port,"
    echo "# and how the .env values reach the api container."
    echo "#"
    echo "# The GPU device list carries the !override tag because Compose MERGES"
    echo "# sequences across files by APPENDING: without it the base file's"
    echo "# reservation of every GPU would be added to the selection below."
    echo "# Requires Compose v2.24+ (install.sh checks)."
    echo "services:"
    echo "  api:"
    echo "    image: $VW_REGISTRY/vllm-warden:\${VERSION:-latest}"
    echo "    pull_policy: missing"
    echo "    restart: unless-stopped"
    echo "    # vLLM's tensor-parallel workers talk over /dev/shm; the 64 MiB default"
    echo "    # kills any model with tensor_parallel_size > 1 with SIGBUS mid-serving."
    echo "    shm_size: 2g"
    echo "    environment:"
    echo "      VW_COOKIE_SECRET: \"\${VW_COOKIE_SECRET:?VW_COOKIE_SECRET is empty in .env -- re-run ./install.sh, or set it to the output of: openssl rand -hex 32}\""
    echo "      VW_JWT_SECRET: \"\${VW_JWT_SECRET:-}\""
    echo "      VW_FRONTEND_ORIGIN: \"\${VW_FRONTEND_ORIGIN:-}\""
    echo "      VW_TRUST_PROXY_ORIGIN: \"\${VW_TRUST_PROXY_ORIGIN:-0}\""
    echo "      VW_CONTAINER_GPU_COUNT: \"\${VW_CONTAINER_GPU_COUNT:-$_sel}\""
    echo "      VW_WARMUP_PROBE_TIMEOUT_S: \"\${VW_WARMUP_PROBE_TIMEOUT_S:-600.0}\""
    echo "      VW_STATS_SAMPLER_INTERVAL_S: \"\${VW_STATS_SAMPLER_INTERVAL_S:-5.0}\""
    echo "      VW_HEADER_METRICS_INTERVAL_S: \"\${VW_HEADER_METRICS_INTERVAL_S:-2.0}\""
    echo "      VW_REQUEST_MAX_WALL_S: \"\${VW_REQUEST_MAX_WALL_S:-0.0}\""
    echo "      VW_GODMODE_ENABLED: \"\${VW_GODMODE_ENABLED:-false}\""
    # Content logging + the runaway detector that reports through it. Off by
    # default, and forwarded here only so that setting them in .env actually
    # reaches the container -- Compose passes nothing to a service that does
    # not name it, so without these lines the knobs .env.example documents
    # would silently do nothing.
    echo "      VW_CONTENT_LOG_ENABLED: \"\${VW_CONTENT_LOG_ENABLED:-false}\""
    echo "      VW_CONTENT_LOG_TOKENS: \"\${VW_CONTENT_LOG_TOKENS:-}\""
    # Deliberately NO literal default: blank means "derive from VW_DATA_DIR"
    # (<VW_DATA_DIR>/logs/content.jsonl). Pinned to /data/logs/content.jsonl
    # it escaped a moved VW_DATA_DIR onto the container's writable layer.
    echo "      VW_CONTENT_LOG_PATH: \"\${VW_CONTENT_LOG_PATH:-}\""
    echo "      VW_CONTENT_LOG_MAX_CHARS: \"\${VW_CONTENT_LOG_MAX_CHARS:-40000}\""
    # File-size ceiling, 512 MiB. On reaching it the logger STOPS and warns
    # once; it does not rotate and does not delete.
    echo "      VW_CONTENT_LOG_MAX_BYTES: \"\${VW_CONTENT_LOG_MAX_BYTES:-536870912}\""
    echo "      VW_RUNAWAY_MODE: \"\${VW_RUNAWAY_MODE:-off}\""
    echo "      HF_HUB_OFFLINE: \"\${HF_HUB_OFFLINE:-0}\""
    if [ "$GPU_MODE" != none ] && [ "${GPU_REQUEST:-nvidia}" = cdi ]; then
      # CDI hands over the devices; "void" stops the legacy hook (should the
      # nvidia runtime be Docker's default) from injecting any on top.
      echo "      NVIDIA_VISIBLE_DEVICES: \"void\""
    else
      case "$GPU_MODE" in
        none) echo "      NVIDIA_VISIBLE_DEVICES: \"none\"" ;;
        all)  echo "      NVIDIA_VISIBLE_DEVICES: \"all\"" ;;
        *)    echo "      NVIDIA_VISIBLE_DEVICES: \"$GPU_SELECTED\"" ;;
      esac
    fi
    echo "    deploy:"
    echo "      resources:"
    echo "        reservations:"
    case "$GPU_MODE:${GPU_REQUEST:-nvidia}" in
      none:*)
        echo "          # --gpus none: no GPU reservation (the base file asks for all)."
        echo "          devices: !override []" ;;
      *:cdi)
        echo "          # CDI (nvidia.com/gpu): Docker registers the devices with the cgroup"
        echo "          # manager, so a systemd reload cannot revoke them (#254)."
        echo "          devices: !override"
        echo "            - driver: cdi"
        if [ "$GPU_MODE" = all ]; then
          echo "              device_ids: [\"nvidia.com/gpu=all\"]"
        else
          echo "              device_ids: [$(echo "$GPU_SELECTED" | sed 's/\([0-9][0-9]*\)/"nvidia.com\/gpu=\1"/g; s/,/, /g')]"
        fi
        echo "              capabilities: [gpu]" ;;
      all:*)
        echo "          devices: !override"
        echo "            - driver: nvidia"
        echo "              count: all"
        echo "              capabilities: [gpu]" ;;
      *)
        echo "          devices: !override"
        echo "            - driver: nvidia"
        echo "              device_ids: [$(echo "$GPU_SELECTED" | sed 's/\([0-9][0-9]*\)/"\1"/g')]"
        echo "              capabilities: [gpu]" ;;
    esac
    echo "    # /healthz answers once the control plane is up; model loads are lazy."
    echo "    # Generous tolerances: a model pull can starve the event loop for a"
    echo "    # while, and that must not flip the container to unhealthy."
    echo "    healthcheck:"
    echo "      test: [\"CMD\", \"python3\", \"-c\", \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=8)\"]"
    echo "      interval: 30s"
    echo "      timeout: 10s"
    echo "      retries: 10"
    echo "      start_period: 60s"
    echo "  ui:"
    echo "    image: $VW_REGISTRY/vllm-warden-ui:\${VERSION:-latest}"
    echo "    pull_policy: missing"
    echo "    restart: unless-stopped"
    echo "  caddy:"
    echo "    ports: !override"
    echo "      - \"\${WARDEN_PORT:-8080}:8080\""
  } > "$_f"
  _via=""; [ "$GPU_MODE" = none ] || _via=" via ${GPU_REQUEST:-nvidia}"
  log "Wrote $_f (release $VERSION, GPUs: $(gpu_describe)$_via, port $WARDEN_PORT)."
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

printf '\n%s%sLLM Warden%s %sinstaller%s\n\n' "$C_BOLD" "$C_GREEN" "$C_OFF" "$C_DIM" "$C_OFF" >&2

log "Checking this host..."
preflight_docker || exit 1
preflight_disk

GPU_RUNTIME_OK=1
GPU_REQUEST_OK=1
resolve_gpus
if [ "$GPU_MODE" != none ]; then
  if ! choose_gpu_request; then
    GPU_REQUEST_OK=0
  elif [ "$GPU_REQUEST" = cdi ]; then
    # CDI is Docker's own device path: the nvidia runtime is not needed.
    # --check writes nothing, host files included.
    [ "$CHECK_ONLY" -eq 1 ] || ensure_nvidia_dev_char
  else
    ensure_nvidia_runtime || GPU_RUNTIME_OK=0
  fi
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  [ "$GPU_RUNTIME_OK" -eq 1 ] && [ "$GPU_REQUEST_OK" -eq 1 ] || { err "Preflight failed: Docker cannot use the GPUs (see above)."; exit 1; }
  [ "$DISK_STATE" != floor ] || { err "Preflight failed: not enough free disk on $DISK_ROOT for the images (see above)."; exit 1; }
  log "Preflight OK."; exit 0
fi

# Below the floor a first pull is certain to fail, so refuse before writing
# anything -- unless a person is there to overrule.
if [ "$DISK_STATE" = floor ]; then
  if interactive && ask_yn "  Continue anyway? The pull will most likely fail." n; then
    warn "Carrying on at your request."
  else
    die "Not enough free disk on $DISK_ROOT for a first pull of the images (see above). Nothing was written."
  fi
fi

# ---- source tree -> install directory --------------------------------------

TMP=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

if [ -z "$SRC" ]; then
  command -v curl >/dev/null 2>&1 || die "curl is required to download the source tree (or run install.sh from a checkout)."
  command -v tar >/dev/null 2>&1 || die "tar is required to unpack the source tree."
  TMP=$(mktemp -d)
  log "Downloading the source tree from $VW_SOURCE_URL ..."
  curl -fsSL "$VW_SOURCE_URL" -o "$TMP/src.tar.gz" || die "Download failed. Clone the repository instead and run ./install.sh from it."
  mkdir -p "$TMP/src"
  tar -xzf "$TMP/src.tar.gz" -C "$TMP/src" --strip-components=1 || die "Could not unpack $TMP/src.tar.gz."
  SRC="$TMP/src"
fi

if [ "$DIR" != "$SRC" ]; then
  if [ ! -d "$DIR" ]; then
    if ! mkdir -p "$DIR" 2>/dev/null; then
      log "Creating $DIR (needs elevated permissions)..."
      as_root mkdir -p "$DIR" || exit 1
      as_root chown "$(id -u):$(id -g)" "$DIR" || exit 1
    fi
  fi
  [ -w "$DIR" ] || die "$DIR is not writable by $(id -un). Pick another --dir or fix its ownership."
  log "Installing into $DIR ..."
  mkdir -p "$DIR/deploy/caddy"
  cp "$SRC/docker-compose.yml" "$DIR/docker-compose.yml"
  cp "$SRC/deploy/caddy/Caddyfile" "$DIR/deploy/caddy/Caddyfile"
  cp "$SRC/.env.example" "$DIR/.env.example"
  cp "$SRC/make/operator.mk" "$DIR/Makefile"
  cp "$SRC/install.sh" "$DIR/install.sh" && chmod +x "$DIR/install.sh"
  for _opt in README.md LICENSE CHANGELOG.md changelog.md; do
    [ -f "$SRC/$_opt" ] && cp "$SRC/$_opt" "$DIR/$_opt"
  done
fi
cd "$DIR"

# ---- .env -------------------------------------------------------------------

ENV_STATE=kept
if [ ! -f .env ]; then
  cp .env.example .env && chmod 600 .env
  ENV_STATE=fresh
  log "Created .env from .env.example."
else
  log "Existing .env kept; only blank secrets and the flags you passed are written."
fi

if [ "$GEN_SECRETS" -eq 1 ]; then
  generate_secrets
else
  warn "--no-generate-secrets: fill in VW_COOKIE_SECRET in .env before starting."
fi

if [ -n "$VERSION_FLAG" ]; then
  env_set VERSION "$VERSION_FLAG"
elif [ "$ENV_STATE" = fresh ]; then
  _rel=$(changelog_release)
  if [ -n "$_rel" ]; then
    env_set VERSION "$_rel"
    log "Pinned VERSION=$_rel, the release this source tree documents (override with --version)."
  fi
fi
[ -z "$PORT" ] || env_set WARDEN_PORT "$PORT"
[ -z "$ORIGIN" ] || env_set VW_FRONTEND_ORIGIN "$ORIGIN"
env_set VW_CONTAINER_GPU_COUNT "$(gpu_selected_count)"

VERSION=$(env_get VERSION); VERSION=${VERSION:-latest}
WARDEN_PORT=$(env_get WARDEN_PORT); WARDEN_PORT=${WARDEN_PORT:-8080}

INCOMPLETE=""
if [ -z "$(env_get VW_COOKIE_SECRET)" ]; then
  INCOMPLETE="VW_COOKIE_SECRET is empty in .env (set it to the output of: openssl rand -hex 32)"
fi

# ---- override + validation ---------------------------------------------------

write_override

# Validate the merged config. A deliberately blank secret would trip the :?
# guard here, so give the check a placeholder in that one case; the real
# `make start` still refuses until it is filled in.
if [ -n "$INCOMPLETE" ]; then
  VW_COOKIE_SECRET=placeholder-for-validation-only docker compose config -q || die "docker compose config rejected the generated files (see above)."
else
  docker compose config -q || die "docker compose config rejected the generated files (see above)."
fi
log "Compose configuration validates."

# ---- images -----------------------------------------------------------------

if [ "$PULL" -eq 1 ]; then
  log "Pulling release images ($VERSION)..."
  docker compose pull || die "Image pull failed. Offline? Load the images first (see documents/INSTALL.md: Offline / air-gapped install) and re-run with --no-pull."
else
  log "--no-pull: not pulling images; the stack expects them to be loaded already (make load-images)."
fi

# ---- start? -----------------------------------------------------------------

if [ "$GPU_RUNTIME_OK" -eq 0 ]; then
  INCOMPLETE="${INCOMPLETE:+$INCOMPLETE; }Docker has no nvidia runtime, so the GPU reservation cannot be satisfied"
fi
if [ "$GPU_REQUEST_OK" -eq 0 ]; then
  INCOMPLETE="${INCOMPLETE:+$INCOMPLETE; }GPU_REQUEST=cdi, but Docker cannot serve the GPUs through CDI"
fi

STARTED=0
if [ -z "$INCOMPLETE" ]; then
  if [ "$START" = 1 ] || { [ -z "$START" ] && interactive && ask_yn "Start LLM Warden now?" y; }; then
    log "Starting..."
    docker compose up -d --no-build --remove-orphans
    STARTED=1
  fi
fi

# ---- summary ----------------------------------------------------------------

printf '\n' >&2
if [ -n "$INCOMPLETE" ]; then
  err "Files are in place in $DIR, but the stack cannot start yet:"
  err "  $INCOMPLETE."
  err "Fix that, then: cd $DIR && make start"
  exit 2
fi

printf '%s%sLLM Warden is installed in %s%s\n\n' "$C_BOLD" "$C_GREEN" "$DIR" "$C_OFF" >&2
if [ "$STARTED" -eq 1 ]; then
  printf '  UI:      http://localhost:%s/ui/    (first run opens the setup wizard)\n' "$WARDEN_PORT" >&2
  printf '  API:     http://localhost:%s/v1/chat/completions\n' "$WARDEN_PORT" >&2
  printf '  Logs:    cd %s && make logs\n' "$DIR" >&2
else
  printf '  Start:   cd %s && make start\n' "$DIR" >&2
  printf '  Then:    http://localhost:%s/ui/\n' "$WARDEN_PORT" >&2
fi
printf '  Config:  %s/.env   (release: %s, port: %s, GPUs: %s)\n' "$DIR" "$VERSION" "$WARDEN_PORT" "$(gpu_describe)" >&2
printf '  Help:    make help\n\n' >&2
exit 0
