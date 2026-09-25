# shellcheck shell=sh
# ---------------------------------------------------------------------------
# Resolve the prebuilt CI dependency image for the current requirements pair.
#
# WHY. Every Python job in .gitlab-ci.yml used to `pip install -r
# requirements.txt -r requirements-dev.txt` straight from PyPI: 22-55 s in each
# of four jobs, 130-200 runner-s per pipeline, to reinstall a pinned dependency
# set that changes a couple of times a month. `make test-unit` / `make lint` /
# `make typecheck` paid the same cost on every local run and failed outright
# whenever PyPI was slow (observed twice on 2026-09-03).
#
# The image tag is a sha256 of the two requirements files, so an unchanged pair
# always resolves to the same tag and only a real dependency change misses.
# `build:ci-deps-image` in .gitlab-ci.yml is what builds and pushes it.
#
# GRACEFUL DEGRADATION IS THE WHOLE POINT. When the tag is not available --
# the first pipeline after this landed, a registry outage, a pruned tag, a
# runner with no pull route -- this falls back to exactly the old behaviour:
# the plain base image plus the same `pip install`. A CI change that hard-fails
# on a cold cache is worse than the slow status quo it replaces.
#
# USAGE. Source it (do not execute it) so the exports survive into the caller:
#
#     . ./scripts/ci-deps-image.sh                       # python-only flavour
#     export VW_PY_FLAVOR=pynode; . ./scripts/ci-deps-image.sh
#
# Inputs (all optional):
#     VW_PY_FLAVOR   `py` (default, python:3.11-slim) or `pynode`
#                    (nikolaik/python-nodejs, which conformance:chat2 needs
#                    because it runs the backend and the node conformance
#                    runner on the same loopback interface)
#     VW_DEPS_PULL   attempt a registry pull on a local miss. Implied in CI
#                    ($CI). Off by default on a workstation so a machine with
#                    no route to registry.podwarden.com falls back instantly
#                    instead of blocking on a doomed pull.
#     VW_DEPS_BUILD  build the image locally on a miss (the CI builder job and
#                    `make deps-image`).
#     VW_DEPS_PUSH   push after building. Set by the CI builder job and by
#                    `make deps-image` -- the arm64 tag exists only because a
#                    workstation pushed it, since CI runs on amd64 runners and
#                    does not build arm64. Best-effort: a push failure warns
#                    and leaves the locally built image in place.
#
# Exports:
#     VW_PY_TAG      requirements digest (16 hex chars) plus `-<arch>`, so an
#                    amd64 runner and an arm64 workstation never collide on
#                    one tag
#     VW_ARCH        normalised machine arch: amd64 / arm64 / other
#     VW_DEPS_IMAGE  the prebuilt image ref for this flavour
#     VW_BASE_IMAGE  the plain base image this flavour falls back to
#     VW_RUN_IMAGE   what to `docker run`: VW_DEPS_IMAGE on a hit, VW_BASE_IMAGE
#                    on a miss
#     VW_PIP         what to run inside the container first: `:` (a no-op) on a
#                    hit, the pip install on a miss. Callers pass it in with
#                    `-e VW_PIP` and start their inner script with
#                    `eval "$VW_PIP"`, which keeps the existing single-quoted
#                    job bodies untouched.
#
# NOTE ON `set -e`. This file is sourced into a shell the GitLab runner already
# put in `set -e`, so every command here is either inside an `if` condition or
# `|| true`-guarded, and the file ends on a deliberate status: non-zero only
# when an explicitly requested build/push failed.
# ---------------------------------------------------------------------------

: "${PYTHON_IMAGE:=python:3.11-slim}"
: "${PYNODE_IMAGE:=nikolaik/python-nodejs:python3.11-nodejs20-slim}"
: "${REGISTRY_NS:=registry.podwarden.com/podwarden/apps}"
: "${VW_PY_FLAVOR:=py}"

VW_DEPS_FAILED=0

# sha256sum on the Linux runners, shasum on a macOS workstation. Hash the
# concatenated CONTENT rather than `sha256sum file file`, whose output embeds
# the paths and would change if the files ever move.
if command -v sha256sum >/dev/null 2>&1; then
  VW_PY_TAG=$(cat requirements.txt requirements-dev.txt | sha256sum | cut -c1-16)
else
  VW_PY_TAG=$(cat requirements.txt requirements-dev.txt | shasum -a 256 | cut -c1-16)
fi

# The tag carries the ARCHITECTURE as well as the requirements digest, because
# a digest alone made an Apple-silicon workstation strictly slower than having
# no prebuilt image at all. `make deps-image` pulls before it builds; the CI
# builder runs on amd64 runners, so the registry held an amd64-only tag; and
# `docker pull` on an arm64 Mac happily fetches it and runs it under QEMU.
# The pull therefore SUCCEEDED, `vw_deps_have_image` went true, and the native
# local build never ran -- turning a 3-4 minute emulated mypy into the normal
# case and leaving the operator to conclude the tooling was simply slow.
#
# With the arch in the tag the arm64 pull misses (nothing pushes that tag), so
# the same `make deps-image` falls through to a local build that is native and
# takes about as long as the pip install it replaces, once.
VW_ARCH=$(uname -m 2>/dev/null || echo unknown)
case "$VW_ARCH" in
  x86_64|amd64)   VW_ARCH=amd64 ;;
  aarch64|arm64)  VW_ARCH=arm64 ;;
  *)              VW_ARCH=$(printf '%s' "$VW_ARCH" | tr -c 'a-zA-Z0-9_.-' '-') ;;
esac
VW_PY_TAG="$VW_PY_TAG-$VW_ARCH"

case "$VW_PY_FLAVOR" in
  pynode)
    VW_DEPS_IMAGE="$REGISTRY_NS/lm-warden/ci-pynode:$VW_PY_TAG"
    VW_BASE_IMAGE="$PYNODE_IMAGE"
    ;;
  *)
    VW_DEPS_IMAGE="$REGISTRY_NS/lm-warden/ci-py:$VW_PY_TAG"
    VW_BASE_IMAGE="$PYTHON_IMAGE"
    ;;
esac

vw_deps_have_image() {
  docker image inspect "$VW_DEPS_IMAGE" >/dev/null 2>&1
}

vw_deps_build() {
  # `-f -` reads the Dockerfile from stdin with `.` as the build context, so
  # the recipe lives here beside the resolver instead of drifting in a second
  # file. Installed system-wide as root: the jobs run the container as an
  # unprivileged --user, which can read /usr/local but could not write it, and
  # /usr/local/bin is already on PATH so `ruff`/`pytest`/`mypy` resolve with no
  # PATH gymnastics. requirements-dev.txt starts with `-r requirements.txt`,
  # which pip resolves relative to the file, hence both are copied.
  echo "ci-deps: building $VW_DEPS_IMAGE from $VW_BASE_IMAGE"
  docker build -t "$VW_DEPS_IMAGE" -f - . <<EOF
FROM $VW_BASE_IMAGE
COPY requirements.txt requirements-dev.txt /tmp/req/
RUN pip install --no-cache-dir --disable-pip-version-check -r /tmp/req/requirements-dev.txt \\
 && rm -rf /tmp/req
EOF
}

if vw_deps_have_image; then
  :
elif [ -n "${CI:-}" ] || [ -n "${VW_DEPS_PULL:-}" ]; then
  docker pull -q "$VW_DEPS_IMAGE" >/dev/null 2>&1 || true
fi

if [ -n "${VW_DEPS_BUILD:-}" ] && ! vw_deps_have_image; then
  if vw_deps_build; then
    if [ -n "${VW_DEPS_PUSH:-}" ]; then
      # Best-effort, and loud about it. A push failure must not red-flag a
      # pipeline whose jobs will simply fall back to pip install -- but it must
      # not be silent either, because the symptom is "CI is mysteriously slow
      # again" rather than a failure.
      docker push "$VW_DEPS_IMAGE" \
        || echo "ci-deps: WARNING push of $VW_DEPS_IMAGE failed; jobs will fall back to pip install"
    fi
  else
    echo "ci-deps: ERROR build of $VW_DEPS_IMAGE failed" >&2
    VW_DEPS_FAILED=1
  fi
fi

if vw_deps_have_image; then
  VW_RUN_IMAGE="$VW_DEPS_IMAGE"
  VW_PIP=":"
  echo "ci-deps: HIT  $VW_DEPS_IMAGE (dependencies baked in)"
else
  docker pull -q "$VW_BASE_IMAGE" >/dev/null 2>&1 || true
  VW_RUN_IMAGE="$VW_BASE_IMAGE"
  VW_PIP="pip install --user -q -r requirements.txt -r requirements-dev.txt"
  echo "ci-deps: MISS $VW_DEPS_IMAGE -- falling back to $VW_BASE_IMAGE + pip install"
fi

export VW_PY_TAG VW_DEPS_IMAGE VW_BASE_IMAGE VW_RUN_IMAGE VW_PIP

# Evict superseded tags of these two repositories, on CI runners only.
# Nothing else would: the build jobs' `docker image prune -f` collects dangling
# layers, and these are tagged. vllm-warden#208 was this project filling
# runner03/04's 38 GB disks, so an image that parks another ~700 MB copy on
# every runner at every dependency bump is not a theoretical problem. Guarded
# on $CI so a workstation's images are never touched, and `rm -f` on an image a
# concurrent job is still running only untags it -- that container keeps going,
# and a job that has not started yet re-pulls.
#
# The pre-rename `vllm-warden/ci-*` and `llm-warden/ci-*` repositories (renamed
# to lm-warden/ci-* in 2026-09) are matched too and never kept: nothing reads
# them any more, so every such tag on a runner is dead weight.
if [ -n "${CI:-}" ]; then
  docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E "^$REGISTRY_NS/(v?llm|lm)-warden/ci-(py|pynode):" \
    | grep -vE "^$REGISTRY_NS/lm-warden/ci-(py|pynode):$VW_PY_TAG\$" \
    | xargs -r -n1 docker image rm -f >/dev/null 2>&1 || true
fi

# Last statement, deliberately: non-zero only when a build we were explicitly
# asked to do failed. Consumer jobs never set VW_DEPS_BUILD, so for them this
# file always ends 0 even on a full miss.
[ "$VW_DEPS_FAILED" = "0" ]
