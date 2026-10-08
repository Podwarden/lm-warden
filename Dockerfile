# syntax=docker/dockerfile:1
# ^ Must be the FIRST line — a parser directive is only recognised before any
# other comment or instruction. Pins the Dockerfile frontend so `RUN --mount=
# type=cache` below means the same thing on every runner instead of depending
# on whichever BuildKit the host's pw-builder happens to ship. Same directive,
# same reason, as frontend/Dockerfile.
#
# TODO(release): replace the digest in VW_BASE_DIGEST below by running
# `docker pull vllm/vllm-openai:v0.26.0` and copying the sha256 from
# `docker images --digests vllm/vllm-openai`.
#
# v0.26.0 (was v0.25.1, sha256:e4f88a83...). Upgraded 2026-08-17 to diagnose an
# EngineCore that dies mid-serving with NO traceback, NO CUDA error and no OOM
# kill (node-wide /proc/vmstat oom_kill stayed 0 across nine deaths). v0.26.0
# adds "log worker exit code when a process dies unexpectedly" (#38641), which
# names the fatal signal instead of leaving us to infer it. Also carries a
# DSv3.2 + MTP sequence-parallel accuracy fix; this deployment runs MTP.
#
# Declared as an ARG, not inlined into FROM, so this line is the SINGLE place
# the base digest lives. CI's post-build cleanup greps it out of this file to
# decide which vllm/vllm-openai layer sets are superseded and safe to evict
# (vllm-warden#208). Inlining it would let the Dockerfile and the cleanup drift,
# and a cleanup that reads a stale digest deletes the base you are still using.
ARG VW_BASE_DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52

# llama.cpp build, pinned. Sub-project C, decision D2: llama-server is baked
# into this image and launched by the existing in-container subprocess driver,
# so its version is a WARDEN release, not a per-model choice. Bumping it means
# bumping this ARG and cutting a release -- and re-capturing
# tests/fixtures/llamacpp/, whose README records the tag this must match.
# tests/unit/system/test_llamacpp_version.py fails if the two drift.
#
# Declared as a greppable ARG for the same reason VW_BASE_DIGEST above is: it is
# the SINGLE place the pin lives, so CI and the docs cannot read a stale value.
#
# WHY WE BUILD RATHER THAN COPY. ghcr.io/ggml-org/llama.cpp:server-cuda is
# mainline and prebuilt, and it does not work here: it is built on Ubuntu 24.04
# and needs GLIBC_2.38 / GLIBCXX_3.4.32, while this base is Ubuntu 22.04 (glibc
# 2.35, GLIBCXX_3.4.30). Its `server-cuda` tag is also CUDA 12.8 against this
# image's CUDA 13.0.2, so its libggml-cuda.so wants libcudart.so.12 sonames we
# do not ship; carrying the CUDA 12.8 runtime to satisfy it would add ~2.4 GiB.
# Building in the base image itself makes glibc, libstdc++ and CUDA match by
# construction. Nothing about the source is modified -- spec §2 -- this is
# upstream's own CMake on an upstream tag.
ARG LLAMACPP_BUILD=b10731

# CUDA architectures llama.cpp is compiled for. The default is EVERY
# architecture this image's toolchain can target. Not asserted -- measured:
# `nvcc --list-gpu-arch` inside vllm/vllm-openai@${VW_BASE_DIGEST} (CUDA 13.0,
# V13.0.88) answers exactly
#   compute_75 80 86 87 88 89 90 100 103 110 120 121
# and that list is the hard ceiling. CUDA 13 dropped Maxwell, Pascal and
# Volta, so 75 (Turing) is the floor and there is nothing below it to add.
#
# This was "75-real;86-real" until 2026-09-06 -- exactly the two cards on the
# machine it was developed on (Quadro RTX 5000, RTX A4000). That is a fine
# choice for a LOCAL build and the wrong one for a PUBLISHED image: -real emits
# native SASS and nothing else, so on an Ada (RTX 40xx, L4, L40S), Hopper
# (H100) or Blackwell (RTX 50xx, B200) card the llama.cpp backend did not run
# slowly, it was ABSENT -- the product's second engine simply missing on
# hardware it should be good at. The published image has to work wherever the
# toolchain allows; narrowing is the local builder's call, so it is an ARG:
#
#   docker compose build --build-arg LLAMACPP_CUDA_ARCHS="86-real" api
#
# That is not a rounding error. Measured on 32 threads, this one step: 158 s
# for "86-real", 240 s for the old "75-real;86-real", 1069 s for the default
# below. The cost is linear -- about 77 s of fixed C++/link work plus ~81 s per
# architecture -- so anyone compiling for one known card should pass the arg and
# come out FASTER than the old two-card default. README.md and
# documents/INSTALL.md carry the full table and the 4-core figures.
#
# THE -virtual ENTRY IS NOT DECORATION. -real is native SASS for one
# architecture; -virtual embeds PTX, which the driver JIT-compiles for any
# newer one. It is the difference between "a card nobody here has heard of is
# slow for a few seconds on first load" and "a card nobody here has heard of
# does not work at all". Without it, the day NVIDIA ships compute capability
# 13, this backend is gone on it and the fix is a rebuild.
#
# It is 90-virtual and NOT 121-virtual -- the numerically highest -- and that
# was settled by running cmake, not by reading the number:
#   * ggml/src/ggml-cuda/CMakeLists.txt rewrites any plain 12X to 12Xa
#     (observed: "Replacing 121-virtual in CMAKE_CUDA_ARCHITECTURES with
#     121a-virtual") because Blackwell's FP4 tensor-core instructions are not
#     forwards compatible. An -a PTX blob JITs for that one architecture and
#     nothing else, so 121-virtual would buy zero future coverage while looking
#     like it bought all of it.
#   * compute capability stopped being a linear feature ladder at Blackwell --
#     sm_120 is consumer Blackwell and does not have sm_100's tcgen05 -- so PTX
#     emitted above 9.0 is not a safe universal JIT source either.
# 90 is the highest architecture-generic PTX the toolchain still offers, and is
# the same fallback upstream llama.cpp ships in its own default list. Nothing
# at or below 121 needs it -- all of that is covered by -real above -- so the
# PTX only ever has to serve hardware that does not exist yet, which is
# precisely the case where "generic" beats "newest".
#
# 87, 88 and 110 are Jetson/Thor SoC parts that cannot appear under this
# x86_64-only image, and they are kept anyway with the cost stated so the next
# person can drop them knowingly: about 4 minutes of the compile on 32 threads
# and ~50 MB. The rule "everything nvcc accepts" is one a base-image bump
# maintains for free; a hand-curated list of what we think exists is one that
# silently rots, which is how this ARG came to say "75;86" in the first place.
#
# 120 and 121 are written plain and become 120a/121a through the rewrite above
# -- deliberately; that is upstream's FP4 policy, and writing them plain keeps
# us on that policy if upstream changes it.
ARG LLAMACPP_CUDA_ARCHS="75-real;80-real;86-real;87-real;88-real;89-real;90-real;100-real;103-real;110-real;120-real;121-real;90-virtual"

# HOW WIDE the compile runs is a different question from how many architectures
# it emits, and widening the list above is what stopped `nproc` being a safe
# answer to it. Every .cu is now 13 device compilations instead of 2, so each
# -j lane holds a cicc/cudafe++/ptxas child roughly 6.5x longer: where the
# two-architecture build's lanes were briefly heavy and then idle, all of them
# are heavy essentially all of the time. Core count alone stopped describing
# the peak, so -j has to answer to memory as well.
#
# Not a hypothesis -- this is the kernel's own OOM report from the first CI run
# that ever attempted the full list (job 244537, 14% in). On a 12-core, 3.7 GiB
# shared runner `-j$(nproc)` put 12 nvcc lanes in flight whose children held
# 1921 MB resident plus 797 MB swapped, and the kernel killed buildkitd rather
# than any compiler, taking the whole build down with it. Largest single lane:
# one cicc at 514 MB.
#
# Hence one lane per 512 MiB of build memory, capped by nproc. 512 MiB is that
# measured worst lane rounded down, not a guess at a number small enough to be
# safe -- and being a per-lane BUDGET rather than a ceiling it is inert on any
# machine actually built for this work: the memory term does not bind until a
# host has less than 512 MiB per core, so the 16C/32T box the architecture list
# was measured on still gets all 32 lanes as long as it has 16 GiB. Only a host
# that genuinely cannot hold the build is slowed down, and slowed is the point
# -- it used to be killed.
#
# The budget follows a cgroup memory limit when there is one, so capping the
# builder (`docker buildx create --driver-opt memory=...`) narrows -j to match
# instead of fighting it. Override both terms if you know better:
#
#   docker compose build --build-arg LLAMACPP_BUILD_JOBS=8 api
ARG LLAMACPP_BUILD_JOBS=""

FROM vllm/vllm-openai@${VW_BASE_DIGEST} AS llamacpp-build
ARG LLAMACPP_BUILD
ARG LLAMACPP_CUDA_ARCHS
ARG LLAMACPP_BUILD_JOBS
# nvcc, the CUDA headers, libcublas-dev, build-essential and g++ are ALREADY in
# this image: vLLM's own Dockerfile re-adds the CUDA development toolchain to
# its vllm-base stage for runtime JIT (FlashInfer, DeepGEMM) and it survives
# into the published image. Only these are missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
# A full clone, not --depth 1: llama.cpp derives its build NUMBER from
# `git rev-list --count HEAD` (cmake/build-info.cmake), so a shallow clone bakes
# a wrong version into --version and into every log line.
RUN git clone https://github.com/ggml-org/llama.cpp /src && \
    git -C /src checkout ${LLAMACPP_BUILD}
# -DCMAKE_CUDA_ARCHITECTURES="${LLAMACPP_CUDA_ARCHS}": every architecture this
#   image's nvcc can target, plus one PTX entry for the ones it cannot yet.
#   The full reasoning, the measured cost and how to narrow it for a local
#   build are on the ARG declaration near the top of this file -- read that
#   before changing this line.
# -DCMAKE_INSTALL_RPATH='$ORIGIN:/opt/llamacpp' + BUILD_WITH_INSTALL_RPATH: the
#   binary finds its own .so files with no env var and from any working
#   directory. Upstream's published images instead rely on a build-tree RUNPATH
#   whose empty trailing element resolves against the CWD, which works only under
#   their WORKDIR /app; our engine is launched by a supervisor from elsewhere.
#
#   BOTH entries, and the second is not redundant. argv[0] is the bare name
#   `llama-server`, which PATH resolves to the SYMLINK /usr/local/bin/llama-server.
#   An exec of that symlink works on $ORIGIN alone -- the kernel sets
#   /proc/self/exe to the real path, so $ORIGIN is /opt/llamacpp -- but `ldd` and
#   `ld.so --list` expand $ORIGIN from the path they are GIVEN, so they report
#   libllama-server-impl.so as "not found" through the symlink while the binary
#   runs perfectly. Depending on that asymmetry is how a subtle loader bug hides;
#   the absolute entry costs nothing, is a path we own, and makes every tool
#   agree with reality.
# GGML_STATIC is deliberately NOT set: it adds a global `-static` link on UNIX,
#   which is incompatible with the dlopen'd backends, and libcublas_static.a
#   alone is 119 MB. Dynamic costs zero extra bytes here because this image
#   already ships libcudart.so.13, libcublas.so.13 and libcublasLt.so.13.
# -j is computed, not $(nproc) -- see the LLAMACPP_BUILD_JOBS block above for
#   why, and for where the 512 MiB per lane comes from. The echo is deliberate:
#   when this step is slow the first question is always "how parallel was it",
#   and the answer should be in the build log rather than inferred afterwards.
RUN set -e; \
    mem_mb="$(awk '/^MemTotal:/ {print int($2/1024)}' /proc/meminfo)"; \
    if [ -r /sys/fs/cgroup/memory.max ]; then \
      lim="$(cat /sys/fs/cgroup/memory.max)"; \
      if [ "$lim" != max ]; then \
        lim_mb=$((lim / 1048576)); \
        if [ "$lim_mb" -lt "$mem_mb" ]; then mem_mb="$lim_mb"; fi; \
      fi; \
    fi; \
    jobs="${LLAMACPP_BUILD_JOBS}"; \
    if [ -z "$jobs" ]; then \
      jobs=$((mem_mb / 512)); \
      if [ "$jobs" -gt "$(nproc)" ]; then jobs="$(nproc)"; fi; \
      if [ "$jobs" -lt 1 ]; then jobs=1; fi; \
    fi; \
    echo "llama.cpp: -j${jobs} (${mem_mb} MiB build memory, $(nproc) cores) for ${LLAMACPP_CUDA_ARCHS}"; \
    cmake -S /src -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${LLAMACPP_CUDA_ARCHS}" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN:/opt/llamacpp' \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON; \
    cmake --build /build -j"${jobs}" --target llama-server
# Collect exactly what llama-server needs into one flat directory. The other
# *-impl.so files are for llama-cli, llama-bench and friends, which this product
# never launches. The ldd runs from / on purpose -- that is the condition a
# CWD-relative RUNPATH would fail under, and the supervisor's cwd is not here.
RUN mkdir -p /opt/llamacpp && \
    cp /build/bin/llama-server /opt/llamacpp/ && \
    cp -a /build/bin/*.so* /opt/llamacpp/ && \
    rm -f /opt/llamacpp/libllama-cli-impl.so* \
          /opt/llamacpp/libllama-bench-impl.so* && \
    cd / && ldd /opt/llamacpp/llama-server > /opt/llamacpp/.ldd.txt && \
    ! grep -q 'not found' /opt/llamacpp/.ldd.txt

FROM vllm/vllm-openai@${VW_BASE_DIGEST}
ARG LLAMACPP_BUILD

WORKDIR /app

# GGUF support moved out of vLLM core into the OOT vllm-gguf-plugin as of the
# 0.25.x line (core vllm/model_executor/model_loader/gguf_loader.py no longer
# exists). Warden supports GGUF model rows, so the plugin is required. Pinned
# for reproducibility; bump together with the base image.
# NOTE: the plugin's adapter-based loader has no _get_gguf_weights_map rename
# table, so the qwen3_5/qwen3_5_moe/vision_num_layers patches we carried on the
# 0.20.0 base (#107/#108/#115, upstream vllm PR #38140 — still OPEN) are gone
# with it. Qwen3.5/3.6 GGUF loadability under the plugin is UNVERIFIED — if a
# Qwen3.x GGUF row regresses, that is the first place to look.
# --mount=type=cache instead of --no-cache-dir: the wheel cache lives in a
# BuildKit cache mount, so it is NOT written into the layer (same size result
# --no-cache-dir gave) but IS reused when this step has to re-run — a base
# digest bump, or a plugin version bump. sharing=locked because pip's cache
# is not concurrency-safe and the two buildx builds in publish:images can
# overlap on the same runner.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install vllm-gguf-plugin==0.0.4

# vLLM 0.25.1's ModelConfig init calls override_quantization_method(quant_cfg,
# user_quant, hf_config=self.hf_config) for EVERY registered quantization method
# (vllm/config/model.py:1064 _verify_quantization). Core's base signature grew a
# third param `hf_config` (base_config.py) but the plugin's GGUFConfig classmethod
# is stuck on the old 2-arg signature, so the hf_config kwarg is a TypeError —
# raised during init for ALL models (even non-GGUF FP8 rows), breaking 100% of
# loads with rc=1. Plugin 0.0.4 is the latest on PyPI (no fixed release), so we
# patch the signature at build time to accept and ignore the extra kwarg. The
# body already discards hf_quant_cfg, so ignoring hf_config too is safe. The
# `s2 != s` guard fails the build loudly if the upstream line ever changes.
#
# The write and the verification run in SEPARATE python processes on purpose:
# resolving the module path with find_spec imports the plugin package, which
# imports this config module as a side effect at the OLD source, caching its
# compiled code object in sys.modules. An in-process re-import would then return
# the stale pre-patch object and the signature assert would spuriously fail even
# though the file on disk is patched correctly. A fresh interpreter compiles the
# patched source and sees the new signature — exactly what the runtime
# vllm-serve subprocess does. (Deleting *.pyc handles the on-disk bytecode cache;
# the separate process handles the in-memory one.)
RUN python3 - <<'PY'
import importlib.util as u
p = u.find_spec("vllm_gguf_plugin.quantization.config").origin
s = open(p).read()
s2 = s.replace(
    "cls, hf_quant_cfg: dict[str, Any], user_quant: str | None",
    "cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None",
)
assert s2 != s, "override_quantization_method signature not found to patch"
open(p, "w").write(s2)
print("gguf-plugin override_quantization_method source patched at", p)
PY
RUN find /usr/local/lib/python3.12/dist-packages/vllm_gguf_plugin -name '*.pyc' -delete
RUN python3 -c "import inspect; from vllm_gguf_plugin.quantization.config import GGUFConfig; sig = inspect.signature(GGUFConfig.override_quantization_method); assert 'hf_config' in sig.parameters, f'patch did not take: {sig}'; print('gguf-plugin override_quantization_method patched:', sig)"

# Strip the nixl_ep packages (NVIDIA cross-node expert-parallel all2all) that
# the v0.25.1 base image bundles. Their compiled CUDA extension dlopens an
# AVX-built UCX library (libucs/libucp) at *import* time; on a CPU without AVX
# the UCX load-time feature check aborts the process:
#   "FATAL: UCX library was compiled with avx but CPU does not support it."
# That is a C-level SIGSEGV, not a catchable Python exception. A production GPU
# node runs a QEMU vCPU with no AVX, so *every* model load segfaulted (rc=-11)
# on 0.25.1 until this. vLLM only guards the crashing import with
# has_nixl_ep() (a bare importlib.find_spec presence check in
# fused_moe/all2all_utils.py), so deleting the packages flips that guard to
# False and the AVX-UCX path is never touched. nixl_ep is cross-node
# expert-parallel — irrelevant to our single-node TP setups. Base `nixl`
# (no import-time UCX dlopen) is intentionally left in place.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/nixl_ep* && \
    python3 -c "import importlib.util as u; assert u.find_spec('nixl_ep') is None, 'nixl_ep still importable after removal'"

# Bump NCCL 2.28.9 -> 2.30.4 (the base image ships 2.28.9).
#
# NCCL 2.28.9 deadlocks on CUDA-graph replay containing NCCL collectives:
# the collective never returns, the TP worker stops answering, and vLLM's
# watchdog eventually fires `TimeoutError: RPC call to sample_tokens timed
# out` -> EngineDeadError. Workers stay ALIVE with no traceback, no CUDA
# error and oom_kill=0, so every memory theory reads clean. Upstream
# vllm#52504 verified deadlock on 2.28.9 and clean on 2.30.4 with nothing
# else changed; `--enforce-eager` also avoids it, which is what isolated the
# hang to captured-graph collectives rather than the collectives themselves.
#
# A production deployment ran exactly 2.28.9 with cudagraph_mode=FULL_AND_PIECEWISE and TP=4,
# and died with that signature three times (2026-08-02, and twice on
# 2026-08-18). Note the `sample_tokens` in the message is a red herring — the
# sampler is innocent; the hang is upstream of it in the collective.
#
# torch 2.11.0+cu130 hard-pins nvidia-nccl-cu13==2.28.9, so pip prints an
# incompatibility warning. It is metadata-only: NCCL keeps ABI compatibility
# within major 2, and torch dlopens libnccl.so.2 from THIS package directory
# (confirmed via /proc/self/maps) rather than the apt-installed
# /usr/lib/x86_64-linux-gnu copy, which is left untouched.
#
# The assertion checks the wheel version AND that torch still loads the lib
# from this package -- do NOT "verify" with torch.cuda.nccl.version(), which
# reports the version torch was COMPILED against (still 2.28.9) and happily
# reports success on a machine where nothing was upgraded.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install "nvidia-nccl-cu13==2.30.4" && \
    python3 -c "import importlib.metadata as m, torch; v = m.version('nvidia-nccl-cu13'); assert v == '2.30.4', 'nccl wheel is ' + v; libs = [l.split()[-1] for l in open('/proc/self/maps') if 'libnccl.so' in l]; assert libs, 'torch mapped no libnccl'; assert 'dist-packages/nvidia/nccl' in libs[0], 'torch loads libnccl from ' + libs[0] + ' -- the upgraded wheel is NOT the runtime library'; print('nccl runtime lib:', libs[0], v)"

COPY requirements.txt /app/
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt

# llama.cpp, from the build stage above. On PATH because the backend's argv[0]
# is the bare name `llama-server` (app/runtime/backends/llamacpp/__init__.py),
# and the subprocess env's PATH is /usr/local/bin:/usr/bin:/bin.
COPY --from=llamacpp-build /opt/llamacpp /opt/llamacpp
# The `cd /` is not decoration: it runs the binary from a directory that is NOT
# its own, which is exactly the condition that exposes a broken rpath. A wrong
# rpath fails the build here instead of failing one operator's load six weeks
# later, from some working directories and not others.
RUN ln -s /opt/llamacpp/llama-server /usr/local/bin/llama-server && \
    cd / && llama-server --version && \
    ldd /usr/local/bin/llama-server | tee /tmp/ldd.txt && \
    ! grep -q 'not found' /tmp/ldd.txt && rm -f /tmp/ldd.txt
ENV VW_LLAMACPP_BUILD=${LLAMACPP_BUILD}

# Application source LAST, because it is the ONLY input that changes on a
# typical commit. Everything above -- the base, the gguf-plugin patch, the
# nixl_ep strip, the NCCL bump, requirements.txt and the whole llama.cpp
# compile -- is keyed on pinned versions, so with a warm BuildKit cache a
# source-only change re-runs nothing and re-pushes exactly this one layer.
#
# It used to sit BEFORE the `COPY --from=llamacpp-build` and the rpath/ldd
# verification that now precede it, which made every app edit invalidate and
# re-run both. Moving it down is filesystem-identical -- nothing above reads
# /app/app, and nothing below writes into it -- it only changes which layers a
# source edit has to rebuild.
COPY app /app/app

# Build-time identity (spec 2026-05-13 §Version surfacing, P1-9). The CI
# build job passes these via --build-arg from $CI_COMMIT_TAG /
# $CI_COMMIT_SHORT_SHA; local builds may pass them too. When unset, GET
# /api/version falls back to "dev" / "unknown" (see app/system/routes_version.py).
ARG VW_BUILD_VERSION=dev
ARG VW_BUILD_SHA=unknown
ENV VW_BUILD_VERSION=${VW_BUILD_VERSION} \
    VW_BUILD_SHA=${VW_BUILD_SHA}

VOLUME ["/data"]

EXPOSE 8080

# Override vllm/vllm-openai's ENTRYPOINT — we run our own FastAPI app
ENTRYPOINT []
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
