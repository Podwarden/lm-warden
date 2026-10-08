# Contributing to LM Warden

Issues and pull requests are welcome.
LM Warden, formerly LLM Warden and vLLM Warden, is developed in the open at
[github.com/Podwarden/lm-warden](https://github.com/Podwarden/lm-warden); the
product site is [lmwarden.com](https://lmwarden.com). Everyone taking part is
expected to follow the [Code of Conduct](../CODE_OF_CONDUCT.md).

## How changes get in

- **Bugs and ideas** — open an
  [issue](https://github.com/Podwarden/lm-warden/issues/new/choose). A bug
  report is most useful with the version (menu footer, or `GET /api/version`),
  the GPU and driver, the engine and model, and the engine log.
- **Security problems** — never in a public issue; see
  [SECURITY.md](../SECURITY.md).
- **Pull requests** — fork, branch from `main`, and open a PR against `main`.
  Keep one change per PR, add or update tests for it, and add a line under
  `## [Unreleased]` in `CHANGELOG.md` if a user would notice the change.
- **How a PR lands** — `main` on GitHub advances one squashed commit per
  release. A PR that is accepted is applied by a maintainer to the next
  release rather than merged with the merge button, so it reaches `main` as
  part of that release commit and is credited in `CHANGELOG.md`. The PR is
  closed with a note naming the release that carries it.

## Checks to run

Please run `make lint`, `make typecheck` and `make test` before opening a PR;
all three run in containers, so a working Docker install is the only
prerequisite. For a UI change, also run `npm ci && npm test && npm run
typecheck` in `frontend/` (Node 20+; no token needed — every dependency is on
public npm). `make typecheck` compares
`mypy --strict` against the committed `mypy-baseline.txt`, so it is green on a
clean checkout and red only for errors you introduced (or baseline entries you
fixed — run `make typecheck-baseline` and commit the shrunken file).

## Adding a backend

Adding a third backend is a documented seam rather than a rewrite: a directory
under `app/runtime/backends/` with an argv builder, an env allowlist, a log
diagnoser and a declared capability set, plus one line in the registry. The
supervisor, the drivers, the health probe, the warmup probe and the proxy are
already engine-agnostic. The one rule that will not bend is the product
invariant: a launch is argv plus environment, because we ship mainline runtimes
and do not patch them.

---

## Build from source

Both images build from this repository with nothing but Docker. There is no
token, no account, and no private registry anywhere in the build: every input is
public npm, public PyPI, Docker Hub, or a pinned commit on GitHub.

The operator files `./install.sh` writes double as the dev loop.
`docker-compose.yml` carries `build:` and the generated override carries
`image:`, so `docker compose build` tags what it builds with the release image
name: `make restart` then runs your build, and `make pull` puts the published
image back.

```bash
git clone https://github.com/Podwarden/lm-warden.git
cd lm-warden

# .env + docker-compose.override.yml. --no-pull/--no-start keep this step
# registry-free: nothing is downloaded, nothing is started.
./install.sh --gpus all --no-pull --no-start -y

docker compose build            # see the timing note below
make restart
make smoke
```

**How long the build takes depends on your core count, not your connection —
and by default it is long.** Measured cold on a 16-core/32-thread host
(Ryzen 9 5950X): **about 19 minutes**, of which **1069 s — 94% — is compiling
llama.cpp** at `-j32` for the thirteen CUDA architectures the default targets.
On 4-8 cores that one step is **1-3 hours**. Add roughly 9 GB of download if
the CUDA base image is not already in your local Docker cache, and 1-3 minutes
if the pip and npm cache mounts are cold. The base image is a one-time cost;
the compile is paid on every `--no-cache` build, and BuildKit reuses it on
every build after that.

**If you are building for your own cards, say so and the build gets shorter
than it ever was.** The architecture list is a build argument:

```bash
# one card, one architecture — sm_86 is Ampere (RTX 30xx, A4000, A100 is 80)
docker compose build --build-arg LLAMACPP_CUDA_ARCHS="86-real" api
```

Measured on the same 32-thread host, the llama.cpp step alone:

| `LLAMACPP_CUDA_ARCHS` | compile | `/opt/llamacpp` in the image |
| --- | --- | --- |
| `86-real` (one card) | 158 s | 68 MB |
| `75-real;86-real` (what this repo shipped before) | 240 s | 85 MB |
| the default: all 12 + `90-virtual` | 1069 s | 285 MB |

The cost is linear — roughly 77 s of fixed work plus 81 s per architecture at
`-j32` — so scale it by your own core count, but not past your RAM: the build
allows one parallel compilation per 512 MiB and caps `-j` there, so a 4 GiB
machine compiles 7-wide however many cores it has. Override with
`--build-arg LLAMACPP_BUILD_JOBS=N`. Your card's number is its compute
capability without the dot: `nvidia-smi --query-gpu=compute_cap --format=csv`.
Use `-real` for a card you own; add `-virtual` only if you want PTX.

The wide default exists because the *published* image has to run on hardware we
do not have. If you are compiling anyway, you know your hardware, and one
architecture is the right answer.

Or one image at a time:

```bash
docker compose build api        # engine image: CUDA base + llama.cpp compile
docker compose build ui         # UI image: chat-ui from source + Next.js build
```

### The `api` image

It starts from the digest-pinned `vllm/vllm-openai` base
(currently v0.26.0), installs `requirements.txt` and `vllm-gguf-plugin` from
PyPI, and compiles llama.cpp from `github.com/ggml-org/llama.cpp` at the tag
pinned in `ARG LLAMACPP_BUILD` (currently `b10731`).

It compiles llama.cpp rather than copying upstream's published server image for
a concrete reason: that image is built on Ubuntu 24.04 against CUDA 12.8, while
this base is Ubuntu 22.04 with CUDA 13 — its `libggml-cuda.so` wants sonames
this image does not ship, and carrying the CUDA 12.8 runtime to satisfy it would
add ~2.4 GiB. Building from an upstream tag *inside* the base image makes glibc,
libstdc++ and CUDA match by construction. Nothing about the source is modified,
and the whole addition is ~285 MB of files at the default architecture list
(85 MB if you narrow it to one card) — well under 1% of the image.

CUDA architectures come from `ARG LLAMACPP_CUDA_ARCHS`, whose default is every
architecture this base image's `nvcc --list-gpu-arch` accepts —
`75 80 86 87 88 89 90 100 103 110 120 121`, all as `-real` native SASS — plus
`90-virtual` for PTX. The PTX entry is the one that matters for hardware that
does not exist yet: `-real` alone means an unrecognised card has no kernels at
all, while PTX means the driver JIT-compiles once and the card works. It is
`compute_90` rather than the numerically highest because llama.cpp's own CMake
rewrites any `12X` to `12Xa`, and `-a` PTX is locked to that one architecture —
`90` is the highest architecture-generic PTX the toolchain still offers, and is
the same fallback upstream llama.cpp ships. `Dockerfile` explains all of this at
the ARG. vLLM is unaffected either way: it comes from upstream's image with
upstream's architecture support.

The api image needs an x86_64 host. No GPU is needed to *build* it — only to run
it. Expect a large download and 40+ GB of free disk. Every patch it applies is
commented in `Dockerfile`, with the failure each one exists to prevent; read it
before bumping the base digest.

### The `ui` image

It compiles
[`@podwarden/chat-ui`](https://github.com/Podwarden/chat-ui) from source in its
own build stage — cloned from GitHub at the exact commit pinned in
`frontend/Dockerfile`:

```dockerfile
ARG CHATUI_REPO=https://github.com/Podwarden/chat-ui.git
ARG CHATUI_REF=<40-char commit SHA>
ARG CHATUI_VERSION=<the version that commit publishes>
```

That stage runs chat-ui's own `npm ci && npm run build && npm pack`, and the
packed result replaces the copy `npm ci` installed for this project. The lockfile
still pins chat-ui's ~20 runtime dependencies with integrity hashes — that
closure is what `package-lock.json` is for — while the component's own code is
the one compiled here. The build asserts the pinned version against both the
lockfile and the freshly built tarball, so a `CHATUI_REF`/`CHATUI_VERSION`
mismatch fails loudly instead of shipping a component the dependency graph was
not resolved for.

The pin is a SHA rather than a tag because the public chat-ui mirror carries no
tags. To build against a fork or an unreleased chat-ui, override the args — no
repo edit needed:

```bash
docker build -t lm-warden-ui \
  --build-arg CHATUI_REPO=https://github.com/you/chat-ui.git \
  --build-arg CHATUI_REF=<sha> --build-arg CHATUI_VERSION=<version> \
  frontend/
```

`CHATUI_VERSION` must still match what `frontend/package.json` depends on,
because the lockfile supplies that release's dependency closure.

`@podwarden/chat-ui` is also on public npm, published from that same GitHub
repository by OIDC trusted publishing with a provenance attestation, so a plain
`npm ci` in `frontend/` (for `npm test`, `npm run typecheck`, or `next dev`)
works without any of the above and without a token. The Docker build compiles
from source anyway, so that what ships in the image is something you built
rather than a tarball you downloaded.

Every dev target runs in Docker — no host Python or Node required:

| Command | What it does |
|---|---|
| `make test` | full pytest suite in `python:3.11-slim` |
| `make test-unit` / `make test-integration` | one suite only |
| `make lint` / `make format` | `ruff check` / `ruff format` |
| `make typecheck` | `mypy app/` gated on `mypy-baseline.txt` — fails on new errors and on stale baseline entries |
| `make typecheck-baseline` | regenerate `mypy-baseline.txt` after fixing (or deliberately accepting) mypy errors |
| `make docker-build` | build the api image as `lm-warden:dev` |
| `make generate-api-types` | regenerate frontend types from the FastAPI OpenAPI schema |
