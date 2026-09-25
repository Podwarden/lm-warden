IMAGE := lm-warden:dev
# The Python base image is no longer named here: scripts/ci-deps-image.sh owns
# it (PYTHON_IMAGE, default python:3.11-slim) so the Makefile and .gitlab-ci.yml
# cannot drift apart on it.

# Run a command against the project's Python dependencies.
#
# scripts/ci-deps-image.sh — the same resolver .gitlab-ci.yml uses, so local
# and CI agree by construction — picks the prebuilt dependency image for the
# current requirements.txt/requirements-dev.txt pair if it is present, and
# otherwise falls back to python:3.11-slim + the same `pip install` these
# targets used to run unconditionally. It sets:
#
#   VW_RUN_IMAGE  the image to run
#   VW_PIP        `:` when the deps are baked in, the pip install when not
#
# WHY IT MATTERS LOCALLY: every `make test-unit` / `make lint` / `make
# typecheck` used to reinstall the whole pinned set from PyPI, and failed
# outright whenever PyPI was slow (twice on 2026-09-03). With the image present
# these targets do no network I/O at all.
#
# The resolver does NOT reach for the registry on a workstation unless asked,
# so a machine with no route to registry.podwarden.com falls back instantly
# instead of blocking on a doomed pull. Fetch (or build) the image once with:
#
#   make deps-image
#
# Recipes concatenate onto the trailing open quote, e.g.
#   $(RUN_DEPS)"pytest -v"
RUN_DEPS = set -e; . ./scripts/ci-deps-image.sh; \
	docker run --rm -e HOME=/tmp -e VW_PIP \
	  -v "$(PWD)":/app -w /app "$$VW_RUN_IMAGE" \
	  sh -c 'eval "$$VW_PIP" && export PATH=/tmp/.local/bin:$$PATH && '

.PHONY: install test test-unit test-integration lint typecheck typecheck-baseline format deps-image docker-build docker-run shell generate-api-types sync-shared-docs check-shared-docs smoke conformance

# Pull the prebuilt dependency image for the current requirements pair, build
# it locally if the registry does not have it, and push what we built so the
# next machine pulls instead of rebuilding.
#
# Pushing from a workstation is safe because this tag is content-addressed --
# the requirements digest plus the architecture -- so a push is idempotent and
# can only ever add the image someone else would have built identically. It is
# a shared cache, not a release. The push is best-effort: a developer without
# registry write access still gets their local image, and sees a warning
# rather than a failed target.
#
# The arm64 tag exists only because a workstation pushed it. CI runs on amd64
# runners and deliberately does not build arm64 -- that would mean emulating
# the thing this whole arrangement avoids.
deps-image:
	@VW_DEPS_PULL=1 VW_DEPS_BUILD=1 VW_DEPS_PUSH=1 sh -c '. ./scripts/ci-deps-image.sh'

# `install` used to `pip install` into a `--rm` container that was deleted a
# second later, so it warmed nothing and left nothing behind. Now it means what
# the name says: make the dependencies available to the other targets, once.
install: deps-image

test:
	@$(RUN_DEPS)"pytest -v $(ARGS)"

test-unit:
	# -n auto here, a fixed count in CI. Locally this is the only job on the
	# machine, so one worker per CPU is right; a CI runner is shared and `auto`
	# starves whatever is co-scheduled (see the note in .gitlab-ci.yml).
	# Override with VW_XDIST_WORKERS to match a CI run.
	@$(RUN_DEPS)"pytest -v -n $${VW_XDIST_WORKERS:-auto} -m 'not integration' tests/unit"

test-integration:
	@$(RUN_DEPS)"pytest -v -m integration $(ARGS) tests/integration"

lint:
	# Uses the pinned ruff from requirements-dev.txt rather than `pip install ruff`
	# (which always pulls the latest version). Without this pin, new ruff
	# releases routinely fire lints CI doesn't have (CI installs the pinned
	# requirements), causing local-vs-CI false-fail drift. The prebuilt image is
	# built from those same pinned files, so the pin survives the switch.
	@$(RUN_DEPS)"ruff check app/ tests/"

format:
	# Same rationale as lint above — pin to requirements-dev.txt so formatting
	# changes never depend on which ruff version the operator happened to have
	# locally.
	@$(RUN_DEPS)"ruff format app/ tests/"

# `mypy app/` gated on mypy-baseline.txt (#243): fails on any error the
# baseline does not list AND on any listed error mypy no longer reports, so
# the baseline can only shrink. CI runs the same script (typecheck:mypy).
# Raw `mypy app/` (via `make shell`) still prints the full list with line
# numbers. See scripts/mypy-baseline.py for the rationale and the key format.
typecheck:
	@$(RUN_DEPS)"python scripts/mypy-baseline.py"

# Regenerate mypy-baseline.txt from the current tree and commit the diff.
# Deletions are the expected direction; additions mean an error was accepted
# rather than fixed and should be called out in the MR.
typecheck-baseline:
	@$(RUN_DEPS)"python scripts/mypy-baseline.py --write"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	# --pid=host: required so /api/system/gpus can attribute nvidia-smi
	# compute holders (host PIDs) to supervisor-tracked PIDs. See #42.
	docker run --rm --gpus all --pid=host -p 8080:8080 \
	  -e VW_COOKIE_SECRET=$$(openssl rand -base64 32) \
	  -e VW_CONTAINER_GPU_COUNT=4 \
	  -v $(PWD)/.data:/data \
	  $(IMAGE)

shell:
	@set -e; . ./scripts/ci-deps-image.sh; \
	  docker run --rm -it -e HOME=/tmp -e VW_PIP \
	    -v "$(PWD)":/app -w /app "$$VW_RUN_IMAGE" \
	    sh -c 'eval "$$VW_PIP" && export PATH=/tmp/.local/bin:$$PATH && exec bash'

# node:20-slim, not node:20-alpine — same change as the CI frontend jobs, so a
# locally regenerated file comes out of the same toolchain typecheck:api-types
# diffs it against. See the npm-ci note in .gitlab-ci.yml.
generate-api-types:
	@$(RUN_DEPS)"python -c \"import json, sys; from app.main import app; sys.stdout.write(json.dumps(app.openapi()))\"" > openapi.json
	docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp -v $(PWD):/work -w /work/frontend node:20-slim \
	  npx -y openapi-typescript@7 ../openapi.json -o src/lib/api-types.generated.ts

# The hazard sections shared between the public docs (documents/HAZARDS.md,
# documents/INSTALL.md, documents/API.md -- all ship to GitHub) and the Hub
# catalogue listing are single-sourced from docs/shared/. Same contract as
# generate-api-types above: the marked regions are generated, hand edits are
# reverted, and CI diffs them (lint:shared-docs). Stdlib only, so it runs in a
# bare python image rather than the deps image.
sync-shared-docs:
	docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp \
	  -v $(PWD):/app -w /app python:3.11-slim python scripts/sync-shared-docs.py

check-shared-docs:
	docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp \
	  -v $(PWD):/app:ro -w /app python:3.11-slim \
	  python scripts/sync-shared-docs.py --check

# Front-door base URL for `make smoke`. Overridable so the installer CI job
# can point it at the port it published (SMOKE_URL=http://127.0.0.1:NNNN).
SMOKE_URL ?= http://localhost:8080

smoke:
	# #155 unified-port: end-to-end smoke against the live Caddy front-door
	# on :8080. Assumes `docker compose up -d` has already brought the api,
	# ui, and caddy services to healthy state. Each curl asserts a STATUS
	# code, not a body, because the bodies differ across builds (build SHA
	# in /api/csrf, Next chunk hashes in /ui/) and would make this brittle.
	#
	# `-L` follows redirects — `/ui/` 308→`/ui` 307→`/ui/models` 200 is
	# the natural Next.js basePath landing flow; we care that the chain
	# terminates in a 200, not the intermediate hops.
	#
	# Expected (final-status, after redirect chain):
	#   /            → 200 (landing page HTML, public)
	#   /_landing    → 200 (same content, direct)
	#   /ui/         → 200 (Next.js root page, possibly /ui/models or /ui/login)
	#   /api/csrf    → 200 (CSRF bootstrap, no auth required)
	#   /healthz     → 200 (uptime probe — Caddy → Next /healthz alias)
	#
	# `|| rc=$$?` is load-bearing, not defensive clutter. Under `set -e` a bare
	# `code=$$(curl ...)` assignment aborts the whole recipe the instant curl
	# exits non-zero, so neither the printf below nor the FAIL branch ever runs
	# and make reports a naked `*** [smoke] Error 7`. That says nothing about
	# WHICH url failed or why — which is the one thing a smoke test exists to
	# tell you. Trapping the status keeps the shell alive long enough to say it.
	#
	# The two failure kinds are genuinely different and must not be merged:
	# a non-zero curl exit means the request never completed (wrong port, stack
	# down, hung app), while a completed request carrying a non-200 means the
	# stack is up and a route is broken. They send you to different places.
	@set -e; \
	for path in / /_landing /ui/ /api/csrf /healthz; do \
	  rc=0; \
	  code=$$(curl -sL -o /dev/null -w "%{http_code}" --max-time 15 "$(SMOKE_URL)$$path") || rc=$$?; \
	  if [ "$$rc" -ne 0 ]; then \
	    case "$$rc" in \
	      6)  why="could not resolve the host";; \
	      7)  why="could not connect - nothing is listening there";; \
	      28) why="timed out after 15s - the port answered but the app did not";; \
	      52) why="empty reply from the server";; \
	      56) why="the connection broke mid-response";; \
	      *)  why="the request did not complete";; \
	    esac; \
	    echo "FAIL: GET $(SMOKE_URL)$$path" >&2; \
	    echo "      $$why (curl exit $$rc)" >&2; \
	    echo "" >&2; \
	    echo "      The stack has to be up before smoke can test it:" >&2; \
	    echo "        docker compose ps    # api, ui and caddy should all be healthy" >&2; \
	    echo "      If it is published somewhere other than $(SMOKE_URL):" >&2; \
	    echo "        make smoke SMOKE_URL=http://127.0.0.1:PORT" >&2; \
	    exit 1; \
	  fi; \
	  printf "GET %-15s -> %s\n" "$$path" "$$code"; \
	  if [ "$$code" != "200" ]; then \
	    echo "FAIL: $$path returned $$code (expected 200)" >&2; \
	    exit 1; \
	  fi; \
	done; \
	echo "smoke OK"

# Run the @podwarden/chat-ui conformance kit against THIS backend.
#
# `tests/conformance/serve.py` boots the real app on 127.0.0.1:18080 with
# only `httpx.AsyncClient.send` faked, then prints
#   READY <port> <bearer> <csrf_id> <csrf_token>
# once /healthz answers and the admin user, the loaded-model row and the
# default model are seeded. Everything after that is the package's own
# vitest suite driving real HTTP.
#
# Unlike the other targets this one uses the local .venv rather than
# docker: the kit needs the backend and the node runner to share a
# loopback interface, and a venv is what the repo's own test loop uses.
#
# Field-splitting the READY line with `cut` rather than bash's `read _ a b
# < <(...)` keeps this identical to the CI job, whose `sh -c` has no
# process substitution.
conformance:
	@set -e; \
	log=/tmp/vw-conformance.log; : > $$log; \
	.venv/bin/python -m tests.conformance.serve > $$log 2>&1 & \
	pid=$$!; \
	trap 'kill $$pid 2>/dev/null || true' EXIT INT TERM; \
	for i in $$(seq 1 150); do \
	  grep -q '^READY' $$log && break; \
	  kill -0 $$pid 2>/dev/null || break; \
	  sleep 0.2; \
	done; \
	if ! grep -q '^READY' $$log; then \
	  echo "conformance harness never printed READY (30s); log follows:" >&2; \
	  cat $$log >&2; exit 1; \
	fi; \
	line=$$(grep '^READY' $$log); \
	PORT=$$(echo "$$line" | cut -d' ' -f2); \
	TOKEN=$$(echo "$$line" | cut -d' ' -f3); \
	CSRF_ID=$$(echo "$$line" | cut -d' ' -f4); \
	CSRF=$$(echo "$$line" | cut -d' ' -f5); \
	cd frontend && \
	CHAT2_BASE_URL=http://127.0.0.1:$$PORT/api/chat2 \
	CHAT2_TOKEN=$$TOKEN CHAT2_CSRF_ID=$$CSRF_ID CHAT2_CSRF=$$CSRF \
	npm run test:conformance

# ---------------------------------------------------------------------------
# Operator targets -- start / stop / restart / logs / status / pull / uninstall
# and the air-gapped transport helpers. They live in make/operator.mk, which
# install.sh ships verbatim as the Makefile of an install directory; including
# it here means a source checkout and an install run the identical targets.
# `make` with no goal now prints that file's help instead of pulling the CI
# dependency image (the previous, accidental default).
# ---------------------------------------------------------------------------
include make/operator.mk
