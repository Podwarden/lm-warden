# LM Warden -- operator targets.
#
# This file is the day-to-day surface for running an installed stack. It is
# self-contained on purpose: install.sh copies it verbatim as the `Makefile`
# of an install directory (e.g. /opt/vllm-warden), and the repository root
# Makefile `include`s it so the same targets work in a source checkout. One
# file, shipped in git, never generated -- so what an operator runs is what
# CI tested.
#
# Every target is a thin wrapper over `docker compose`; the stack contract is
# docker-compose.yml, the runtime choices (images, GPUs, port, secrets) are
# in .env and docker-compose.override.yml, both written by install.sh.
#
# Targets read no environment beyond what compose reads itself. The recipes
# are POSIX sh: `make` runs them under /bin/sh, which is dash on Debian and
# Ubuntu, so no bashisms (`read -p`, `[[`, `$'...'`) here.

.PHONY: start stop restart logs status pull config preflight uninstall \
        save-images load-images export-hf-cache import-hf-cache help

COMPOSE ?= docker compose

# Names the operator will paste into a browser; kept in one place.
_vw_port = $$(grep -E '^WARDEN_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | cut -d'#' -f1 | tr -d '[:space:]')

# --no-build on every `up`: these targets run the release images named in
# docker-compose.override.yml, never the from-source `build:` in
# docker-compose.yml. A missing image is pulled (pull_policy: missing), and
# if it cannot be pulled the error says so instead of starting a CUDA build.
start: ## Start the stack (detached); creates containers on first run
	@test -f .env || { echo "no .env here -- run ./install.sh first (or: cp .env.example .env)" >&2; exit 1; }
	$(COMPOSE) up -d --no-build --remove-orphans
	@p=$(_vw_port); p=$${p:-8080}; echo ""; echo "  UI:   http://localhost:$$p/ui/"; echo "  API:  http://localhost:$$p/v1/"; echo ""

stop: ## Stop and remove the containers (volumes and data are kept)
	$(COMPOSE) down --remove-orphans

restart: ## stop + start, re-reading .env and the override file
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) up -d --no-build --remove-orphans

logs: ## Follow the logs of every service (make logs S=api for one)
	$(COMPOSE) logs -f --tail=200 $(S)

status: ## Show container state and health
	$(COMPOSE) ps

pull: ## Pull the release named by VERSION in .env and restart on it
	$(COMPOSE) pull
	$(COMPOSE) up -d --no-build --remove-orphans

config: ## Print the fully merged compose config (what will actually run)
	$(COMPOSE) config

preflight: ## Re-run the installer's host checks without writing anything
	@if [ -x ./install.sh ]; then ./install.sh --check; else sh ./install.sh --check; fi

uninstall: ## Stop the stack and DELETE its volumes (models, database) -- asks first
	@echo "This stops LM Warden and deletes its data volumes (database, HF model cache)."
	@printf 'Type yes to continue: '; read ans </dev/tty; [ "$$ans" = "yes" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v --remove-orphans
	@echo "Volumes removed. Delete this directory to finish: rm -rf $$(pwd)"

# ---- air-gapped transport --------------------------------------------------
# See README "Offline / air-gapped install". save-images runs on a connected
# machine; load-images and import-hf-cache on the isolated one. All three
# address the same images and volume names the stack uses, so nothing has to
# be typed twice.

IMAGES_FILE ?= lm-warden-images.tar
CACHE_FILE  ?= hf-cache.tar

# The HF cache volume is `<project>_vw-hfcache`; the project name is the
# first line of the merged config, so this stays right if COMPOSE_PROJECT_NAME
# is ever changed in .env.
_hf_volume = $$($(COMPOSE) config 2>/dev/null | sed -n 's/^name: //p' | head -1)_vw-hfcache

save-images: ## docker save the images of the release in .env to IMAGES_FILE (default lm-warden-images.tar)
	@imgs=$$($(COMPOSE) config --images); echo "saving: $$imgs"; docker save -o "$(IMAGES_FILE)" $$imgs && ls -lh "$(IMAGES_FILE)"

load-images: ## docker load the images saved by save-images from IMAGES_FILE
	docker load -i "$(IMAGES_FILE)"

export-hf-cache: ## tar the HuggingFace model cache volume to CACHE_FILE (default hf-cache.tar)
	@v=$(_hf_volume); echo "exporting volume $$v -> $(CACHE_FILE)"; \
	  docker run --rm -v "$$v":/cache:ro -v "$$(pwd)":/out alpine:3 tar -C /cache -cf "/out/$(CACHE_FILE)" . && ls -lh "$(CACHE_FILE)"

import-hf-cache: ## untar CACHE_FILE (default hf-cache.tar) into the HuggingFace cache volume (created if absent)
	@v=$(_hf_volume); echo "importing $(CACHE_FILE) -> volume $$v"; \
	  docker volume create "$$v" >/dev/null && \
	  docker run --rm -v "$$v":/cache -v "$$(pwd)":/in:ro alpine:3 tar -C /cache -xf "/in/$(CACHE_FILE)" && \
	  docker run --rm -v "$$v":/cache:ro alpine:3 du -sh /cache

help: ## List these targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-16s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
