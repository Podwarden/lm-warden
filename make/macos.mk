# LM Warden -- macOS app + DMG targets (Task 7).
#
# Included by the root Makefile (`include make/macos.mk`, mirroring
# make/operator.mk). Three targets:
#
#   make macos-test            -- `swift test` in macos/ + `pytest -m macos`
#                                 (with a job venv at .venv-macos if present,
#                                 else the repo's usual python).
#   make macos-app             -- `swift build -c release` and assemble
#                                 `dist/LM Warden.app` (the bundle the DMG is
#                                 built from), with the pinned Sparkle.framework
#                                 in Contents/Frameworks
#                                 (scripts/macos/embed-sparkle.sh; cached in
#                                 SPARKLE_CACHE). VERSION defaults to "dev".
#   make macos-dmg VERSION=x   -- macos-bundle-python (the app with its own
#                                 Python, signed and verified), then
#                                 scripts/macos/make-dmg.sh packs it into
#                                 `dist/LM-Warden-<version>.dmg`.
#   make macos-beta-dmg VERSION=<tag> -- a beta DMG of a release tag (no
#                                 Sparkle keys) + its .sha256; see its section.
#   make macos-requirements    -- re-lock requirements-macos.txt (the bundle's
#                                 hash-locked Python deps) from
#                                 requirements-macos.in via uv pip compile.
#   make macos-requirements-check -- re-resolve and diff; fails on drift.
#   make macos-bundle-python   -- macos-app, then embed the pinned
#                                 python-build-standalone + requirements-macos.txt
#                                 + app/ (scripts/macos/bundle-python.sh), the web
#                                 UI export (scripts/macos/bundle-webui.sh;
#                                 MACOS_WEBUI=0 leaves it out), sign
#                                 inside-out and verify. Downloads are cached in
#                                 PYTHON_CACHE (default .cache/macos-python).
#
# `dist/` is gitignored. The macOS CI jobs (test:macos / build:macos /
# publish:macos in .gitlab-ci.yml) drive these targets on the M1 runner.
#
# Recipes are POSIX-ish sh: `make` runs them under /bin/sh. The DMG script
# itself (scripts/macos/make-dmg.sh) is bash-3.2-safe because the M1 runner's
# /bin/bash is 3.2.

MACOS_PKG   := macos
MACOS_BIN   := LMWarden
DIST_DIR    := dist
MACOS_VENV  := .venv-macos

# `make macos-app` defaults VERSION to "dev" (a source checkout with no
# tag) via the recipe below. `make macos-dmg` does NOT default: VERSION is
# required on the command line, and the recipe's `test -n` guard fails the
# target with a usage message if it is unset. Do NOT add a global
# `VERSION ?= dev` here — it would make that guard dead.

# Where `swift build -c release` leaves the executable.
MACOS_RELEASE_BIN := $(MACOS_PKG)/.build/release/$(MACOS_BIN)

# The pinned Sparkle zip (P8; the pin lives in scripts/macos/embed-sparkle.sh
# and macos/Package.swift). SwiftPM fetches its own copy for linking;
# embed-sparkle.sh fetches, checks and places the shipped one from here.
SPARKLE_CACHE ?= $(HOME)/Library/Caches/lmw-macos-sparkle

.PHONY: macos-test macos-app macos-dmg macos-requirements macos-requirements-check macos-bundle-python macos-beta-dmg

# ---------------------------------------------------------------------------
# macos-test: Swift unit tests + the macos-marked pytest suite.
#
# `swift test` runs the LMWardenTests target (AppVersion etc.). `pytest -m
# macos` runs the DMG packaging test and any other Apple-Silicon-only tests;
# the default `addopts` in pyproject.toml deselects `macos`, so this is the
# only place it runs. A job-local venv at $(MACOS_VENV) (created by the
# test:macos CI job) wins over the system python so the app's Python deps are
# importable; locally the repo's usual python is fine.
# ---------------------------------------------------------------------------
macos-test:
	@echo "==> swift test ($(MACOS_PKG))"
	swift test --package-path $(MACOS_PKG)
	@echo "==> pytest -m macos"
	@if [ -n "$(VENV)" ] && [ -x "$(VENV)/bin/python" ]; then \
		PY="$(VENV)/bin/python"; \
	elif [ -x "$(MACOS_VENV)/bin/python" ]; then \
		PY="$(MACOS_VENV)/bin/python"; \
	elif [ -x ".venv/bin/python" ]; then \
		PY=".venv/bin/python"; \
	else \
		PY="python3"; \
	fi; \
	"$$PY" -m pytest -m macos tests/unit
	@echo "==> pytest tests/unit/farm (farm scripts need git/ps/zsh: run on the Mac lane)"
	@if [ -n "$(VENV)" ] && [ -x "$(VENV)/bin/python" ]; then \
		PY="$(VENV)/bin/python"; \
	elif [ -x "$(MACOS_VENV)/bin/python" ]; then \
		PY="$(MACOS_VENV)/bin/python"; \
	elif [ -x ".venv/bin/python" ]; then \
		PY=".venv/bin/python"; \
	else \
		PY="python3"; \
	fi; \
	"$$PY" -m pytest tests/unit/farm -n 4

# ---------------------------------------------------------------------------
# macos-app: release build + assemble dist/LM Warden.app.
#
# VERSION defaults to "dev" (a source checkout with no tag). The CI build:macos
# job overrides VERSION with the tag or short sha (through macos-dmg). The
# binary links @rpath/Sparkle.framework (macos/Package.swift's Sparkle
# binaryTarget) with LC_RPATH @executable_path/../Frameworks, and
# embed-sparkle.sh places the pinned framework there, refusing a zip whose
# sha256 differs or a framework other than the one SwiftPM linked
# (--build-dir). The
# assembled bundle is ad-hoc signed so `codesign -v` passes locally;
# macos-bundle-python re-signs it inside-out after embedding Python.
# ---------------------------------------------------------------------------
macos-app:
	@echo "==> swift build -c release ($(MACOS_PKG))"
	swift build -c release --package-path $(MACOS_PKG)
	@echo "==> assemble $(DIST_DIR)/LM Warden.app"
	@rm -rf "$(DIST_DIR)/LM Warden.app"
	@mkdir -p "$(DIST_DIR)/LM Warden.app/Contents/MacOS" \
	          "$(DIST_DIR)/LM Warden.app/Contents/Resources"
	@cp "$(MACOS_RELEASE_BIN)" "$(DIST_DIR)/LM Warden.app/Contents/MacOS/$(MACOS_BIN)"
	@VERSION="$(VERSION)"; \
	[ -n "$$VERSION" ] || VERSION="dev"; \
	scripts/macos/write-info-plist.sh \
		"$(DIST_DIR)/LM Warden.app/Contents/Info.plist" \
		"$$VERSION" "$(MACOS_BIN)"
	scripts/macos/embed-sparkle.sh "$(DIST_DIR)/LM Warden.app" \
		--cache-dir "$(SPARKLE_CACHE)" --build-dir "$(MACOS_PKG)/.build/release"
	@codesign --force --deep -s - "$(DIST_DIR)/LM Warden.app"

# ---------------------------------------------------------------------------
# macos-dmg: the bundled-Python app, packed into a DMG.
#
# VERSION is required (the CI job passes $CI_COMMIT_TAG or the short sha).
# OUT optionally overrides the output path (the test harness uses this to
# place the DMG in a tmp dir); it defaults to
# `$(DIST_DIR)/LM-Warden-$(VERSION).dmg`. The guard runs before anything is
# built, so macos-bundle-python is a recursive make AFTER it, not a
# prerequisite (a prerequisite would build first and fail last). It gets the
# same VERSION, and PYTHON_CACHE passes through. make-dmg.sh re-verifies the
# staged copy and refuses an app without the embedded interpreter or with a
# different version. bundle-python.sh refuses a dirty app/ and ships app/
# from `git archive HEAD`. A VERSION shaped like a release tag (v + digit) must
# be a well-formed vYYYY.MM.DD.N (scripts/macos/bundle-version.sh, which also
# derives the CFBundleVersion write-info-plist.sh writes: YYYYMMDDNN for a tag,
# 0 for a sha/dev build); a malformed one fails here, before anything is built.
# ---------------------------------------------------------------------------
macos-dmg:
	@test -n "$(VERSION)" || { echo "usage: make macos-dmg VERSION=<version> [OUT=<path>]" >&2; exit 2; }
	@scripts/macos/bundle-version.sh "$(VERSION)" >/dev/null || { echo "make macos-dmg: VERSION=$(VERSION) is a malformed release tag (want vYYYY.MM.DD.N)" >&2; exit 2; }
	$(MAKE) --no-print-directory macos-bundle-python VERSION="$(VERSION)" PYTHON_CACHE="$(PYTHON_CACHE)" MACOS_WEBUI="$(MACOS_WEBUI)" WEBUI_EXPORT="$(WEBUI_EXPORT)" MACOS_LLAMACPP="$(MACOS_LLAMACPP)" LLAMACPP_CACHE="$(LLAMACPP_CACHE)" SPARKLE_CACHE="$(SPARKLE_CACHE)"
	@mkdir -p "$(DIST_DIR)"
	@OUT="$(OUT)"; \
	if [ -z "$$OUT" ]; then OUT="$(DIST_DIR)/LM-Warden-$(VERSION).dmg"; fi; \
	scripts/macos/make-dmg.sh "$(DIST_DIR)/LM Warden.app" "$(VERSION)" "$$OUT"

# ---------------------------------------------------------------------------
# macos-requirements: re-lock the Python embedded in the app bundle.
#
# requirements-macos.in -> requirements-macos.txt (CPython 3.12, arm64 macOS
# 14+, wheels only, --generate-hashes). Needs uv; run it on an Apple Silicon
# Mac. tests/unit/test_requirements_macos.py keeps the shared pins identical
# to requirements.txt. The -check variant re-resolves to a temp file and
# fails if the committed lock has drifted.
# ---------------------------------------------------------------------------
macos-requirements:
	scripts/macos/lock-requirements.sh

macos-requirements-check:
	scripts/macos/lock-requirements.sh --check

# ---------------------------------------------------------------------------
# macos-bundle-python: dist/LM Warden.app with its own Python (P8, first slice).
#
# Runs on Apple Silicon only. bundle-python.sh lays the pinned interpreter,
# the hash-locked deps and app/ into Contents/Resources/python (the path the
# shell launches); sign-bundle.sh then re-signs every embedded Mach-O
# inside-out with the hardened runtime and the committed entitlements in
# macos/entitlements (its per-item lines go to $(DIST_DIR)/sign-bundle.log) and
# verify-bundle.sh checks the result, runtime flag and entitlement set
# included (--require-sparkle: the binary must link
# Sparkle and the framework's XPCs, Updater.app, Autoupdate and the framework
# itself must each verify). macos-dmg packs this app. PYTHON_CACHE
# can point outside the checkout (CI: a runner cache dir).
#
# The web UI (P8 slice 4), bundled by default: bundle-webui.sh builds the
# static export from HEAD's frontend/ (needs frontend/node_modules: `npm ci`
# in frontend/, never run by the build itself) into Contents/Resources/webui,
# which the launcher hands the server as VW_WEBUI_DIR; WEBUI_EXPORT=<dir>
# bundles a prebuilt export instead. It runs before signing, so the bundle
# signature seals it. MACOS_WEBUI=0 is the opt-out for a fast local build
# (no node, no next build): that bundle boots its API with no /ui (the
# launcher passes no VW_WEBUI_DIR when Resources/webui/index.html is absent).
# macos-dmg passes both through, so a DMG ships the UI unless told not to.
#
# llama.cpp-Metal (P5), bundled by default: bundle-llamacpp.sh builds the
# pinned llama-server (Metal, static, kernels embedded; gated on `otool -L`
# listing only system dylibs) and places it in Contents/Resources/python/bin,
# the dir the apple engine PATH resolves `llama-server` from. It runs after
# bundle-python.sh (which replaces python/ wholesale) and before signing. The
# build is cached in LLAMACPP_CACHE (pinned cmake + source + build tree).
# MACOS_LLAMACPP=0 leaves it out (that bundle has no llama.cpp engine).
# ---------------------------------------------------------------------------
PYTHON_CACHE ?= .cache/macos-python
MACOS_WEBUI ?= 1
WEBUI_EXPORT ?=
MACOS_LLAMACPP ?= 1
LLAMACPP_CACHE ?= $(HOME)/Library/Caches/lmw-macos-llamacpp

macos-bundle-python: macos-app
	@echo "==> embed python in $(DIST_DIR)/LM Warden.app"
	scripts/macos/bundle-python.sh "$(DIST_DIR)/LM Warden.app" --cache-dir "$(PYTHON_CACHE)"
	@if [ "$(MACOS_LLAMACPP)" = 0 ]; then \
		echo "==> llama-server not bundled (MACOS_LLAMACPP=0)"; \
	else \
		echo "==> build and bundle llama-server (Metal)"; \
		scripts/macos/bundle-llamacpp.sh "$(DIST_DIR)/LM Warden.app" --cache-dir "$(LLAMACPP_CACHE)"; \
	fi
	@if [ -n "$(WEBUI_EXPORT)" ]; then \
		echo "==> bundle the web UI export $(WEBUI_EXPORT)"; \
		scripts/macos/bundle-webui.sh "$(DIST_DIR)/LM Warden.app" --export-dir "$(WEBUI_EXPORT)"; \
	elif [ "$(MACOS_WEBUI)" = 0 ]; then \
		echo "==> web UI not bundled (MACOS_WEBUI=0)"; \
	else \
		echo "==> build and bundle the web UI export"; \
		scripts/macos/bundle-webui.sh "$(DIST_DIR)/LM Warden.app"; \
	fi
	@echo "==> sign-bundle.sh (log: $(DIST_DIR)/sign-bundle.log)"
	@scripts/macos/sign-bundle.sh "$(DIST_DIR)/LM Warden.app" - > "$(DIST_DIR)/sign-bundle.log" \
		|| { tail -n 20 "$(DIST_DIR)/sign-bundle.log"; exit 1; }
	@tail -n 1 "$(DIST_DIR)/sign-bundle.log"
	scripts/macos/verify-bundle.sh "$(DIST_DIR)/LM Warden.app" --require-sparkle

# ---------------------------------------------------------------------------
# macos-beta-dmg: a BETA DMG of a release tag (P8, OWNER DECISION 4).
#
#   make macos-beta-dmg VERSION=vYYYY.MM.DD.N
#
# A beta is ad-hoc signed and not notarized; it is published only as a GitHub
# pre-release download (publish/github-beta-release.sh) and never offered as an
# update. VERSION must be a strict release tag (bundle-version.sh: non-zero);
# a sha or dev build is exit 2 before anything builds. It runs macos-dmg with
# LMW_BETA_BUILD=1 (write-info-plist.sh then leaves the Sparkle keys out) and
# both Sparkle vars unset, writing dist/LM-Warden-<tag>-beta.dmg, then
# beta-dmg-seal.sh mounts it, refuses any SU* key or a version mismatch, and
# writes dist/LM-Warden-<tag>-beta.dmg.sha256. A refused DMG has no checksum.
# ---------------------------------------------------------------------------
macos-beta-dmg:
	@test -n "$(VERSION)" || { echo "make macos-beta-dmg: usage: make macos-beta-dmg VERSION=vYYYY.MM.DD.N" >&2; exit 2; }
	@bv=$$(scripts/macos/bundle-version.sh "$(VERSION)" 2>/dev/null) && [ "$$bv" != 0 ] \
		|| { echo "make macos-beta-dmg: VERSION=$(VERSION) is not a release tag (want vYYYY.MM.DD.N)" >&2; exit 2; }
	@rm -f "$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg" "$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg.sha256"
	env -u LMW_SPARKLE_FEED_URL -u LMW_SPARKLE_PUBLIC_ED_KEY LMW_BETA_BUILD=1 \
		$(MAKE) --no-print-directory macos-dmg VERSION="$(VERSION)" OUT="$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg"
	scripts/macos/beta-dmg-seal.sh "$(DIST_DIR)/LM-Warden-$(VERSION)-beta.dmg" "$(VERSION)"
