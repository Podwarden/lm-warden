# syntax=docker/dockerfile:1
# ^ Must be the FIRST line — a parser directive is only recognised before any
# other comment or instruction. `env=` on a secret mount needs the dockerfile
# 1.10+ frontend, hence the unpinned `1`.

# ---------------------------------------------------------------------------
# @podwarden/chat-ui pin. THREE args, all declared here so this block is the
# single greppable place the pin lives (the same discipline VW_BASE_DIGEST and
# LLAMACPP_BUILD have in the root Dockerfile).
#
#   CHATUI_REPO     where the source comes from. Overridable so a fork can be
#                   built without editing this file.
#   CHATUI_REF      the exact commit built. The public mirror carries NO tags —
#                   it receives one squashed "Release vX" commit per release —
#                   so a full commit SHA is the only pin available, and it is
#                   the stronger one anyway.
#   CHATUI_VERSION  what that commit's package.json says. Asserted twice below
#                   (against the lockfile, and against the built tarball), so a
#                   ref/version mismatch fails the build instead of silently
#                   shipping a different component than package-lock.json
#                   resolved dependencies for.
#
# tests/unit/system/test_chatui_pin.py fails if CHATUI_VERSION drifts from the
# `@podwarden/chat-ui` version in frontend/package.json.
#
# Bumping chat-ui: bump the version in package.json, `npm install` to refresh
# the lockfile, then set CHATUI_VERSION and CHATUI_REF to the release commit on
# github.com/Podwarden/chat-ui whose package.json carries that version.
ARG CHATUI_REPO=https://github.com/Podwarden/chat-ui.git
ARG CHATUI_REF=83f100ff28a412d530d170cc8b9bf7c9787656ab
ARG CHATUI_VERSION=0.1.23

# ---------------------------------------------------------------------------
# Stage 1 — compile chat-ui from source.
#
# WHY BUILD RATHER THAN CONSUME THE PUBLISHED TARBALL. @podwarden/chat-ui is on
# public npm and the lockfile resolves it there, so `npm ci` alone produces a
# working image with no credentials. What it does NOT produce is an image built
# from source: the npm tarball ships `dist/` only (see the package's `files`),
# so the bundle an operator runs would be a binary they did not compile. This
# stage compiles it, in the image, from the public GitHub tree — the same thing
# the root Dockerfile does with llama.cpp, and for the same reason.
#
# The two are not in tension: the build here is REPRODUCIBLE against the
# published copy. `npm pack` on the freshly compiled 0.1.23 tree emits sha1
# 6296e1db... / sha512-AiBzIbTJCMtFw...DqVsyEoruMEJg==, byte-identical to the
# tarball package-lock.json's `integrity` already pins on registry.npmjs.org
# (which npm publishes from that same GitHub repo by OIDC trusted publishing,
# with a provenance attestation). So the registry copy is not a second artifact
# that happens to share a version number — and compiling it here is a
# verification of that, not a workaround for distrust of it.
#
# A plain npm git dependency ("github:Podwarden/chat-ui#SHA") does NOT work and
# is not the mechanism here. chat-ui declares `prepack`, not `prepare`, and npm
# only builds a git dependency that declares `prepare`; with `files` listing
# `dist`, npm 10.8.2 installs LICENSE/NOTICE/README/config/package.json and no
# code at all — measured, not assumed. Making a git dependency work would mean
# changing chat-ui's package.json, i.e. blocking this repo on another repo's
# release cycle. Cloning it here does not.
FROM node:20-alpine AS chatui
ARG CHATUI_REPO
ARG CHATUI_REF
RUN apk add --no-cache git
# A pinned SHA, checked out detached: no branch, no tag, nothing that can move
# under us. `git clone` without --depth because a SHA that is not the branch tip
# is not fetchable from a shallow clone.
RUN git clone "$CHATUI_REPO" /src \
 && git -C /src checkout --detach "$CHATUI_REF" \
 && git -C /src rev-parse HEAD
WORKDIR /src
# chat-ui's OWN committed lockfile — `npm ci`, not `npm install`, so its build
# toolchain (tsup, typescript, esbuild) is pinned too. All public npm; no token.
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci
RUN npm run build
# --ignore-scripts because `prepack` would run the build a second time. The
# tarball is what `npm publish` would upload from this tree: `files` applied,
# nothing else.
# `mkdir -p /out` is load-bearing: npm 10.8.2's --pack-destination does NOT
# create the directory, it opens the tarball for writing inside it and dies with
# `ENOENT ... open '/out/podwarden-chat-ui-0.1.23.tgz'` — a message that reads
# like a missing tarball rather than a missing directory.
RUN mkdir -p /out \
 && npm pack --ignore-scripts --pack-destination /out \
 && ls -l /out

# ---------------------------------------------------------------------------
# Stage 2 — dependencies.
FROM node:20-alpine AS deps
ARG CHATUI_VERSION
WORKDIR /app
# `.npmrc*` is a GLOB on purpose: the file is not in the public tree (it routes
# the @podwarden scope at a private registry that a public clone cannot reach),
# and a bare `.npmrc` here made every credential-free build die at COPY with
# `"/.npmrc": not found` before npm was even invoked. The glob copies it when a
# checkout has one and is a no-op when it does not.
COPY package.json package-lock.json* .npmrc* ./
# The npm secret is OPTIONAL (no `required=true`). It used to be mandatory,
# which meant a public clone — with no private registry in its lockfile and
# nothing to authenticate to — still failed with `secret npm: not found`
# before npm ran at all. It stays available for a checkout that DOES carry a
# private .npmrc; when no secret is passed the mount is empty, npm reads no
# token, and the public registry serves everything in the lockfile anonymously.
#
# The guard below keeps the fail-fast that `required=true` was there for,
# without extending it to builds that need no credential: a token is demanded
# only when this checkout actually has an .npmrc to authenticate against.
# Otherwise a forgotten `--secret id=npm,...` on an internal build would
# surface as a late, confusing npm E401 partway through the install.
#
# The npm cache is a BuildKit cache mount, not a layer: `npm ci` deletes and
# recreates node_modules every time, so when this step IS invalidated (a
# package-lock.json change) the tarballs are already local and only the
# unpack cost remains. sharing=locked because npm's cache is not safe against
# concurrent writers, and build:ui / publish:images can overlap on a runner.
RUN --mount=type=secret,id=npm,env=NPM_TOKEN \
    --mount=type=cache,target=/root/.npm,sharing=locked \
    set -eu; \
    if [ -f .npmrc ] && [ -z "${NPM_TOKEN:-}" ]; then \
      echo "FATAL: this checkout has frontend/.npmrc, which routes the" >&2; \
      echo "@podwarden scope at an authenticated registry, but no npm secret" >&2; \
      echo "was passed. Re-run with --secret id=npm,env=NPM_TOKEN." >&2; \
      exit 1; \
    fi; \
    npm ci

# Swap the registry copy of chat-ui for the one stage 1 compiled.
#
# `npm ci` still installs it from the lockfile first, and that is deliberate:
# the lockfile is what pins chat-ui's ~20 runtime dependencies (shiki, katex,
# react-markdown, ...) with integrity hashes, and those are installed into the
# top-level tree, not inside the package. We keep that closure and replace only
# the package's own files. The result is: dependency graph from the lockfile,
# component code compiled here from GitHub source.
#
# The overlay copies OVER the directory rather than deleting it, preserving any
# nested node_modules npm placed there to resolve a version conflict.
COPY --from=chatui /out/ /tmp/chatui/
RUN set -eu; \
    tgz="$(ls /tmp/chatui/*.tgz)"; \
    pkg=/app/node_modules/@podwarden/chat-ui; \
    locked="$(node -p "require('$pkg/package.json').version")"; \
    if [ "$locked" != "${CHATUI_VERSION}" ]; then \
      echo "FATAL: package-lock.json installed @podwarden/chat-ui@${locked}," >&2; \
      echo "but this Dockerfile pins CHATUI_VERSION=${CHATUI_VERSION}." >&2; \
      exit 1; \
    fi; \
    rm -rf /tmp/cu; mkdir -p /tmp/cu; \
    tar -xzf "$tgz" -C /tmp/cu --strip-components=1; \
    find "$pkg" -mindepth 1 -maxdepth 1 ! -name node_modules -exec rm -rf {} +; \
    cp -a /tmp/cu/. "$pkg"/; \
    rm -rf /tmp/cu /tmp/chatui; \
    built="$(node -p "require('$pkg/package.json').version")"; \
    if [ "$built" != "${CHATUI_VERSION}" ]; then \
      echo "FATAL: the source at CHATUI_REF builds @podwarden/chat-ui@${built}," >&2; \
      echo "not the pinned ${CHATUI_VERSION}. Fix CHATUI_REF or CHATUI_VERSION." >&2; \
      exit 1; \
    fi; \
    test -f "$pkg/dist/index.js"; \
    test -f "$pkg/dist/theme.css"; \
    echo "@podwarden/chat-ui ${CHATUI_VERSION} compiled from source and installed"

# ---------------------------------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# next.config.ts:rewrites() reads BACKEND_URL — but Next 15 standalone bakes
# rewrites into routes-manifest.json at BUILD time, so passing BACKEND_URL at
# runtime is a no-op. Hard-coding the Docker service name here works for both
# the local compose stack and the Hub catalogue compose (both expose the API
# as `api:8080`). If a future deployment uses a different service name, this
# becomes a build arg.
ENV BACKEND_URL=http://api:8080

# #67 fix — bake the release identity into the UI bundle. Next.js
# inlines ``NEXT_PUBLIC_*`` env vars into the client bundle at build
# time, so consumers of these values (today: none; tomorrow: the
# version banner footer + e2e tooling) get the same identity as the
# backend's GET /api/version response. The default fallbacks ("dev" /
# "unknown") mirror the backend Dockerfile (see Dockerfile §74-77 in
# the repo root) so a no-build-arg local build produces "v? · ?" in
# the footer fallback path. CI passes both via the build:ui job in
# .gitlab-ci.yml. Manual release builds pass them via the buildx
# command in docs/releasing.md.
ARG VW_BUILD_VERSION=dev
ARG VW_BUILD_SHA=unknown
ENV NEXT_PUBLIC_VW_BUILD_VERSION=${VW_BUILD_VERSION} \
    NEXT_PUBLIC_VW_BUILD_SHA=${VW_BUILD_SHA}

# PERFORMANCE NOTE, deliberately not "fixed". NEXT_PUBLIC_VW_BUILD_SHA changes
# on every commit and is set immediately above this line, so this step is
# invalidated on every commit even when `COPY . .` reports CACHED -- measured at
# 53-70 s per build (job 228612). It cannot be reordered away: Next inlines
# NEXT_PUBLIC_* into the client bundle AT BUILD TIME, so the value has to be in
# the environment before `npm run build` runs, and moving the ENV after the
# build would silently return the footer to the `dev · unknown` banner that #45
# and #67 exist to prevent. Making the UI read its identity at runtime instead
# would recover the 53-70 s, but that is a product change to how the version
# banner works, not a build optimisation, and it belongs in its own MR.
RUN mkdir -p public && npm run build

FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production PORT=3000
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs
COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=build --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
