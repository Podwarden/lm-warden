#!/usr/bin/env bash
# Verify Caddy bare-path + wildcard + favicon redirects, and that the landing
# page's assets and text routes are proxied to the api.
# Runs caddy:2-alpine + hashicorp/http-echo mocks, curls each path, asserts
# the Location header is exactly /ui/<path> (NOT /ui/<path>/<path>).
# Exits non-zero on any mismatch. Safe to run from CI or QA harness.
#
# Usage: scripts/smoke-caddy-redirects.sh [/path/to/Caddyfile]
# Default Caddyfile: deploy/caddy/Caddyfile relative to repo root.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CADDYFILE="${1:-$REPO_ROOT/deploy/caddy/Caddyfile}"
COMPOSE_DIR=$(mktemp -d)
trap 'docker compose -f "$COMPOSE_DIR/compose.yml" down -v --remove-orphans >/dev/null 2>&1 || true; rm -rf "$COMPOSE_DIR"' EXIT

cat > "$COMPOSE_DIR/compose.yml" <<EOF
services:
  caddy:
    image: caddy:2-alpine
    ports: ["18080:8080"]
    volumes:
      - $CADDYFILE:/etc/caddy/Caddyfile:ro
    depends_on: [api, ui]
  api:
    image: hashicorp/http-echo
    command: ["-text=mock-api", "-listen=:8080"]
    expose: ["8080"]
  ui:
    image: hashicorp/http-echo
    command: ["-text=mock-ui", "-listen=:3000"]
    expose: ["3000"]
EOF

docker compose -f "$COMPOSE_DIR/compose.yml" up -d >/dev/null
sleep 3

fail=0
check() {
  local path="$1" expected_status="$2" expected_loc="$3"
  local got_status got_loc
  got_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:18080$path")
  got_loc=$(curl -s -o /dev/null -w "%{redirect_url}" "http://localhost:18080$path")
  if [[ "$got_status" == "$expected_status" && "$got_loc" == "http://localhost:18080$expected_loc" ]]; then
    printf "PASS %-30s -> %s %s\n" "$path" "$got_status" "$got_loc"
  else
    printf "FAIL %-30s -> %s %s (expected %s %s)\n" "$path" "$got_status" "$got_loc" "$expected_status" "http://localhost:18080$expected_loc"
    fail=1
  fi
}

# Bare paths -> 308 /ui/<path>
for p in login models settings setup stats tokens cache chat; do
  check "/$p" 308 "/ui/$p"
done

# Wildcard paths -> 308 /ui/<path>/<suffix>  (regression guard for {uri} bug)
check /login/sub-page    308 /ui/login/sub-page
check /models/abc        308 /ui/models/abc
check /chat/bar/baz      308 /ui/chat/bar/baz

# Favicon -> 301 /ui/icon.svg
check /favicon.ico 301 /ui/icon.svg

# Landing companions are PROXIED to the api (no redirect): the page's
# assets and the crawler / LLM text routes. The mock api answers every path
# with "mock-api", so the body says which backend Caddy picked.
proxied() {
  local path="$1" got_status got_body
  got_status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:18080$path")
  got_body=$(curl -s "http://localhost:18080$path")
  if [[ "$got_status" == 200 && "$got_body" == mock-api* ]]; then
    printf "PASS %-30s -> %s api\n" "$path" "$got_status"
  else
    printf "FAIL %-30s -> %s %q (expected 200 from api)\n" "$path" "$got_status" "$got_body"
    fail=1
  fi
}
for p in / /_landing /_landing/assets/shot-stats.webp /robots.txt /sitemap.xml /llms.txt /llms-full.txt; do
  proxied "$p"
done

exit $fail
