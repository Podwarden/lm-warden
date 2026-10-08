#!/bin/sh
# Assert that .env.example is the single source of truth for the env contract.
#
# Checks, in order:
#   1. every ${VAR...} interpolated in docker-compose.yml and in the override
#      that install.sh generates has a line in .env.example (active or
#      commented knob);
#   2. every VW_* knob in .env.example is read by the application (app/) or
#      referenced by the stack -- a knob nothing reads is drift;
#   3. every default install.sh gives a VW_* variable (${VAR:-default}) equals
#      the default shown for it in .env.example, and, where app/config.py
#      reads the variable with a literal default, equals that too;
#   4. VW_COOKIE_SECRET carries the REQUIRED and generate: markers
#      install.sh and the PodWarden Hub both act on.
#
# Runs anywhere with sh, grep, sed, awk and sort -- no Docker. Exit 1 on the
# first failing check, with every offending name listed.
#
# Usage: scripts/check-env-contract.sh [repo-root]

set -eu

ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"

fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }

# Names defined in .env.example: active `VAR=` lines and commented `# VAR=`
# knobs (the grammar the file's header documents).
example_names=$(sed -n 's/^\(# \)\{0,1\}\([A-Z_][A-Z0-9_]*\)=.*/\2/p' .env.example | sort -u)

# ${VAR}, ${VAR:-...}, ${VAR:?...} in docker-compose.yml.
compose_names=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*' docker-compose.yml | sed 's/^\${//' | sort -u || true)

# Same, in the override template inside install.sh. Every interpolation the
# template writes into YAML is spelled `\${VAR...}` there (escaped so the
# shell leaves it for Compose), which is exactly what separates it from the
# script's own $VARIABLES.
override_names=$(grep -oE '\\\$\{[A-Z_][A-Z0-9_]*' install.sh | sed 's/^\\\${//' | sort -u || true)

# ---- 1. everything interpolated is described ---------------------------------
missing=""
for n in $compose_names $override_names; do
  case "$n" in
    # Compose-internal names that are legitimately never in .env.
    COMPOSE_PROJECT_NAME|COMPOSE_FILE) continue ;;
  esac
  printf '%s\n' "$example_names" | grep -qx "$n" || missing="$missing $n"
done
if [ -n "$missing" ]; then
  bad "interpolated in docker-compose.yml / install.sh but not described in .env.example:$missing"
else
  say "ok: every interpolated variable is described in .env.example"
fi

# ---- 2. every described VW_* knob is read somewhere -----------------------------
stale=""
for n in $example_names; do
  case "$n" in VW_*) ;; *) continue ;; esac
  if ! grep -rq --include='*.py' "\"$n\"" app/ \
     && ! printf '%s\n' "$compose_names" | grep -qx "$n" \
     && ! printf '%s\n' "$override_names" | grep -qx "$n"; then
    stale="$stale $n"
  fi
done
if [ -n "$stale" ]; then
  bad "described in .env.example but read by nothing in app/, docker-compose.yml or install.sh:$stale"
else
  say "ok: every VW_* knob in .env.example is consumed"
fi

# ---- 3. defaults agree ------------------------------------------------------------
# install.sh: VW_X: "${VW_X:-default}"  -> "VW_X default"
installer_defaults=$(grep -oE '\\\$\{VW_[A-Z0-9_]+:-[^}]*\}' install.sh | sed 's/^\\\${//; s/}$//; s/:-/ /' | sort -u || true)
mismatch=""
for pair in $(printf '%s\n' "$installer_defaults" | tr ' ' '='); do
  n=${pair%%=*}; d=${pair#*=}
  # skip defaults that install.sh computes (contain a $) and empty ones
  case "$d" in *'$'*|"") continue ;; esac
  ex=$(sed -n "s/^\(# \)\{0,1\}$n=\(.*\)$/\2/p" .env.example | head -1 | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//')
  if [ -n "$ex" ] && [ "$ex" != "$d" ]; then
    mismatch="$mismatch $n(.env.example=$ex,install.sh=$d)"
  fi
  cfg=$(grep -oE "os\.environ\.get\(\"$n\", \"[^\"]*\"\)" app/config.py | sed 's/.*, "\(.*\)")/\1/' | head -1 || true)
  if [ -n "$cfg" ] && [ "$cfg" != "$d" ]; then
    case "$n" in
      # Truthy flags: the app default is "" (falsy); "false"/"0" mean the same.
      VW_GODMODE_ENABLED|VW_TRUST_PROXY_ORIGIN) case "$d" in false|0|"") continue ;; esac ;;
    esac
    mismatch="$mismatch $n(app/config.py=$cfg,install.sh=$d)"
  fi
done
if [ -n "$mismatch" ]; then
  bad "defaults disagree:$mismatch"
else
  say "ok: installer defaults match .env.example and app/config.py"
fi

# ---- 4. the secret's markers -----------------------------------------------------------
if grep -qE '^VW_COOKIE_SECRET=[[:space:]]*#.*REQUIRED' .env.example \
   && grep -qE '^VW_COOKIE_SECRET=[[:space:]]*#.*generate:hex(32|64)' .env.example; then
  say "ok: VW_COOKIE_SECRET is REQUIRED and generate:hex32/64"
else
  bad "VW_COOKIE_SECRET in .env.example must be blank with '# REQUIRED | generate:hex32' markers"
fi

exit $fail
