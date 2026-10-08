#!/bin/sh
# Compare the env contract the PodWarden Hub catalog renders for vllm-warden
# with this repository's .env.example, the declared source of truth.
#
# The Hub's install script names a bundle URL; the bundle carries the
# .env.example the Hub generates from its env_schema column. Any variable
# present on one side and not the other is drift: a knob the Hub offers that
# nothing here documents, or one documented here that a Hub install never
# writes. Values are reported too, but only names fail the check -- defaults
# are the application's business and app/config.py is checked separately.
#
# Needs curl and tar. HUB_SCRIPT_URL overrides the endpoint (tests, staging).
#
# Exit 0 in agreement, 1 on drift, 2 when the Hub could not be reached (the
# CI job is allow_failure, so either shows up orange, never red).

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HUB_SCRIPT_URL=${HUB_SCRIPT_URL:-https://podwarden.com/api/v1/catalog/install/vllm-warden/script}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/vw-hubdrift.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

curl -fsSL "$HUB_SCRIPT_URL" -o "$WORK/script.sh" || { echo "could not fetch $HUB_SCRIPT_URL" >&2; exit 2; }
bundle=$(sed -n 's/^BUNDLE_URL="\(.*\)"$/\1/p' "$WORK/script.sh" | head -1)
[ -n "$bundle" ] || { echo "no BUNDLE_URL in the Hub script (format changed?)" >&2; exit 2; }
curl -fsSL "$bundle" -o "$WORK/bundle.tgz" || { echo "could not fetch $bundle" >&2; exit 2; }
mkdir -p "$WORK/b"
tar -xzf "$WORK/bundle.tgz" -C "$WORK/b" --strip-components=1 || { echo "could not unpack the bundle" >&2; exit 2; }
[ -f "$WORK/b/.env.example" ] || { echo "the Hub bundle has no .env.example" >&2; exit 2; }

# Both files: names of active `VAR=` lines and, here, of commented `# VAR=`
# knobs (the grammar .env.example's header documents). Compose-internal names
# are excluded on both sides: a Hub install is not driven by this repo's
# compose file and has no reason to carry them, so their absence is not drift.
# scripts/check-env-contract.sh skips the same two for the same reason.
names() {
  sed -n 's/^\(# \)\{0,1\}\([A-Z_][A-Z0-9_]*\)=.*/\2/p' "$1" \
    | grep -vxE 'COMPOSE_PROJECT_NAME|COMPOSE_FILE' | sort -u
}
value() { sed -n "s/^\(# \)\{0,1\}$2=\(.*\)$/\2/p" "$1" | head -1 | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//'; }

names "$ROOT/.env.example" > "$WORK/repo.txt"
names "$WORK/b/.env.example" > "$WORK/hub.txt"

only_repo=$(comm -23 "$WORK/repo.txt" "$WORK/hub.txt" | tr '\n' ' ')
only_hub=$(comm -13 "$WORK/repo.txt" "$WORK/hub.txt" | tr '\n' ' ')

rc=0
if [ -n "$only_repo" ]; then
  echo "DRIFT: in this repo's .env.example but not in the Hub's: $only_repo"; rc=1
fi
if [ -n "$only_hub" ]; then
  echo "DRIFT: in the Hub's .env.example but not in this repo's: $only_hub"; rc=1
fi
for n in $(comm -12 "$WORK/repo.txt" "$WORK/hub.txt"); do
  a=$(value "$ROOT/.env.example" "$n"); b=$(value "$WORK/b/.env.example" "$n")
  [ "$a" = "$b" ] || echo "note: $n defaults differ: repo='$a' hub='$b'"
done
[ "$rc" -eq 0 ] && echo "ok: the Hub's env contract matches .env.example ($(wc -l < "$WORK/repo.txt" | tr -d ' ') names)"
exit $rc
