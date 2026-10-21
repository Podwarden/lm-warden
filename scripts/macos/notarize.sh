#!/bin/bash
# notarize.sh — notarize a signed .dmg or .app via notarytool and staple it (Task 82).
#
# Usage:
#   scripts/macos/notarize.sh <artifact.dmg|.app> <keychain-profile>
#
# Credentials come ONLY from the keychain profile, passed as
# `--keychain-profile "$profile"`. The script never passes an Apple ID,
# password, team id, key, key id or issuer, reads no credential env var and
# prints no secret (no set -x). The profile is made with
# `xcrun notarytool store-credentials "<profile>"` (an App Store Connect API
# key, or Apple ID + app-specific password -- either works). Per the P8 owner
# decision the CI job stores it in a per-job temporary keychain, built from
# masked+protected GitLab CI variables, that is on the keychain search list
# (codesign needs that anyway) and is deleted when the job ends; notarytool
# finds the profile through the search list. So the secret never reaches the
# repo, this script's argv or a log. CI wiring is a later orchestrator change;
# this is the script only. Output: the submission JSON / rejection log (job
# id, status, issues; no credential) goes to stderr, stdout has only the
# final success line.
#
# Steps:
#   1. A .app is zipped with `ditto -c -k --keepParent <app> <tmpdir>/<name>.zip`
#      (mktemp -d, removed by an EXIT trap) and the zip is submitted; a .dmg
#      is submitted as itself. Submit is
#      `xcrun notarytool submit <file> --keychain-profile <profile> --wait
#      --output-format json` with stdout captured.
#   2. `id` and `status` are parsed from that JSON with python3 (json from
#      stdin); a missing key or non-JSON output is a hard error.
#      A non-zero submit exit is a failure (exit 1); when its output still
#      names an id with a status other than Accepted (a finished rejection, or
#      a --wait that gave up), the log in step 3 is fetched as well.
#   3. status != Accepted: `xcrun notarytool log <id> --keychain-profile
#      <profile>` is run with its output on stderr (its own failure ignored),
#      then the result line is printed and the script exits 1; the stapler is
#      not called.
#   4. Accepted: `xcrun stapler staple <artifact>` then `xcrun stapler
#      validate <artifact>` on the original .app or .dmg (not the zip).
#
# Exit codes: 0 accepted and stapled, 1 a notarytool or stapler call failed
# (the step is named), 2 usage error (arg count, empty profile, artifact
# missing or not a .dmg/.app; usage line on stderr).
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no associative arrays, no
# ${x,,}, no mapfile. Paths may contain spaces, so everything is quoted.
set -eu

usage() {
    echo "usage: notarize.sh <artifact.dmg|.app> <keychain-profile>" >&2
}

artifact=${1:-}
profile=${2:-}
if [ "$#" -ne 2 ] || [ -z "$profile" ] || [ -z "$artifact" ]; then
    usage
    exit 2
fi
if [ -f "$artifact" ]; then
    case "$artifact" in
        *.dmg) ;;
        *) usage; exit 2 ;;
    esac
elif [ -d "$artifact" ]; then
    case "$artifact" in
        *.app) ;;
        *) usage; exit 2 ;;
    esac
else
    usage
    exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

submit_file="$artifact"
case "$artifact" in
    *.app)
        app_base=${artifact##*/}
        app_base=${app_base%.app}
        submit_file="$tmpdir/${app_base}.zip"
        ditto -c -k --keepParent "$artifact" "$submit_file" || {
            echo "notarize.sh: ditto failed: $artifact" >&2
            exit 1
        }
        ;;
esac

# Prints "<id>\n<status>" from notarytool JSON on stdin; non-zero on bad input.
parse_submission() {
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["id"] + "\n" + d["status"])'
}

submit_rc=0
submit_out=$(xcrun notarytool submit "$submit_file" --keychain-profile "$profile" --wait --output-format json) || submit_rc=$?
if [ "$submit_rc" -ne 0 ]; then
    printf '%s\n' "$submit_out" >&2
    echo "notarize.sh: notarytool submit failed" >&2
    # notarytool may exit non-zero for a submission that finished rejected:
    # still fetch the log when the output names one.
    if parsed=$(printf '%s' "$submit_out" | parse_submission 2>/dev/null); then
        id=${parsed%%
*}
        status=${parsed#*
}
        if [ -n "$id" ] && [ "$status" != "Accepted" ]; then
            xcrun notarytool log "$id" --keychain-profile "$profile" >&2 || true
        fi
    fi
    exit 1
fi

parsed=$(printf '%s' "$submit_out" | parse_submission) || {
    echo "notarize.sh: could not parse notarytool output" >&2
    printf '%s\n' "$submit_out" >&2
    exit 1
}
id=${parsed%%
*}
status=${parsed#*
}

if [ "$status" != "Accepted" ]; then
    xcrun notarytool log "$id" --keychain-profile "$profile" >&2 || true
    echo "notarize.sh: notarization $status (id $id)" >&2
    exit 1
fi

# stapler's own chatter goes to stderr so stdout stays the one result line.
xcrun stapler staple "$artifact" >&2 || {
    echo "notarize.sh: stapler staple failed: $artifact" >&2
    exit 1
}
xcrun stapler validate "$artifact" >&2 || {
    echo "notarize.sh: stapler validate failed: $artifact" >&2
    exit 1
}

echo "notarize.sh: accepted and stapled ${artifact##*/} (id $id)"
