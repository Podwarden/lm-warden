#!/bin/bash
# write-info-plist.sh — write the LM Warden.app Info.plist (Task 7).
#
# Usage:
#   scripts/macos/write-info-plist.sh <plist_path> <version> <executable> [bundle_id]
#
# One writer for both `make macos-app` (which assembles dist/LM Warden.app)
# and `scripts/macos/make-dmg.sh` (which stages the app for the DMG), so the
# two copies cannot drift. The bundle id lives here and in
# macos/Sources/LMWarden/AppVersion.swift (which the tests assert against
# via the DMG); keep them in sync. The optional bundle_id defaults to
# com.podwarden.lmwarden, so the 3-argument callers above keep producing the
# app id unchanged.
#
# Sparkle auto-update (P8): LMW_SPARKLE_FEED_URL (the appcast) and
# LMW_SPARKLE_PUBLIC_ED_KEY (the public half of the EdDSA keypair) are read from
# the environment and, when both are set, written as SUFeedURL / SUPublicEDKey.
# The key is the PUBLIC half only: the private EdDSA key never touches this
# script or the repo. No default feed URL is shipped until the owner picks one,
# so neither set leaves the plist unchanged.
#
# CFBundleVersion (P8, the number Sparkle compares) is NOT <version> verbatim:
# scripts/macos/bundle-version.sh maps a CalVer release tag vYYYY.MM.DD.N to the
# integer YYYYMMDDNN (v2026.10.21.3 -> 2026102103) and any other version (a
# short sha, "dev") to 0. CFBundleShortVersionString stays <version> as given
# (human-readable; make-dmg.sh checks it against the DMG name). A malformed
# release tag is exit 2 with no file. A dev build (CFBundleVersion 0) never
# gets SUFeedURL / SUPublicEDKey even when both env vars are set: it is left
# out with a note on stderr, so the plist is byte-identical to the unset case.
#
# Beta builds (P8, OWNER DECISION 4): LMW_BETA_BUILD=1 (set by `make
# macos-beta-dmg`) leaves SUFeedURL / SUPublicEDKey out even for a release tag
# with both vars set, with a note on stderr, so the plist is byte-identical to
# a feed-less release. A beta is a download only: it must never check a feed.
# "" and "0" mean not a beta; any other value is exit 2 with no file.
#
# Bash-3.2-safe (printf only, no here-strings, no mapfile).
set -eu

usage="usage: write-info-plist.sh <plist_path> <version> <executable> [bundle_id]"

plist=${1:?$usage}
version=${2:?$usage}
executable=${3:?$usage}

# The 4th argument is the seam the queued bundle-smoke task uses to build a
# smoke bundle with its own identity. Default to the app id only when it is not
# given at all (3 args) so the make/macos.mk and make-dmg.sh callers are
# unchanged; a given-but-empty 4th arg is rejected like any other invalid id.
# Restrict it to a conservative id charset (first char alphanumeric, then
# alphanumerics, dots, dashes) so a value can never break the plist XML;
# checked before mkdir/redirect so an invalid id writes no file.
if [ "$#" -gt 3 ]; then
    bundle_id=${4}
else
    bundle_id=com.podwarden.lmwarden
fi
# The character classes are spelled out letter by letter, not as A-Z ranges:
# under a UTF-8 locale bash 3.2 ranges also match non-ASCII letters (e.g. an
# accented e), which a bundle id must not contain.
alnum=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
# Leading char must be alphanumeric (rejects empty and a leading dot).
case "$bundle_id" in
    ["$alnum"]*)
        ;;
    *)
        printf '%s\n' "write-info-plist.sh: invalid bundle id" "$usage" >&2
        exit 2
        ;;
esac
# Every other char must be an alphanumeric, dot or dash (rejects spaces,
# <, &, newlines and every other control/symbol).
case "$bundle_id" in
    *[!"$alnum".-]*)
        printf '%s\n' "write-info-plist.sh: invalid bundle id" "$usage" >&2
        exit 2
        ;;
esac

# <version> lands in the plist XML as CFBundleShortVersionString: restrict it
# to alphanumerics, dots and dashes (a tag, a short sha, "dev", "1.2.3-test")
# so it can never break the XML.
case "$version" in
    *[!"$alnum".-]*)
        printf '%s\n' "write-info-plist.sh: invalid version" "$usage" >&2
        exit 2
        ;;
esac
# CFBundleVersion: YYYYMMDDNN for a release tag, 0 for a dev build; a
# malformed tag fails here (bundle-version.sh prints why), before any file.
bundle_version=$(/bin/bash "$(dirname "$0")/bundle-version.sh" "$version") || {
    printf '%s\n' "$usage" >&2
    exit 2
}

# Sparkle auto-update keys (P8): SUFeedURL (the appcast) and SUPublicEDKey (the
# public half of the EdDSA keypair). Both are optional and must be set together;
# neither set (or both empty) leaves the plist unchanged, like the bundle id.
# The public key is the PUBLIC half only; the private EdDSA key never touches
# this script or the repo. Checked before mkdir/redirect so a bad value writes
# no file, like the bundle id.
feed_url=${LMW_SPARKLE_FEED_URL-}
ed_key=${LMW_SPARKLE_PUBLIC_ED_KEY-}
if [ -z "$feed_url" ] || [ -z "$ed_key" ]; then
    if [ -z "$feed_url" ] && [ -z "$ed_key" ]; then
        feed_url=
        ed_key=
    else
        printf '%s\n' \
            "write-info-plist.sh: LMW_SPARKLE_FEED_URL and LMW_SPARKLE_PUBLIC_ED_KEY must be set together" \
            "$usage" >&2
        exit 2
    fi
else
    # The feed URL must start with https:// and have at least one char after it
    # (rejects a bare scheme and "https://" with nothing else).
    case "$feed_url" in
        https://?*)
            ;;
        *)
            printf '%s\n' "write-info-plist.sh: invalid LMW_SPARKLE_FEED_URL" "$usage" >&2
            exit 2
            ;;
    esac
    # Every char after the scheme must be in a conservative set that cannot
    # break the plist XML (no ? & = < > quotes, spaces or non-ASCII).
    url_alnum=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
    url_tail=${feed_url#https://}
    case "$url_tail" in
        *[!"$url_alnum".:~/_%-]*)
            printf '%s\n' "write-info-plist.sh: invalid LMW_SPARKLE_FEED_URL" "$usage" >&2
            exit 2
            ;;
    esac
    # The public key is the base64 of a 32-byte Ed25519 public key: exactly 44
    # chars, the first 43 from the standard base64 alphabet, the 44th '='.
    if [ "${#ed_key}" -ne 44 ]; then
        printf '%s\n' "write-info-plist.sh: invalid LMW_SPARKLE_PUBLIC_ED_KEY" "$usage" >&2
        exit 2
    fi
    b64=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/
    b64_head=${ed_key%?}
    case "$b64_head" in
        *[!"$b64"]*)
            printf '%s\n' "write-info-plist.sh: invalid LMW_SPARKLE_PUBLIC_ED_KEY" "$usage" >&2
            exit 2
            ;;
    esac
    if [ "${ed_key:(-1)}" != "=" ]; then
        printf '%s\n' "write-info-plist.sh: invalid LMW_SPARKLE_PUBLIC_ED_KEY" "$usage" >&2
        exit 2
    fi
fi

# The beta switch: only "", "0" or "1" (a typo must not silently mean either).
beta=${LMW_BETA_BUILD-}
case "$beta" in
    ""|0|1)
        ;;
    *)
        printf '%s\n' "write-info-plist.sh: invalid LMW_BETA_BUILD (want 1, 0 or unset)" "$usage" >&2
        exit 2
        ;;
esac
if [ -n "$feed_url" ] && [ "$beta" = "1" ]; then
    printf '%s\n' "write-info-plist.sh: beta build (LMW_BETA_BUILD=1); leaving SUFeedURL/SUPublicEDKey out" >&2
    feed_url=
    ed_key=
fi

# A dev build never carries the feed keys: with CFBundleVersion 0 every
# appcast item would look newer, so it must not check a feed at all.
if [ -n "$feed_url" ] && [ "$bundle_version" = "0" ]; then
    printf '%s\n' "write-info-plist.sh: '$version' is not a release tag; leaving SUFeedURL/SUPublicEDKey out" >&2
    feed_url=
    ed_key=
fi

mkdir -p "$(dirname "$plist")"

{
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '<plist version="1.0">\n<dict>\n'
    printf '  <key>CFBundleName</key><string>LM Warden</string>\n'
    printf '  <key>CFBundleDisplayName</key><string>LM Warden</string>\n'
    printf '  <key>CFBundleIdentifier</key><string>%s</string>\n' "$bundle_id"
    printf '  <key>CFBundleVersion</key><string>%s</string>\n' "$bundle_version"
    printf '  <key>CFBundleShortVersionString</key><string>%s</string>\n' "$version"
    printf '  <key>CFBundleExecutable</key><string>%s</string>\n' "$executable"
    printf '  <key>CFBundlePackageType</key><string>APPL</string>\n'
    printf '  <key>CFBundleSignature</key><string>????</string>\n'
    printf '  <key>LSMinimumSystemVersion</key><string>14.0</string>\n'
    printf '  <key>NSHighResolutionCapable</key><true/>\n'
    printf '  <key>LSUIElement</key><true/>\n'
    printf '  <key>NSLocalNetworkUsageDescription</key><string>LM Warden uses your local network only when you turn on Share on Local Network, so other devices on your network can reach its model server.</string>\n'
    if [ -n "$feed_url" ]; then
        printf '  <key>SUFeedURL</key><string>%s</string>\n' "$feed_url"
        printf '  <key>SUPublicEDKey</key><string>%s</string>\n' "$ed_key"
    fi
    printf '</dict>\n</plist>\n'
} > "$plist"
