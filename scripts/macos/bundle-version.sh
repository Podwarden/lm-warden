#!/bin/bash
# bundle-version.sh — map a build VERSION to the app's CFBundleVersion (P8).
#
# Usage:
#   scripts/macos/bundle-version.sh <version>
#
# Prints the CFBundleVersion for <version> on stdout and exits 0, or exits 2
# (usage / malformed release tag) with nothing on stdout.
#
# Why: Sparkle decides "is the appcast item newer?" by comparing CFBundleVersion
# (sparkle:version in the appcast), not the human-readable
# CFBundleShortVersionString. The build VERSION is either a CalVer release tag
# `vYYYY.MM.DD.N` (docs/releasing.md; N is a daily counter from 1) or, for a
# non-tag build, the short git sha / "dev". Neither is a safe CFBundleVersion
# as-is: a sha does not order at all, and Apple documents CFBundleVersion as
# one to three period-separated integers (a 4-part tag is not).
#
# The scheme:
#
#   release tag vYYYY.MM.DD.N  ->  YYYYMMDDNN   (one integer, N zero-padded)
#                                  e.g. v2026.10.21.3 -> 2026102103
#   anything else (dev build)  ->  0
#
# One integer orders the same under every comparator (Sparkle's
# SUStandardVersionComparator compares numeric runs as numbers; it is also a
# plain integer for Apple's tooling): later date > earlier date across
# day/month/year boundaries, and N=10 > N=9 on the same day. It fits in a
# signed 32-bit int for every accepted year (max 2099123199).
#
# A dev build's CFBundleVersion is 0: it sorts below every release, so it can
# never be mistaken for a newer build than one a user has. write-info-plist.sh
# also leaves the Sparkle feed keys out for a dev build, so a dev build never
# checks a feed at all.
#
# Strict tag validation: a VERSION that starts with "v" followed by a digit is
# a release tag candidate and must be exactly vYYYY.MM.DD.N with
#   YYYY 2000..2099, MM 01..12, DD a real day of that month (leap years
#   counted), N 1..99 with no leading zero;
# anything else in that shape (v2026.13.01.1, v2026.02.30.1, v2026.10.21.0,
# v2026.10.21.01, v2026.10.21.100, v2026.1.2.3, trailing junk) is exit 2, so a
# mistyped tag fails the build instead of shipping as a dev build. N is capped
# at 99 because it is two digits of the integer.
#
# Bash-3.2-safe (the M1 runner's /bin/bash is 3.2): no =~ captures, no mapfile.
set -eu

usage="usage: bundle-version.sh <version>"

[ "$#" -eq 1 ] && [ -n "$1" ] || { printf '%s\n' "$usage" >&2; exit 2; }
version=$1

bad() {
    printf '%s\n' "bundle-version.sh: malformed release tag '$version' (want vYYYY.MM.DD.N, N 1..99)" >&2
    exit 2
}

case "$version" in
    v[0-9]*) ;;
    *) printf '0\n'; exit 0 ;;
esac

# Spelled-out digit classes, not ranges (bash 3.2 ranges are locale-dependent).
d='[0123456789]'
# shellcheck disable=SC2254
case "$version" in
    v$d$d$d$d.$d$d.$d$d.$d | v$d$d$d$d.$d$d.$d$d.$d$d) ;;
    *) bad ;;
esac

rest=${version#v}
year=${rest%%.*}; rest=${rest#*.}
month=${rest%%.*}; rest=${rest#*.}
day=${rest%%.*}
n=${rest#*.}

# 10# forces base 10: "08"/"09" are invalid octal otherwise.
y=$((10#$year)); m=$((10#$month)); dd=$((10#$day))

[ "$y" -ge 2000 ] && [ "$y" -le 2099 ] || bad
[ "$m" -ge 1 ] && [ "$m" -le 12 ] || bad
case "$n" in
    0*) bad ;;  # N=0 and leading zeros (01) are not tags
esac

case "$m" in
    4|6|9|11) max=30 ;;
    2)
        if [ $((y % 4)) -eq 0 ] && { [ $((y % 100)) -ne 0 ] || [ $((y % 400)) -eq 0 ]; }; then
            max=29
        else
            max=28
        fi
        ;;
    *) max=31 ;;
esac
[ "$dd" -ge 1 ] && [ "$dd" -le "$max" ] || bad

printf '%s%s%s%02d\n' "$year" "$month" "$day" "$((10#$n))"
