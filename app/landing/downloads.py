"""Where the Mac beta download lives (one place, one line per beta).

The DMG stays on GitHub; lmwarden.com/download/mac and
/download/mac.sha256 are short redirects to it, so every link on the website
and in the README is stable. To move to the next beta, change MAC_BETA_TAG
and MAC_BETA_VERSION: the redirects, the DMG name in the landing page's
checksum command (the ``{{mac_dmg}}`` slot) and the release-notes link all
derive from them. The README necessarily hard-codes the version (it is
rendered by GitHub, not by this app), so edit its release line and checksum
command by hand.
"""

from __future__ import annotations

#: The GitHub pre-release tag of the current Mac beta, and the release
#: version its DMG was built from.
MAC_BETA_TAG = "v2026.10.21.2-macos-beta"
MAC_BETA_VERSION = "v2026.10.21.2"

_RELEASES = "https://github.com/Podwarden/lm-warden/releases"
_DOWNLOAD_BASE = f"{_RELEASES}/download/{MAC_BETA_TAG}"

MAC_DMG_NAME = f"LM-Warden-{MAC_BETA_VERSION}-beta.dmg"
MAC_DMG_URL = f"{_DOWNLOAD_BASE}/{MAC_DMG_NAME}"
MAC_SHA256_URL = f"{MAC_DMG_URL}.sha256"
#: The release page, for the release notes.
MAC_RELEASE_URL = f"{_RELEASES}/tag/{MAC_BETA_TAG}"
