# macos/

The native macOS app for LM Warden: a menu-bar app that starts and supervises
the LM Warden server, which runs models with MLX on Apple silicon. It is in
development; no signed, notarized release exists yet.

`swift build` and `swift test` here build and test the Swift package. The
full app bundle, with its embedded Python, is built from the repository root
with `make macos-bundle-python`; the "On a Mac (preview)" section of
[documents/INSTALL.md](../documents/INSTALL.md) explains it. It also builds
the web UI as a static export from the committed `frontend/` (it needs
`frontend/node_modules`: run `npm ci` in `frontend/` first) into
`Contents/Resources/webui`, which the app's server serves at `/ui`.
`MACOS_WEBUI=0` leaves the UI out for a faster local build without node.

## Build and test

```sh
swift build
swift test
```

`swift run LMWarden` is a smoke test that prints the app version
(`dev` until the package is wrapped in a real `LMWarden.app` bundle).

## Layout

- `Sources/LMWarden/` — the executable target (binary `LMWarden`, copied into
  `LM Warden.app/Contents/MacOS` by `make macos-app`).
- `Tests/LMWardenTests/` — XCTest target.

## Updates

The app updates itself with Sparkle, but only a bundle whose Info.plist has
both `SUFeedURL` and `SUPublicEDKey` (written by `write-info-plist.sh` from
`LMW_SPARKLE_FEED_URL` / `LMW_SPARKLE_PUBLIC_ED_KEY`). A dev build has
neither: Check for Updates… is disabled and the About window shows no update
notice. Sparkle shows no window of its own: Check for Updates… checks in the
background and opens the About window, where a found update appears as the
notice with Install and Relaunch and Release Notes (opened in the browser).
With **Check for updates automatically** on (the default; it also
turns on Sparkle's automatic download), a newer release is downloaded in the
background and installs when you quit, or at once with **Install and
Relaunch**, which asks first if requests are in flight.
`"LM Warden.app/Contents/MacOS/LMWarden" --update-probe` checks the bundle's
feed once and prints `update-probe: available <version> <display>`, `none`,
`disabled` or `error …`, without downloading anything.

## Remote access

The app serves on `127.0.0.1:8080` (falling back to `8081`–`8099`), plain
HTTP, with **Share on Local Network** to bind `0.0.0.0`. HTTPS comes from a
tool in front of it, not from the app:
[documents/MACOS-REMOTE-ACCESS.md](../documents/MACOS-REMOTE-ACCESS.md)
covers the defaults, the LAN warning, staying signed in, and Caddy and
Tailscale recipes.
