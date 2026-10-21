# Reaching the Mac app from other machines

The LM Warden Mac app serves the dashboard and the API on this Mac only. This
page covers what it listens on, what **Share on Local Network** changes, and
two ways to put HTTPS in front of it: Caddy, and Tailscale.

The app has no built-in TLS and no certificate settings, and that is by design.
It speaks plain HTTP. When you need HTTPS, another tool you already trust
provides it.

## What the app listens on

- **Address.** `127.0.0.1`, which is loopback, so only programs on this Mac can
  connect.
- **Port.** `8080`. If something else already holds 8080 when the server
  starts, the app takes the first free port from `8081` to `8099`. It checks
  again every time the server starts, so the port can change after a restart
  when 8080 has been freed or taken in the meantime.
- **Which port it is using now.** The menu-bar header says
  `Serving on this Mac only · 127.0.0.1:8080`, and **Copy API Address**
  copies the same address. The dashboard is at `/ui/`, the OpenAI-compatible
  API at `/v1/` and the health check at `/healthz`. All three use the same
  port.

There is no setting that fixes the port. If a proxy below points at `8080` and
the app had to fall back to `8081`, the proxy answers `502`. Free port 8080
and restart the server (**Stop Server**, then **Start Server**), or point the
proxy at the port the menu shows.

## Share on Local Network

**Share on Local Network** in the menu-bar menu is off by default. Turning it
on restarts the server bound to `0.0.0.0`, so every network interface on this
Mac accepts connections on the same port. The header then reads
`Serving on your network · 192.168.1.20:8080`, and **Copy API Address** copies
that LAN address. Turning it off restarts the server on `127.0.0.1` again.

> **Warning: it is plain HTTP.** With sharing on, everything between another
> machine and this Mac crosses the network unencrypted: your dashboard
> password when you sign in, the session cookie, every API key a client sends,
> and every prompt and completion. Anyone on the same Wi-Fi or LAN segment who
> can capture traffic can read and reuse them. Use it on a network you trust,
> such as your home network, and never on public or shared Wi-Fi. Anything
> beyond that needs one of the HTTPS setups below.

Sharing on the LAN does not open anything to the internet by itself. Your
router still decides that. Do not forward the app's port from your router: if
you want access from outside your home, use Tailscale or Caddy below.

Sign-in works over plain LAN HTTP. The warden notices that the browser used
`http://` and leaves the `Secure` flag off its session cookies, so a page
reload keeps you signed in.

## The Mac has to stay signed in and awake

The server is a child process of the menu-bar app. It is not a background
system service.

- **Quitting the app stops the server.** So does logging out of macOS, and so
  does a restart until someone signs in again.
- **Open at Login** in the menu starts the app when you sign in. For a Mac
  that must come back by itself after a power cut or an update, also turn on
  automatic login (**System Settings → Users & Groups → Automatically log in
  as**). macOS turns that option off while FileVault is on. On a FileVault Mac,
  someone has to unlock it after every restart.
- **Sleep.** The app keeps the Mac from idle-sleeping while a model is loaded.
  It does not stop a laptop from sleeping when you close the lid, and with no
  model loaded the Mac sleeps on its usual schedule. On a Mac you use as a
  server, turn on **System Settings → Energy → Prevent automatic sleeping when
  the display is off** (on a laptop, under **Battery → Options**).

## HTTPS with Caddy

[Caddy](https://caddyserver.com) is a web server that obtains and renews
certificates by itself. Run it on the same Mac and leave **Share on Local
Network off**. Caddy then is the only thing other machines can reach, and it
forwards to the app over loopback. The cleartext port never leaves the Mac.

Install it with Homebrew:

```sh
brew install caddy
```

Homebrew's Caddy reads `$(brew --prefix)/etc/Caddyfile`, which is
`/opt/homebrew/etc/Caddyfile` on Apple silicon.

### A real domain, with a public certificate

You need a domain name whose DNS record points at your home's public address,
with ports 80 and 443 forwarded from your router to this Mac. Caddy uses them
to prove to Let's Encrypt that it owns the name.

```caddyfile
{
    servers {
        # Caddy 2.11 and newer end a streamed response after 60 s of
        # request-body idle. Long Claude Code and chat streams need it off.
        timeouts {
            read_body_idle -1s
        }
    }
}

llm.example.com {
    reverse_proxy 127.0.0.1:8080 {
        # Send each streamed token as it arrives instead of buffering.
        flush_interval -1
    }
}
```

Then start Caddy and keep it running across restarts:

```sh
brew services start caddy
```

Open `https://llm.example.com/ui/`. Clients use `https://llm.example.com/v1`
as their base URL.

### A LAN name, with Caddy's own certificate

On a home network with no public domain, `tls internal` makes Caddy sign the
certificate with its own local certificate authority:

```caddyfile
{
    servers {
        timeouts {
            read_body_idle -1s
        }
    }
}

https://my-mac.local, https://192.168.1.20 {
    tls internal
    reverse_proxy 127.0.0.1:8080 {
        flush_interval -1
    }
}
```

`my-mac.local` is this Mac's Bonjour name (**System Settings → General →
Sharing → Local hostname**). Other devices trust the certificate only after
you install Caddy's root certificate on them. Until then the browser shows a
certificate warning. `caddy trust` installs it on this Mac. For other devices,
copy `root.crt` from
`~/Library/Application Support/Caddy/pki/authorities/local/` and install it on
each one as a trusted root.

### Headers: leave Caddy's defaults alone

The dashboard signs you in with a short-lived token and renews it through a
call that checks the browser's `Origin` header. The renewal is accepted only
when that origin matches the address the browser actually used. Behind a
proxy, the warden learns that address from the request:

- the scheme from `X-Forwarded-Proto` (`https`)
- the host from `X-Forwarded-Host`, or `Host` when that header is missing

Caddy's `reverse_proxy` already sends both and passes the original `Host`
through, so the recipes above need no `header_up` lines. Do not remove or
overwrite `X-Forwarded-Proto` or `X-Forwarded-Host`. Without
`X-Forwarded-Proto` the warden sees `http://llm.example.com`, the browser
sends `https://llm.example.com`, and every page reload signs you out with
`403 origin mismatch`. If you also rewrite `Host` (for example
`header_up Host {upstream_hostport}`) and drop `X-Forwarded-Host`, it sees
`http://127.0.0.1:8080` instead, with the same result.

You cannot set environment variables in the Mac app, and nothing above needs
them. When no public URL is configured, the app works out its origin from each
request. That covers Caddy, Tailscale and plain LAN access at the same time.

### Caddy on another machine

If Caddy runs on a different box, for example a home server, that box has to
reach this Mac over the LAN, so turn **Share on Local Network on** and point
Caddy at the Mac's LAN address:

```caddyfile
llm.example.com {
    reverse_proxy 192.168.1.20:8080 {
        flush_interval -1
    }
}
```

Sign-in works the same way: the origin check reads `X-Forwarded-Proto` and
`X-Forwarded-Host` from the request whichever machine sent them. One thing
differs. The app's web server believes `X-Forwarded-Proto` only from a proxy
on this Mac (`127.0.0.1`), so behind a Caddy on another machine it does not
know the browser used HTTPS and sets the session cookies without the `Secure`
flag. They still work over `https://`.

The hop from Caddy to the Mac is plain HTTP, so the warning in
[Share on Local Network](#share-on-local-network) applies to that hop. And
with sharing on, any device on the LAN can also skip Caddy and open
`http://192.168.1.20:8080` directly. Keep both machines on a network you
trust.

## HTTPS with Tailscale

[Tailscale](https://tailscale.com) gives this Mac a name on your private
tailnet and a real certificate for it. Nothing is forwarded on your router, and
**Share on Local Network** stays off.

Install Tailscale on this Mac and on each device that needs access, and sign
them all in to the same tailnet. Turn on **MagicDNS** and **HTTPS
Certificates** in the Tailscale admin console under **DNS**. The command-line
tool is `tailscale`. In the standalone Mac app it is also at
`/Applications/Tailscale.app/Contents/MacOS/Tailscale`.

### Your own devices only: `tailscale serve`

```sh
tailscale serve --bg 8080
```

This publishes `https://my-mac.tail1234.ts.net/` to devices on your tailnet
and forwards it to `127.0.0.1:8080`. `--bg` keeps the setting after the
command exits and after a restart. `tailscale serve status` shows what is
being served, and `tailscale serve reset` stops it.

### Anyone on the internet: `tailscale funnel`

```sh
tailscale funnel --bg 8080
```

> **Warning: this puts the warden on the public internet.** Funnel publishes
> the same address to everyone, not just your tailnet. Anyone who finds the
> URL reaches your sign-in page, the health check and the `/v1` API. The
> dashboard and the API still need a dashboard password or an API key, but
> the sign-in page is now open to password guessing from anywhere. If you
> only want your own devices to connect, use `tailscale serve` above, not
> Funnel. If you do use Funnel, use a long password and give each client its
> own API key. `tailscale funnel reset` stops it.

Sign-in works through both. Tailscale passes the original `Host` through and
sets `X-Forwarded-Host`, and `X-Forwarded-Proto: https`, which is what the
origin check above needs.

## Addresses shown in the dashboard

The **Connect** page builds its copy-paste snippets from the address the
browser used. Through Caddy or Tailscale it shows the `https://` URL. To pin
one address for every snippet, set it in **Settings → Networking → Public
access**.
