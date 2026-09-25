# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting:
[open a draft security advisory](https://github.com/Podwarden/lm-warden/security/advisories/new)
(the **Security** tab of the repository, then **Report a vulnerability**). Only
the maintainers can see it.

Please include:

- the LM Warden version (the release tag at the foot of the menu in the UI,
  or `GET /api/version` when logged in) and how it is deployed;
- what an attacker can do, and from where — unauthenticated network access,
  a holder of an API key, a logged-in admin, or local access to the host;
- steps to reproduce, or a proof of concept;
- any configuration that matters (reverse proxy, TLS, `VW_*` settings that
  differ from the defaults).

What happens next: we acknowledge the report, confirm or rule out the problem,
and agree a disclosure date with you. The fix ships in a normal release and is
called out in [CHANGELOG.md](CHANGELOG.md); with your permission, the advisory
credits you.

## Supported versions

Security fixes land on the latest release only. LM Warden releases are
date-versioned (`vYYYY.MM.DD.N`); upgrading is described in
[documents/OPERATING.md](documents/OPERATING.md).

## Scope

In scope: this repository — the control plane, the OpenAI-compatible proxy,
the web UI, the installer and the published container images as built from
this source.

Out of scope, and best reported upstream:

- vLLM itself — <https://github.com/vllm-project/vllm/security>
- llama.cpp itself — <https://github.com/ggml-org/llama.cpp/security>
- the behaviour of a model you chose to load.

Things that are by design rather than vulnerabilities: god mode and the content
log can show request content to an admin when an operator switches them on
(both are off by default — see
[Where request content can end up](documents/OPERATING.md#where-request-content-can-end-up)),
and the stack serves plain HTTP on its one port, expecting you to terminate TLS
in front of it.
