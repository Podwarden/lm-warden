// Copy for the unified login and the key holder's area (/key/*). Kept apart from the pages so tests can name it.

export const K = {
  login: {
    tabLogin: "Login",
    tabKey: "API key",
    username: "Username",
    password: "Password",
    submit: "Sign in",
    keyLabel: "Your API key",
    keySubmit: "Open my key",
    keyFine:
      "For people who were given a key. The key is checked once and not stored in your browser; the sign-in ends when it times out, when you sign out, or as soon as the key is rotated or revoked.",
    keyErrors: {
      401: "Key not accepted",
      403: "Key is paused",
      429: "Too many attempts. Wait and try again.",
    } as Record<number, string>,
    keyGeneric: "Could not sign in. Try again in a moment.",
    keyStorage: "This browser blocks session storage, so the sign-in cannot be kept. Allow it for this site and try again.",
  },
  shell: {
    nav: [
      { href: "/key", label: "Overview" },
      { href: "/key/stats", label: "Stats" },
      { href: "/key/prompt-lens", label: "Prompt Lens" },
      { href: "/sessions", label: "Sessions" },
      { href: "/key/guard-log", label: "Sensitive data log" },
      { href: "/key/settings", label: "Settings" },
      { href: "/key/manage", label: "Manage key" },
    ],
    signOut: "Sign out",
    created: "created",
    expires: "expires",
    never: "never expires",
    status: { active: "active", paused: "paused", revoked: "revoked", expired: "expired" },
    loadError: "Could not load this key. Try again in a moment.",
  },
  stats: {
    privacy:
      "This page shows only requests made with your key. It shows nothing fleet-wide: no other keys, users, IP addresses, sessions, prompts, installs or server load, and no cache figure tells you whether anyone else sent the same text.",
    /** CAPTURE_NOTE of app/tokens/routes_me_stats.py, word for word. */
    capture:
      "Captures are kept briefly in server memory, not on disk, and the platform's admins can see them too. You only ever see captures of requests made with your key.",
    /** CAPTURE_NOTE_KEPT: the same for a key an admin set to keep captures. */
    captureKept:
      "Captures are kept briefly in server memory, and the platform's admins can see them too. An admin set this key to keep captures: the last 50 captures are kept on disk for this key. You only ever see captures of requests made with your key.",
  },
  overview: {
    requests: "requests, last 24 h",
    prompt: "prompt tokens",
    fromCache: (pct: number) => `${pct}% from cache`,
    generated: "generated tokens",
    hidden: "values hidden by the guard, last 7 days",
    perModel: "By model, last 24 h",
    cols: ["Model", "Requests", "Prompt", "From cache"],
    noTraffic: "No requests from this key in the last 24 hours.",
  },
  settings: {
    cols: ["Setting", "Value", ""],
    byAdmin: "set by the admin",
    models: "Models this key may use",
    allModels: "All models",
    priority: "Priority",
    thinking: "Thinking",
    thinkingValues: { client: "As the client asks", off: "Off for this key", on: "On for this key" } as Record<string, string>,
    relay: "Relay to Anthropic",
    yes: "Allowed",
    no: "Not allowed",
    guard: "Sensitive data guard",
    guardOff: "Off",
    guardModes: { sanitize: "Sanitize", block: "Block", detect: "Detect only", off: "Off" } as Record<string, string>,
    fromPlatform: "from the platform settings",
    ownRules: (n: number) => `${n} custom ${n === 1 ? "rule" : "rules"}`,
    nameLabel: "Name shown on this key",
    save: "Save",
    saved: "Saved",
    saveError: "Could not save the name.",
  },
  manage: {
    rotateTitle: "Rotate",
    rotateBody:
      "Get a new key and retire this one. Keep the old key working for a while, so you can update your tools without downtime.",
    graceLabel: "Old key keeps working for",
    grace: [
      { value: 1, label: "1 hour" },
      { value: 24, label: "24 hours" },
      { value: 0, label: "no grace (stop now)" },
    ],
    rotate: "Rotate key",
    rotating: "Rotating…",
    newKey: "Your new key, shown once. Copy it now:",
    copy: "Copy",
    copied: "Copied",
    newKeyFine: "Same models, limits and settings as the old key. You are now signed in with the new key.",
    rotateError: "Could not rotate the key.",
    revokeTitle: "Revoke",
    revokeBody:
      "Stops this key at once, for everyone who uses it. This cannot be undone; only an admin can issue a new key.",
    confirmLabel: "Type the key's name to confirm",
    revoke: "Revoke key",
    revokeError: "Could not revoke the key.",
  },
  guardLog: {
    title: "Sensitive data log, last 7 days",
    intro:
      "What the guard did with this key's requests. Values are never stored: each row shows the kind of value, where it was and its length.",
    cols: ["When", "Action", "Kind", "Where", "Length", "Value id"],
    totals: (sanitized: number, blocked: number, distinct: number) =>
      `${sanitized} hidden · ${blocked} blocked · ${distinct} distinct values`,
    empty: "Nothing caught in the last 7 days.",
    actions: { sanitized: "Hidden", blocked: "Blocked", restored: "Restored", error: "Error" } as Record<string, string>,
  },
} as const;
