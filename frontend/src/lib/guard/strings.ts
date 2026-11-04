// Copy for the sensitive data guard's settings, try-it box and action log (one module, for a
// later i18n pass and so tests can name it).

export const G = {
  title: "Sensitive data",
  intro:
    "Stops secrets in requests from reaching a model: your own GPUs, and Anthropic when relayed. Values are never stored; the log keeps only which rule matched, where, the length and a fingerprint.",
  platformIntro:
    "The default for every key. A key can use it, switch the guard off for itself, or have its own settings (on the key's page).",
  save: "Save",
  saving: "Saving…",
  saved: "Saved",
  mode: {
    legend: "What happens when something matches",
    off: "Off",
    sanitize: "Sanitize",
    block: "Block, then sanitize",
    help:
      "Sanitize: each match is replaced with a placeholder like ⟦LMW-SECRET-7c41e09b2a⟧ before the model sees it, and the answer ends with a note listing what was replaced. Block, then sanitize: the first request carrying a new secret is answered by LM Warden instead of the model; if the same secret is sent again (it stays in the conversation), it is replaced so the session can go on.",
  },
  packs: {
    legend: "Built-in rules",
    hint: "Tuned to skip lockfiles, hashes and UUIDs",
    rules: (n: number) => `${n} rule${n === 1 ? "" : "s"}`,
    actionLabel: (pack: string) => `Action for ${pack}`,
    actions: [
      { value: "", label: "Default action" },
      { value: "sanitize", label: "Sanitize" },
      { value: "block", label: "Block, then sanitize" },
    ],
    showRules: "Rules",
  },
  rules: {
    legend: "Your rules",
    hint: "Patterns and fixed words, checked when you save",
    addPattern: "+ Pattern",
    addFixed: "+ Fixed words",
    idLabel: "Rule name",
    patternLabel: "Pattern",
    wordsLabel: "Words, one per line",
    remove: "Remove",
    kinds: { regex: "Pattern", fixed: "Fixed words" } as Record<string, string>,
  },
  restore: {
    label: "Put values back in answers",
    options: [
      { value: "all", label: "Everywhere" },
      { value: "text_only", label: "Text only (leave placeholders in tool calls)" },
      { value: "off", label: "Never" },
    ],
  },
  trailer: {
    label: "Note at the end of answers",
    options: [
      { value: "new_only", label: "When something new was replaced" },
      { value: "always", label: "Every turn with a placeholder" },
      { value: "off", label: "Never" },
    ],
  },
  fail: {
    label: "If the guard can't check a request",
    options: [
      { value: "closed", label: "Refuse the request" },
      { value: "open", label: "Send it unchecked" },
    ],
  },
  key: {
    legend: "This key",
    inherit: "Use the platform setting",
    off: "Off for this key",
    own: "Own settings for this key",
    platformOff: "The platform has no guard, so this key is not checked.",
    effective: "In effect:",
    customRules: (n: number) => `${n} custom rule${n === 1 ? "" : "s"}`,
  },
  tryIt: {
    title: "Try it",
    hint: "Nothing you paste here is stored",
    label: "Text to check",
    run: "Check",
    running: "Checking…",
    none: "Nothing matched.",
    matches: (n: number, ms: number) => `${n} match${n === 1 ? "" : "es"} · ${ms} ms`,
  },
  log: {
    title: "What the guard did",
    intro: "Last 7 days. Each row is one value in one request: the rule, where it was, its length and a fingerprint (the same fingerprint means the same value, nothing more).",
    all: "All actions",
    filterLabel: "Action",
    actions: { sanitized: "Sanitized", blocked: "Blocked", restored: "Restored", error: "Guard error" } as Record<string, string>,
    cols: ["When", "Action", "Rule", "Where", "Value"],
    empty: "Nothing in the last 7 days.",
    totals: (t: Record<string, number>) =>
      `${t.sanitized ?? 0} sanitized · ${t.blocked ?? 0} blocked · ${t.restored ?? 0} restored · ${t.distinct_values ?? 0} distinct values · ${t.error ?? 0} guard errors`,
    value: (len: number | null, fp: string | null) =>
      [len != null ? `${len} chars` : null, fp ? "fp" : null].filter(Boolean).join(" · "),
    relayed: "relayed",
  },
  loadError: "Could not load the guard settings.",
} as const;
