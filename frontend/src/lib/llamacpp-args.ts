// llama.cpp flags that get a real control, stored inside `extra_args`.
//
// WHY extra_args AND NOT NEW COLUMNS. Three of llama.cpp's most-used dials --
// flash attention and the two KV cache quantisation types -- had no first-class
// control, so the only way to set them was to hand-type into Extra args. They
// are also, deliberately, not emitted by the argv builder: app/runtime/backends/
// llamacpp/args.py states that they are "real dials with no column, and
// inventing a policy for a mainline default is what spec §2 forbids". That
// reasoning is still right. What was wrong was concluding that a knob with no
// column cannot have a control.
//
// So the control writes the flag into `extra_args`, which is exactly where a
// hand-typed one would go and exactly what the builder already appends last.
// The consequences are all the ones we want:
//
//   * no migration, and no schema widening per flag as more get exposed;
//   * the launch command for an unset control is BYTE-IDENTICAL to today's --
//     "unset" means the flag is absent, not a default written out;
//   * precedence needs no new rule. These ARE extra_args, appended last, so
//     they win over the builder's own flags exactly as before.
//
// AND NO DUPLICATION, which is the part that needs code rather than argument.
// If a control appended `--flash-attn on` while `--flash-attn off` sat in the
// free-text box, llama-server would receive both and honour the last. So the
// managed flags are EXTRACTED from extra_args on load: the control shows them
// and the free-text box shows only what is left. There is exactly one place
// each managed flag can live, so the two cannot disagree. An operator who types
// `--flash-attn on` into the box finds it in the control on the next load --
// moved, never duplicated, never silently dropped.

/** A llama.cpp flag this UI owns, plus how its value is spelled. */
export interface ManagedFlag {
  /** UI field key. Also the key in `BACKEND_ONLY_FIELDS.llamacpp`. */
  key: string;
  /** Canonical long spelling emitted into extra_args. */
  flag: string;
  /** Every spelling accepted when reading extra_args back, aliases included. */
  aliases: string[];
  /** Allowed values. Empty selection means "omit the flag". */
  values: readonly string[];
  label: string;
  hint: string;
}

// Values verified against `/opt/llamacpp/llama-server --help` in the shipped
// image (b10731, 0eadefebd) rather than from upstream docs -- several spellings
// have changed there and the binary is the authority.
export const LLAMACPP_MANAGED_FLAGS: readonly ManagedFlag[] = [
  {
    key: "flash_attn",
    flag: "--flash-attn",
    // `-fa` is the short form. Both are read; only the long form is written.
    aliases: ["--flash-attn", "-fa"],
    // NOT a bare boolean in this build: `-fa, --flash-attn [on|off|auto]`,
    // defaulting to `auto`. A UI that emitted a valueless `--flash-attn`
    // would be relying on an optional-argument form that upstream has already
    // changed once.
    values: ["on", "off", "auto"],
    label: "Flash attention",
    hint:
      "llama.cpp only. Leave unset to let the engine decide (its own default is 'auto', which resolves at load time against the card). 'on' forces it, which matters on this hardware; 'off' is the escape hatch when a model or a quantisation trips over it.",
  },
  {
    key: "cache_type_k",
    flag: "--cache-type-k",
    aliases: ["--cache-type-k", "-ctk"],
    values: ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    label: "KV cache type (K)",
    hint:
      "llama.cpp only. Quantising the K cache is the main lever for fitting a longer context on a small card -- q8_0 roughly halves its memory against the f16 default. Unset keeps the engine default (f16).",
  },
  {
    key: "cache_type_v",
    flag: "--cache-type-v",
    aliases: ["--cache-type-v", "-ctv"],
    values: ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    label: "KV cache type (V)",
    hint:
      "llama.cpp only. The V half of the same lever. Set both to see the full saving; the two are independent flags and quantising only one is a legitimate, if unusual, choice.",
  },
] as const;

export const LLAMACPP_MANAGED_KEYS: readonly string[] =
  LLAMACPP_MANAGED_FLAGS.map((f) => f.key);

/** Values of the managed flags, keyed by field key. Absent = flag not present. */
export type ManagedArgValues = Record<string, string | undefined>;

function flagFor(key: string): ManagedFlag | undefined {
  return LLAMACPP_MANAGED_FLAGS.find((f) => f.key === key);
}

/**
 * Split `extraArgs` into the flags this UI owns and everything else.
 *
 * Handles both `--flag value` (two tokens) and `--flag=value` (one), because
 * an operator who typed either into the box must find it in the control rather
 * than see it silently survive as free text and then be duplicated.
 *
 * A LAST-ONE-WINS read, matching what llama-server itself does with a repeated
 * flag: if the box somehow holds two, the control shows the one that would
 * actually take effect, and saving collapses them to one.
 *
 * An unrecognised VALUE is left in `rest` untouched. The control offers a fixed
 * list, so adopting a value it cannot display would mean either dropping the
 * operator's setting or rendering a select with no matching option — both
 * worse than leaving a flag we do not understand exactly where it was.
 */
export function splitManagedArgs(extraArgs: readonly string[]): {
  managed: ManagedArgValues;
  rest: string[];
} {
  const managed: ManagedArgValues = {};
  const rest: string[] = [];
  for (let i = 0; i < extraArgs.length; i += 1) {
    const tok = extraArgs[i];
    const eq = tok.indexOf("=");
    const name = eq === -1 ? tok : tok.slice(0, eq);
    const spec = LLAMACPP_MANAGED_FLAGS.find((f) => f.aliases.includes(name));
    if (!spec) {
      rest.push(tok);
      continue;
    }
    const inlineValue = eq === -1 ? undefined : tok.slice(eq + 1);
    const value = inlineValue ?? extraArgs[i + 1];
    if (value === undefined || !spec.values.includes(value)) {
      // Unknown value, or a flag with nothing after it. Leave it alone.
      rest.push(tok);
      continue;
    }
    managed[spec.key] = value;
    if (inlineValue === undefined) i += 1; // consume the value token
  }
  return { managed, rest };
}

/**
 * Recombine free-text args with the managed values into one `extra_args`.
 *
 * Managed flags are appended AFTER `rest` and in a fixed order. Order within
 * extra_args is not load-bearing for these flags — each appears once and none
 * of them is positional — but a stable order keeps the saved value byte-stable,
 * so a save with no edits produces no diff and the dirty-marker on the page
 * stays honest.
 *
 * An empty/absent value OMITS the flag entirely. That is what makes "unset"
 * reachable: writing out the engine's own default would change the launch
 * command for every existing row, and a default that cannot be returned to is
 * the settings bug this codebase keeps finding.
 */
export function mergeManagedArgs(
  rest: readonly string[],
  managed: ManagedArgValues,
): string[] {
  const out = [...rest];
  for (const spec of LLAMACPP_MANAGED_FLAGS) {
    const v = managed[spec.key];
    if (v === undefined || v === "") continue;
    if (!spec.values.includes(v)) continue;
    out.push(spec.flag, v);
  }
  return out;
}

/** Set one managed value on an `extra_args` list, returning the new list. */
export function setManagedArg(
  extraArgs: readonly string[],
  key: string,
  value: string | undefined,
): string[] {
  const spec = flagFor(key);
  if (!spec) return [...extraArgs];
  const { managed, rest } = splitManagedArgs(extraArgs);
  if (value === undefined || value === "") delete managed[key];
  else managed[key] = value;
  return mergeManagedArgs(rest, managed);
}
