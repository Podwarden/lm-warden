// The rule for a NEW admin password -- the setup wizard's admin step and the
// Settings password change. Mirrors app/auth/passwords.py; keep them in step.
//
// At least 12 characters, at most 72 bytes (UTF-8). The ceiling is bcrypt's:
// the server refuses a longer password rather than silently truncating it.
// Login never applies the rule, so a password set under the old 6-character
// minimum keeps working until it is changed.

export const PASSWORD_MIN_CHARS = 12;
export const PASSWORD_MAX_BYTES = 72;

export const PASSWORD_RULE_HINT = `At least ${PASSWORD_MIN_CHARS} characters (at most ${PASSWORD_MAX_BYTES} bytes).`;

/** Why `password` may not be set, or null when it may. */
export function newPasswordProblem(password: string): string | null {
  // Count code points, as Python's len() does -- not UTF-16 units.
  if ([...password].length < PASSWORD_MIN_CHARS) {
    return `Password must be at least ${PASSWORD_MIN_CHARS} characters`;
  }
  if (new TextEncoder().encode(password).length > PASSWORD_MAX_BYTES) {
    return `Password must be at most ${PASSWORD_MAX_BYTES} bytes`;
  }
  return null;
}
