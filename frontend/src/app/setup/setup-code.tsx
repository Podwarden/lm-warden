'use client';
// The first-run setup code (app/setup/bootstrap.py).
//
// A fresh install asks for a one-time code before the wizard does anything,
// so whoever reaches a new public install first cannot make themselves its
// admin. The server says whether it wants one (GET /api/setup/state:
// code_required) and where the operator finds it (code_source). The code is
// kept for this tab only (sessionStorage) and sent as X-Setup-Code on every
// setup POST; a refusal clears it and asks again.
import { useEffect, useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

const KEY = 'lmw.setup-code';
const NEEDED = 'lmw:setup-code-needed';
const REFUSALS = new Set(['setup code required', 'setup code invalid']);

export type CodeSource = 'env' | 'file' | null;

export function getSetupCode(): string | null {
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setSetupCode(code: string): void {
  try {
    window.sessionStorage.setItem(KEY, code);
  } catch {
    /* storage blocked: the code lives only until the next refusal */
  }
}

export function clearSetupCode(): void {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    /* nothing stored */
  }
}

/** A setup POST: JSON, with the setup code when the tab has one. A refusal
 * for the code clears it and tells the gate to ask again. */
export async function setupPost(url: string, body?: unknown): Promise<Response> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const code = getSetupCode();
  if (code) headers['X-Setup-Code'] = code;
  const r = await fetch(url, {
    method: 'POST',
    credentials: 'include',
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (r.status === 403) {
    const detail = await r.clone().json().then((b) => b?.detail, () => undefined);
    if (typeof detail === 'string' && REFUSALS.has(detail)) {
      clearSetupCode();
      window.dispatchEvent(new CustomEvent(NEEDED, { detail }));
    }
  }
  return r;
}

export function whereToFind(source: CodeSource): string {
  if (source === 'env') {
    return (
      'It was set when LM Warden was installed: install.sh printed it when it finished, and it is ' +
      'LM_SETUP_CODE in the .env file next to docker-compose.yml. On PodWarden Hub it is ' +
      "LM_SETUP_CODE in the deployment's settings."
    );
  }
  return (
    'LM Warden printed it in its log when it started (look for "Setup code"), and keeps it in ' +
    'the file setup-code in its data directory. With Docker: docker compose exec api cat /data/setup-code'
  );
}

/** Renders the code form while the server wants a code this tab does not
 * have; the wizard step otherwise. */
export function SetupCodeGate({
  required,
  source,
  children,
}: {
  required: boolean;
  source: CodeSource;
  children: React.ReactNode;
}) {
  const [have, setHave] = useState<boolean>(() => getSetupCode() !== null);
  const [value, setValue] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    function onNeeded(e: Event) {
      const detail = (e as CustomEvent<string>).detail;
      setHave(false);
      setError(
        detail === 'setup code invalid'
          ? 'That setup code was not accepted. Check it and enter it again.'
          : 'This step needs the setup code.',
      );
    }
    window.addEventListener(NEEDED, onNeeded);
    return () => window.removeEventListener(NEEDED, onNeeded);
  }, []);

  if (!required || have) return <>{children}</>;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const code = value.trim();
    if (!code) {
      setError('Enter the setup code.');
      return;
    }
    setSetupCode(code);
    setError(null);
    setHave(true);
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <h1 className="text-2xl font-semibold">Enter the setup code</h1>
      <p className="text-sm text-slate-400">
        This is a new LM Warden. So that nobody else who finds it can make themselves its
        administrator, setting it up needs the one-time setup code.
      </p>
      <p className="text-sm text-slate-400" id="setup-code-where">{whereToFind(source)}</p>
      <label className="block space-y-1">
        <span className="text-sm">Setup code</span>
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoComplete="off"
          spellCheck={false}
          aria-describedby="setup-code-where"
        />
      </label>
      {error && <p className="text-red-500 text-sm" role="alert">{error}</p>}
      <Button type="submit">Continue</Button>
    </form>
  );
}
