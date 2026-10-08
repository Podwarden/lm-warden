/**
 * The @podwarden/chat-ui conformance kit, pointed at a live vllm-warden.
 *
 * `runConformance` only registers describe/it blocks — it talks to the
 * server exclusively through the `fetch` injected below, which is where
 * this backend's authentication lives: a JWT bearer token plus the
 * double-submit CSRF pair (`X-CSRF-Token` + the `vw_csrf_id` cookie it is
 * an HMAC of), exactly what `src/lib/auth-fetch.ts` attaches in the browser.
 *
 * Boot the server with `make conformance`; it exports the three values.
 */
import { runConformance } from '@podwarden/chat-ui/conformance';

const base = process.env.CHAT2_BASE_URL; // e.g. http://127.0.0.1:18080/api/chat2
const token = process.env.CHAT2_TOKEN;
const csrf = process.env.CHAT2_CSRF;
const csrfId = process.env.CHAT2_CSRF_ID;

if (!base || !token || !csrf || !csrfId) {
  throw new Error(
    'CHAT2_BASE_URL, CHAT2_TOKEN, CHAT2_CSRF and CHAT2_CSRF_ID are required (see `make conformance`)',
  );
}

// `new Headers(init?.headers)` rather than spreading a cast: `HeadersInit`
// is also `Headers` and `[string, string][]`, and spreading either of those
// into an object literal silently drops every header the kit set — the cast
// only silences the type error, it does not make the spread work.
const authed: typeof fetch = (input, init) => {
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${token}`);
  headers.set('X-CSRF-Token', csrf);
  headers.set('Cookie', `vw_csrf_id=${csrfId}`);
  return fetch(input, { ...init, headers });
};

runConformance({
  baseUrl: base,
  fetch: authed,
  // The fake upstream answers a short prompt in a few hundred milliseconds.
  // The abort / turn-in-flight / live-replay groups need a turn that is still
  // streaming a beat later, and the harness stretches its reply for any
  // prompt over 200 characters.
  fixtures: { makeLongPrompt: () => 'x'.repeat(300) },
});
