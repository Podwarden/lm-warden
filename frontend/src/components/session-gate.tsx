'use client';
// ---------------------------------------------------------------------------
// SessionGate — don't paint the app until we know who is looking at it.
// ---------------------------------------------------------------------------
//
// Before this gate, a signed-out visitor opening /ui/ saw the Models page —
// heading, skeleton cards, the lot — for the length of one round trip, and was
// then bounced to /ui/login. The first impression of the product was a page
// that flashed empty and threw the visitor out; "it's broken" is a fair
// reading of that.
//
// The cause is ordering, not auth: the shell renders immediately, and the
// session is only discovered later by whichever SWR fetcher gets a 401 first.
// So resolve it up front instead. `ensureSession()` performs the one
// `POST /api/auth/refresh` that `authFetch`'s eager-refresh guard was going to
// make anyway before its first request, and `refresh()` de-dupes concurrent
// callers — so this costs no extra network, it just moves a decision earlier.
//
// Three outcomes:
//   'ok'        — render the shell.
//   'transient' — render the shell anyway. The backend is restarting or the
//                 link flapped; the cookie was never rejected. Wedging the
//                 whole UI behind a spinner during a backend bounce would be
//                 a worse failure than the one being fixed, and each page's
//                 own error handling is already written for this.
//   'invalid'   — the session really is dead. Redirect and stay closed, so
//                 nothing behind the gate ever paints.
//
// /login and the /setup/* wizard render unconditionally: they are the pages
// you reach WITHOUT a session, and gating them would deadlock the product on
// first run. The route list mirrors `isUnauthRoute` in nav-bar.tsx — the same
// question ("is this page for someone not signed in?") asked by the two
// components that both have to know the answer.
//
// The session forest has its own door for key holders (Task 8):
//   - /forest/login is public, like /login.
//   - /forest opens without an admin session when this tab holds an unexpired
//     forest token (no admin session check is made at all). An expired token,
//     or `?key=1` with none, goes straight to /forest/login with no request.
//     With neither token nor session, /forest goes to /forest/login, never to
//     the admin /login.

import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { ensureSession, getAccessToken, redirectToLogin } from '@/lib/auth-fetch';
import { forestKeyRequested, forestTokenState } from '@/lib/forest/token';

// Exact match, not `startsWith('/login')` — see nav-bar.tsx's #39 note: a
// hypothetical /login-help page must NOT inherit the unauthenticated
// treatment. The wizard genuinely owns its whole subtree.
const UNAUTH_EXACT_PATHS: ReadonlySet<string> = new Set([
  '/login',
  '/login/',
  '/setup',
  '/setup/',
  '/forest/login',
  '/forest/login/',
]);

export function isUnauthRoute(path: string): boolean {
  return UNAUTH_EXACT_PATHS.has(path) || path.startsWith('/setup/');
}

/** The forest page, which a forest token holder may open without an admin session. Exact match, as above. */
const FOREST_EXACT_PATHS: ReadonlySet<string> = new Set(['/forest', '/forest/']);
const FOREST_LOGIN = '/forest/login';

export function SessionGate({ children }: { children: React.ReactNode }) {
  // `usePathname` reports the path WITHOUT the '/ui' basePath, so these
  // compare against '/login', not '/ui/login'.
  const path = usePathname() ?? '';
  const router = useRouter();
  const openToEveryone = isUnauthRoute(path);
  const forestPage = FOREST_EXACT_PATHS.has(path);
  // What opened the gate. An admin session opens it for every page, for the
  // life of the layout. A forest token opens it for the one forest path it
  // was checked on, and only until the path changes: any client navigation
  // (to an admin page, or out to /forest/login and back) checks again, so a
  // key holder's token never paints an admin page and a cleared token never
  // reopens /forest.
  const [opened, setOpened] = useState<
    { by: 'session' } | { by: 'forest-token'; path: string } | null
  >(null);
  const resolved =
    opened !== null &&
    (opened.by === 'session' || (forestPage && opened.path === path));

  useEffect(() => {
    if (opened?.by === 'forest-token' && opened.path !== path) setOpened(null);
  }, [opened, path]);

  useEffect(() => {
    if (openToEveryone || resolved) return;
    if (forestPage) {
      // forestTokenState() clears an expired token as it reads it.
      const held = forestTokenState();
      if (held === 'valid') {
        setOpened({ by: 'forest-token', path });
        return;
      }
      if (held === 'expired' || forestKeyRequested()) {
        router.replace(FOREST_LOGIN);
        return;
      }
    }
    // A token already in memory means a client-side navigation, not a fresh
    // load; there is nothing to wait for.
    if (getAccessToken() !== null) {
      setOpened({ by: 'session' });
      return;
    }
    let cancelled = false;
    void ensureSession().then((check) => {
      if (cancelled) return;
      if (check === 'invalid') {
        // Stay closed. The navigation is what ends this render.
        if (forestPage) router.replace(FOREST_LOGIN);
        else redirectToLogin();
        return;
      }
      setOpened({ by: 'session' });
    });
    return () => {
      cancelled = true;
    };
  }, [openToEveryone, forestPage, resolved, path, router]);

  if (openToEveryone || resolved) return <>{children}</>;

  return (
    <div
      data-testid="session-gate-pending"
      className="flex items-center justify-center py-24"
      role="status"
      aria-label="Checking your session"
    >
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-500" />
    </div>
  );
}
