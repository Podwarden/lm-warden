'use client';
// Nav shell — matches PodWarden Core / Hub design language.
//
// - left brand block: Shield + "LM Warden" in emerald
// - right cluster: ThemeSwitcher + emerald hamburger button
// - dropdown menu with user-style menu rows + version footer
// Hidden entirely on /login and /setup so the unauthenticated flows render
// without the chrome.
//
// V1 deliberately omits the System Messages / NotificationBell row from
// PodWarden Core; that schema + API lands in a later MR.
//
// The Benchmarks entry was removed in epic/overhaul (S1) along with the
// /benchmarks page itself; the post-overhaul flow surfaces benchmark-like
// guidance through an opt-in "Suggest values" affordance on the settings
// page (see plan §S3+S4).
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import useSWR from 'swr';
import {
  Activity,
  Box,
  HardDrive,
  KeyRound,
  Layers,
  LogOut,
  Menu,
  MessagesSquare,
  Settings,
  Shield,
} from 'lucide-react';
import { authFetch, authFetchJSON } from '@/lib/auth-fetch';
import { HeaderMetrics } from './header-metrics';
import { ThemeSwitcher } from './theme-switcher';

// Routes where the nav chrome (and the header-metrics widget that hangs
// off it) must NOT render. Exact match — previously this was a
// ``startsWith`` guard which silently swallowed any path beginning with
// /login or /setup (e.g. a hypothetical /login-help marketing page would
// have rendered as unauthenticated chrome). Trailing-slash variants are
// listed explicitly because Next.js can canonicalise either way
// depending on routing config, and we don't want one form to leak GPU
// info while the other suppresses it. Setup wizard subroutes
// (/setup/welcome, /setup/users) keep their hide via the
// ``startsWith('/setup/')`` check below — that path tree is fully
// owned by the wizard and the widget would be visually distracting
// during onboarding.
const UNAUTH_EXACT_PATHS: ReadonlySet<string> = new Set([
  '/login',
  '/login/',
  '/setup',
  '/setup/',
]);

function isUnauthRoute(path: string): boolean {
  // The wizard owns the entire /setup/* subtree (multiple steps).
  // /login has no subroutes today, so it gets the exact-match
  // treatment. If we ever add /login/2fa or similar we'll bring it
  // under the same prefix carve-out — for now the tighter check
  // closes #39 by rejecting /login-help, /login.json, etc.
  return UNAUTH_EXACT_PATHS.has(path) || path.startsWith('/setup/');
}

interface MenuItem {
  href: string;
  label: string;
  Icon: typeof Box;
}

const MENU_ITEMS: readonly MenuItem[] = [
  { href: '/models', label: 'Models', Icon: Box },
  { href: '/templates', label: 'Templates', Icon: Layers },
  { href: '/chat2', label: 'Chat', Icon: MessagesSquare },
  { href: '/tokens', label: 'Tokens', Icon: KeyRound },
  { href: '/stats', label: 'Stats', Icon: Activity },
  { href: '/cache', label: 'Cache', Icon: HardDrive },
  { href: '/settings', label: 'Settings', Icon: Settings },
];

interface VersionResponse {
  version: string;
  sha: string;
}

// Footer text format — "{version} · {sha:0:7}" with a graceful fallback.
// The spec writes it as "v<version> — sha <short(7)>"; we match the punctuation
// style used by PodWarden Core's nav (centred dot, no "sha" prefix).
//
// #83 fix: do NOT prefix an extra "v". The backend already returns a CalVer
// tag including the leading "v" when present (e.g. "v2026.05.19.1"), so
// adding our own produced "vv2026.05.19.1" in the footer. We now emit
// `version` verbatim — the fallback string keeps the literal "v?" so the
// shape of "could not load" stays unambiguous (a bare "?" would read as
// "missing sha" instead of "missing everything").
function formatVersion(data: VersionResponse | undefined, error: unknown): string {
  if (error || !data) return 'v? · ?';
  const sha = (data.sha ?? '').slice(0, 7) || '?';
  const version = data.version || '?';
  return `${version} · ${sha}`;
}

export function NavBar() {
  const path = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Gate the version-probe SWR call on unauthenticated routes — see
  // v2026.05.15.4 fix below.
  //
  // The component returns `null` at the bottom of the function when the
  // pathname starts with /login or /setup, but React hooks (including
  // useSWR) execute regardless of any conditional return that comes
  // after them. Pre-fix, that meant `useSWR('/api/version', ...)` fired
  // unconditionally — even on /login, where the operator may not be
  // authenticated yet — producing this loop:
  //
  //   1. Component mounts on /login (NavBar lives in the root layout).
  //   2. SWR fires GET /api/version → 401 (no session yet).
  //   3. authFetch's refresh attempt also 401s.
  //   4. The 401 fallback calls window.location.replace('/login') — but
  //      we're already on /login, so it triggers a full-page reload of
  //      the same page.
  //   5. The new page imports auth-fetch.ts fresh, the in-flight flag
  //      resets, and step 1 repeats. Reload-loop.
  //
  // Passing `null` as the SWR key makes useSWR a no-op (no fetcher
  // invocation, no cache write) while preserving hook order. The
  // `defense in depth` companion fix in auth-fetch.ts handles any peer
  // component that regresses similarly.
  const versionKey = isUnauthRoute(path) ? null : '/api/version';

  // Version footer — single fetch on mount, no polling. The endpoint reads
  // build-time env vars baked into the image, so the value is immutable
  // for the lifetime of the running container. SWR will revalidate on
  // window focus, which is fine — costs one cached-response round-trip.
  const { data: versionData, error: versionError } = useSWR<VersionResponse>(
    versionKey,
    authFetchJSON,
    {
      // Don't surface in the React error boundary — the fallback is part
      // of the contract.
      shouldRetryOnError: false,
    },
  );

  // Close the dropdown when the route changes — without this the menu
  // stays open after a click navigates, which feels stale.
  useEffect(() => {
    setOpen(false);
  }, [path]);

  // Outside-click close. Pointer event (vs. mousedown) so the close fires
  // before the next interaction's click target loses focus. Touch screens
  // also dispatch pointer events first.
  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: PointerEvent) {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open]);

  // Escape-to-close — keyboard parity with the click-outside above.
  useEffect(() => {
    if (!open) return;
    function onKeydown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', onKeydown);
    return () => document.removeEventListener('keydown', onKeydown);
  }, [open]);

  async function signOut() {
    setOpen(false);
    // Fire-and-forget — even if the server-side logout fails (e.g. session
    // already gone), we still want to land on /login. authFetch handles
    // CSRF; a 4xx here is acceptable.
    try {
      await authFetch('/api/auth/logout', { method: 'POST' });
    } catch {
      /* fall through to redirect */
    }
    router.replace('/login');
  }

  if (isUnauthRoute(path)) return null;

  const versionText = formatVersion(versionData, versionError);

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur relative z-20">
      <nav className="container mx-auto px-4 flex h-14 items-center justify-between">
        {/* Brand block — Shield + "LM Warden" in emerald. The product name
            is engine-neutral: it serves vLLM and llama.cpp, so the wordmark
            must not name one of them. Identity strings (the `vllm-warden`
            slug, image names, `VW_` env vars) are deliberately unchanged. */}
        <Link
          href="/models"
          className="flex items-center gap-2 text-lg font-semibold text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          <Shield className="h-5 w-5" aria-hidden="true" />
          <span>LM Warden</span>
        </Link>

        <div className="flex items-center gap-2">
          {/* Live VRAM% / GPU% / active-model instrument cluster.
              Hidden on small viewports (the cluster needs ~280px and we
              don't want to crowd the brand on phones). Reuses a single
              EventSource per tab — see lib/header-metrics-stream.ts. */}
          <HeaderMetrics />
          <ThemeSwitcher />
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              aria-label={open ? 'Close menu' : 'Open menu'}
              aria-expanded={open}
              aria-haspopup="menu"
              onClick={() => setOpen((v) => !v)}
              className="h-12 w-12 rounded-full bg-emerald-600 hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-400 flex items-center justify-center text-white transition-colors"
            >
              <Menu className="h-5 w-5" aria-hidden="true" />
            </button>
            {open && (
              <div
                role="menu"
                aria-label="Main menu"
                className="absolute right-0 mt-2 w-56 rounded-lg bg-slate-900 border border-slate-700 shadow-lg overflow-y-auto"
                style={{ maxHeight: 'calc(100vh - 6rem)' }}
              >
                <ul className="py-1">
                  {MENU_ITEMS.map(({ href, label, Icon }) => {
                    // Active highlighting — startsWith(href) catches sub-routes
                    // (e.g. /models/<id> still highlights "Models"). Use
                    // strict equality only for "/" to avoid every route
                    // lighting up the brand link.
                    const active =
                      path === href || path.startsWith(`${href}/`);
                    return (
                      <li key={href}>
                        <Link
                          href={href}
                          role="menuitem"
                          className={`flex items-center gap-3 px-4 py-2 text-sm transition-colors ${
                            active
                              ? 'text-emerald-400 bg-slate-800'
                              : 'text-slate-200 hover:bg-slate-800 hover:text-emerald-300'
                          }`}
                          aria-current={active ? 'page' : undefined}
                        >
                          <Icon className="h-4 w-4" aria-hidden="true" />
                          <span>{label}</span>
                        </Link>
                      </li>
                    );
                  })}
                  <li role="separator" aria-hidden="true">
                    <hr className="my-1 border-slate-700" />
                  </li>
                  <li>
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        void signOut();
                      }}
                      className="w-full flex items-center gap-3 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800 hover:text-emerald-300 transition-colors"
                    >
                      <LogOut className="h-4 w-4" aria-hidden="true" />
                      <span>Sign out</span>
                    </button>
                  </li>
                </ul>
                <div className="border-t border-slate-700 px-4 py-2 text-xs text-slate-500 font-mono">
                  {versionText}
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>
    </header>
  );
}
