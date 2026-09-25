// NavBar — pins the contract:
//   - hidden on /login and /setup
//   - emerald brand "LM Warden" + Shield icon
//   - dropdown contains Models / Tokens / Stats / Settings
//     (Benchmarks entry removed in epic/overhaul S1 along with the page)
//   - version footer fetched from GET /api/version; fallback on error
//   - hamburger opens/closes the menu
// We don't pin the exact punctuation of the version string (the spec leaves
// "v<version> — sha <short>" vs. "v<version> · <short>" ambiguous); the
// test asserts the *data* — the digits/sha bytes appear.
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { SWRConfig } from 'swr';

// Stub the HeaderMetrics widget — the nav-bar test focuses on the nav
// chrome contract (brand, dropdown, version footer, hide-on-/login).
// Without this stub, NavBar mounts <HeaderMetrics /> which opens a
// real EventSource via the singleton stream and pollutes module state
// between tests. The widget has dedicated coverage in
// `header-metrics.test.tsx` + `header-metrics-stream.test.ts`.
vi.mock('@/components/header-metrics', () => ({
  HeaderMetrics: () => null,
}));

// Imported AFTER the mock so the navbar picks up the stub.
import { NavBar } from '@/components/nav-bar';

// Each test wraps NavBar in a SWRConfig with a fresh provider so the
// /api/version cache from a previous test doesn't leak across. Without
// this, the "errors → fallback" test sees a cached success from an
// earlier test and never re-fetches.
function renderNav() {
  return render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <NavBar />
    </SWRConfig>,
  );
}

// next/navigation mocks — usePathname is the gate-keeper for the
// hide-on-/login/setup branch; useRouter is exercised by sign-out.
const mockReplace = vi.fn();
let mockPath = '/models';
vi.mock('next/navigation', () => ({
  usePathname: () => mockPath,
  useRouter: () => ({ replace: mockReplace }),
}));

// The version fetch uses authFetchJSON which goes through authFetch which
// goes through window.fetch. Stubbing fetch at the global level is the
// least invasive — same pattern as tests/component/login.test.tsx.
function stubVersionFetch(body: object | null, status = 200) {
  const fetchMock = vi.fn().mockImplementation(async (url: RequestInfo) => {
    const u = typeof url === 'string' ? url : (url as Request).url;
    if (u.endsWith('/api/version')) {
      if (body === null) return new Response('boom', { status });
      return new Response(JSON.stringify(body), { status });
    }
    // CSRF preflight + anything else → return an empty 200 so authFetch
    // doesn't blow up. Tests don't exercise mutating paths.
    return new Response('{}', { status: 200 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('NavBar', () => {
  beforeEach(() => {
    mockPath = '/models';
    mockReplace.mockReset();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders the emerald brand on a non-auth route', async () => {
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1234567890' });
    renderNav();
    expect(screen.getByText('LM Warden')).toBeInTheDocument();
  });

  it('returns null on /login', () => {
    mockPath = '/login';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    const { container } = renderNav();
    expect(container).toBeEmptyDOMElement();
  });

  it('returns null on /setup', () => {
    mockPath = '/setup/welcome';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    const { container } = renderNav();
    expect(container).toBeEmptyDOMElement();
  });

  it('toggles the dropdown and shows the post-overhaul menu entries', () => {
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1234567890' });
    renderNav();
    // Closed → menu items not in DOM.
    expect(screen.queryByRole('menuitem', { name: /^models$/i })).toBeNull();
    // Open by clicking the hamburger.
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    expect(screen.getByRole('menuitem', { name: /models/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /tokens/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /stats/i })).toBeInTheDocument();
    // Benchmarks entry removed in epic/overhaul S1.
    expect(screen.queryByRole('menuitem', { name: /benchmarks/i })).toBeNull();
    // Cache entry added in epic/overhaul S6 (#105) — re-homed from /stats.
    expect(screen.getByRole('menuitem', { name: /^cache$/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /settings/i })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /sign out/i })).toBeInTheDocument();
  });

  it('renders the version footer from /api/version', async () => {
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1234567890' });
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    // The version + short-sha should both appear; we pin the data not
    // the separator. SWR resolves async, so wait for the value to land.
    await waitFor(() => {
      const footer = screen.getByText(/2026\.05\.13\.1/);
      expect(footer.textContent).toMatch(/abcdef1/);
    });
  });

  it('falls back to "v? · ?" when /api/version errors', async () => {
    stubVersionFetch(null, 500);
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    await waitFor(() => {
      // Match the "? · ?" pattern — punctuation may be · or — depending
      // on style; pin only the literal "v? " prefix and a trailing "?".
      expect(screen.getByText(/v\?.*\?/)).toBeInTheDocument();
    });
  });

  it('highlights the active route in the dropdown', () => {
    mockPath = '/tokens';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    const active = screen.getByRole('menuitem', { name: /^tokens$/i });
    expect(active).toHaveAttribute('aria-current', 'page');
    const inactive = screen.getByRole('menuitem', { name: /^models$/i });
    expect(inactive).not.toHaveAttribute('aria-current', 'page');
  });

  it('highlights the parent route for sub-paths (e.g. /models/abc → Models)', () => {
    mockPath = '/models/some-model-id';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    const models = screen.getByRole('menuitem', { name: /^models$/i });
    expect(models).toHaveAttribute('aria-current', 'page');
  });

  // #83 regression — the backend can return a version string that
  // already includes a leading "v" (e.g. `v2026.05.19.1` for a tagged
  // release). The nav footer used to prefix its own "v", producing the
  // double "vv" we shipped on production for ~24h. The fix in
  // formatVersion drops the prefix; we pin both shapes here so the
  // bug can never come back via a "helpful" template re-edit.
  it('does NOT double the "v" prefix when /api/version already returns one', async () => {
    stubVersionFetch({ version: 'v2026.05.19.1', sha: 'abcdef1234567890' });
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    await waitFor(() => {
      const footer = screen.getByText(/2026\.05\.19\.1/);
      expect(footer.textContent ?? '').not.toMatch(/vv/);
      // The literal version is preserved verbatim.
      expect(footer.textContent ?? '').toMatch(/v2026\.05\.19\.1/);
    });
  });

  it('renders a bare version without inventing a "v" prefix', async () => {
    stubVersionFetch({ version: '2026.05.19.1', sha: 'abcdef1234567890' });
    renderNav();
    fireEvent.click(screen.getByRole('button', { name: /open menu/i }));
    await waitFor(() => {
      const footer = screen.getByText(/2026\.05\.19\.1/);
      // No "v" anywhere in the version segment — the footer is
      // "{version} · {sha}" with the version emitted verbatim.
      expect(footer.textContent ?? '').not.toMatch(/v2026\.05/);
    });
  });

  // #39 regression — the previous unauth-route guard was a
  // `startsWith('/login')` check that swallowed every path beginning
  // with "/login". A hypothetical /login-help marketing page would
  // have rendered as unauthenticated chrome (no nav, no header
  // widget). Exact-match closes that hole; we pin a few realistic
  // shapes so the fix can't regress to a prefix check by accident.
  it('renders normally on /login-help (exact-match guard)', () => {
    mockPath = '/login-help';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    renderNav();
    expect(screen.getByText('LM Warden')).toBeInTheDocument();
  });

  it('renders normally on /login.json (exact-match guard)', () => {
    mockPath = '/login.json';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    renderNav();
    expect(screen.getByText('LM Warden')).toBeInTheDocument();
  });

  it('still hides on the trailing-slash /login/ variant', () => {
    mockPath = '/login/';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    const { container } = renderNav();
    expect(container).toBeEmptyDOMElement();
  });

  it('renders normally on /setup-guide (NOT a wizard subroute)', () => {
    mockPath = '/setup-guide';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    renderNav();
    expect(screen.getByText('LM Warden')).toBeInTheDocument();
  });

  it('still hides on /setup/<step> wizard subroutes', () => {
    mockPath = '/setup/admin';
    stubVersionFetch({ version: '2026.05.13.1', sha: 'abcdef1' });
    const { container } = renderNav();
    expect(container).toBeEmptyDOMElement();
  });
});
