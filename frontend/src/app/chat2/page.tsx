'use client';
// /chat2 — host shell around the shared chat UI. Warden supplies auth, theme,
// capabilities and the initial chat id; the package owns everything else.
// `useSearchParams` opts the tree into client-side bailout, hence <Suspense>.
import { Suspense, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import { ChatApp } from '@podwarden/chat-ui';
import { createHttpAdapters } from '@podwarden/chat-ui/adapters-http';
import { authFetch } from '@/lib/auth-fetch';
import { useTheme } from '@/lib/theme';

// only the non-default: `systemPrompt: 'editable'` is the package default.
const CAPABILITIES = { toolPolicy: 'hidden' as const };

function Chat2Shell() {
  const params = useSearchParams();
  const { theme } = useTheme();
  // stable across renders — the package hooks tolerate churn, but there is no reason to cause it
  const adapters = useMemo(() => createHttpAdapters({ baseUrl: '/api/chat2', fetch: authFetch }), []);
  // Full-height shell: 100vh minus the nav bar (3.5rem, h-14) and the
  // breadcrumb strip (30px, BREADCRUMB_BAR_PX in components/breadcrumb-header).
  return (
    <ChatApp
      adapters={adapters}
      capabilities={CAPABILITIES}
      theme={theme === 'retro' ? 'light' : 'dark'}
      initialChatId={params.get('c')}
      syncUrlParam="c"
      rootInertId="app-root"
      className="-m-6 h-[calc(100vh-3.5rem-30px)]"
    />
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <Chat2Shell />
    </Suspense>
  );
}
