import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
vi.mock('next/navigation', () => ({ useSearchParams: () => new URLSearchParams('c=c1') }));
vi.mock('@podwarden/chat-ui', async (orig) => {
  const real = await orig<typeof import('@podwarden/chat-ui')>();
  return { ...real, ChatApp: (p: Record<string, unknown>) => <pre data-testid="props">{JSON.stringify({ theme: p.theme, initialChatId: p.initialChatId, syncUrlParam: p.syncUrlParam, id: (p.adapters as { id: string }).id, toolPolicy: (p.capabilities as { toolPolicy: string }).toolPolicy, rootInertId: p.rootInertId })}</pre> };
});
import { ThemeProvider } from '@/lib/theme';
import Page from '@/app/chat2/page';
afterEach(cleanup);
describe('/chat2 shell', () => {
  it('hands the package Warden auth, theme, capabilities and the ?c= id', async () => {
    render(<ThemeProvider><Page /></ThemeProvider>);
    await waitFor(() => expect(screen.getByTestId('props')).toBeInTheDocument());
    expect(JSON.parse(screen.getByTestId('props').textContent!)).toEqual({
      theme: 'dark', initialChatId: 'c1', syncUrlParam: 'c', id: '/api/chat2', toolPolicy: 'hidden', rootInertId: 'app-root',
    });
  });
});
