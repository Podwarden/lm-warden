// "Download report" on the scorecard (analysis Task 4.4): HTML or JSON, "include tool names", the
// right route and credential per scope; the menu is absent without a session id.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

import { Scorecard } from '@/components/prompt-lens/sessions/scorecard';
import { reportFileName, reportUrl } from '@/lib/prompt-lens/report';

import { scorecardFixture } from './prompt-lens-scorecard-fixture';

const dl = vi.hoisted(() => ({ fn: vi.fn(async (_s: string, _r: string, _o: Record<string, unknown>) => true) }));
vi.mock('@/lib/prompt-lens/report', async (orig) => ({ ...(await orig<typeof import('@/lib/prompt-lens/report')>()), downloadReport: dl.fn }));

afterEach(() => {
  cleanup();
  dl.fn.mockClear();
});

describe('report urls', () => {
  it('uses the key route for a key holder and the admin route for the admin', () => {
    expect(reportUrl('key', 's_abc', { format: 'html', toolNames: false })).toBe('/api/me/prompt-lens/sessions/s_abc/report?format=html&tool_names=0');
    expect(reportUrl('admin', 's_abc', { format: 'json', toolNames: true })).toBe('/api/admin/prompt-lens/sessions/s_abc/report?format=json&tool_names=1');
    expect(reportFileName('json')).toBe('prompt-lens-report.json');
  });
});

describe('Download report menu', () => {
  it('is absent without a session id', () => {
    render(<Scorecard data={scorecardFixture()} scope="key" onShowTurn={vi.fn()} />);
    expect(screen.queryByTestId('report-menu')).toBeNull();
  });

  it('downloads HTML and JSON, and passes "include tool names"', async () => {
    render(<Scorecard data={scorecardFixture()} scope="key" onShowTurn={vi.fn()} session="s_abc" />);
    fireEvent.click(screen.getByRole('button', { name: 'Download report' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /HTML/ }));
    await waitFor(() => expect(dl.fn).toHaveBeenCalledWith('key', 's_abc', { format: 'html', toolNames: false }));
    fireEvent.click(screen.getByRole('button', { name: 'Download report' }));
    fireEvent.click(screen.getByLabelText('Include tool names'));
    fireEvent.click(screen.getByRole('menuitem', { name: /JSON/ }));
    await waitFor(() => expect(dl.fn).toHaveBeenLastCalledWith('key', 's_abc', { format: 'json', toolNames: true }));
  });

  it('says what the file holds and shows a failure', async () => {
    dl.fn.mockResolvedValueOnce(false);
    render(<Scorecard data={scorecardFixture()} scope="admin" onShowTurn={vi.fn()} session="s_abc" />);
    fireEvent.click(screen.getByRole('button', { name: 'Download report' }));
    expect(document.body.textContent).toMatch(/Numbers and codes only/);
    fireEvent.click(screen.getByRole('menuitem', { name: /HTML/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('could not be downloaded');
  });
});
