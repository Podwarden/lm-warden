// Component tests for the god-mode media strip + lightbox (spec 2026-08-03).
// authFetch is mocked; URL.createObjectURL/revokeObjectURL are stubbed since
// jsdom lacks them.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, cleanup, fireEvent, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({ authFetch: vi.fn() }));
vi.mock('@/lib/auth-fetch', () => ({ authFetch: mocks.authFetch }));

import { MediaStrip } from '@/components/godmode/godmode-media';

beforeEach(() => {
  mocks.authFetch.mockReset();
  (URL as any).createObjectURL = vi.fn(() => 'blob:fake-url');
  (URL as any).revokeObjectURL = vi.fn();
});

afterEach(() => cleanup());

function okBlobResponse() {
  return Promise.resolve({
    ok: true,
    status: 200,
    blob: () => Promise.resolve(new Blob(['x'], { type: 'image/png' })),
  } as unknown as Response);
}

describe('MediaStrip', () => {
  it('renders a thumbnail per stored entry via authFetch -> objectURL', async () => {
    mocks.authFetch.mockImplementation(okBlobResponse);
    render(
      <MediaStrip
        media={[
          { kind: 'image', media_id: 'a'.repeat(16), mime: 'image/png', chars: 100 },
          { kind: 'image', media_id: 'b'.repeat(16), mime: 'image/jpeg', chars: 200 },
        ]}
      />,
    );
    await waitFor(() => {
      expect(screen.getAllByTestId('godmode-media-thumb')).toHaveLength(2);
    });
    expect(mocks.authFetch).toHaveBeenCalledWith(
      `/api/admin/godmode/media/${'a'.repeat(16)}`,
    );
    const imgs = screen.getAllByRole('img');
    expect(imgs[0]).toHaveAttribute('src', 'blob:fake-url');
  });

  it('renders remote-url entries as direct <img> without authFetch', () => {
    render(<MediaStrip media={[{ kind: 'image', url: 'https://example.com/cat.jpg' }]} />);
    expect(screen.getByRole('img')).toHaveAttribute('src', 'https://example.com/cat.jpg');
    expect(mocks.authFetch).not.toHaveBeenCalled();
  });

  it('renders an evicted chip on 404', async () => {
    mocks.authFetch.mockResolvedValue({ ok: false, status: 404 } as Response);
    render(
      <MediaStrip media={[{ kind: 'image', media_id: 'c'.repeat(16), mime: 'image/png', chars: 5 }]} />,
    );
    await waitFor(() => {
      expect(screen.getByTestId('godmode-media-evicted')).toBeInTheDocument();
    });
    expect(screen.queryByRole('img')).toBeNull();
  });

  it('renders dropped placeholders as non-clickable chips', () => {
    render(
      <MediaStrip
        media={[
          { kind: 'image', mime: 'image/png', chars: 14_000_000, dropped: 'too_large' },
          { kind: 'image', dropped: 'count', count: 3 },
        ]}
      />,
    );
    const tooLarge = screen.getByTestId('godmode-media-dropped');
    expect(tooLarge.textContent).toMatch(/image\/png/);
    expect(tooLarge.textContent).toMatch(/too large/i);
    expect(screen.getByTestId('godmode-media-overflow').textContent).toMatch(/\+3 more/i);
    fireEvent.click(tooLarge);
    expect(screen.queryByTestId('godmode-lightbox')).toBeNull();
  });

  it('opens a lightbox on thumb click and closes on Escape and backdrop click', async () => {
    mocks.authFetch.mockImplementation(okBlobResponse);
    render(
      <MediaStrip media={[{ kind: 'image', media_id: 'd'.repeat(16), mime: 'image/png', chars: 5 }]} />,
    );
    await waitFor(() => screen.getByTestId('godmode-media-thumb'));
    fireEvent.click(screen.getByTestId('godmode-media-thumb'));
    expect(screen.getByTestId('godmode-lightbox')).toBeInTheDocument();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('godmode-lightbox')).toBeNull();
    fireEvent.click(screen.getByTestId('godmode-media-thumb'));
    fireEvent.click(screen.getByTestId('godmode-lightbox'));
    expect(screen.queryByTestId('godmode-lightbox')).toBeNull();
  });

  it('revokes object URLs on unmount', async () => {
    mocks.authFetch.mockImplementation(okBlobResponse);
    const { unmount } = render(
      <MediaStrip media={[{ kind: 'image', media_id: 'e'.repeat(16), mime: 'image/png', chars: 5 }]} />,
    );
    await waitFor(() => screen.getByTestId('godmode-media-thumb'));
    unmount();
    expect((URL as any).revokeObjectURL).toHaveBeenCalledWith('blob:fake-url');
  });

  it('revokes the object URL even when unmount races ahead of the pending fetch', async () => {
    // Regression for the leak in review round 1: unmount (or a mediaId
    // change) can land while authFetch/blob() is still in flight. Cleanup
    // runs first — with nothing to revoke yet — then the .then callback
    // creates the object URL *after* the component is gone. That URL must
    // be revoked immediately rather than adopted, or it leaks for the life
    // of the tab.
    let resolveFetch!: (r: Response) => void;
    mocks.authFetch.mockImplementation(
      () => new Promise<Response>((resolve) => { resolveFetch = resolve; }),
    );
    const { unmount } = render(
      <MediaStrip media={[{ kind: 'image', media_id: 'f'.repeat(16), mime: 'image/png', chars: 5 }]} />,
    );
    // Fetch is still pending — no thumbnail yet.
    expect(screen.queryByTestId('godmode-media-thumb')).toBeNull();

    unmount();
    expect((URL as any).revokeObjectURL).not.toHaveBeenCalled();

    // Now let the fetch resolve, after the component is gone.
    await act(async () => {
      resolveFetch({
        ok: true,
        status: 200,
        blob: () => Promise.resolve(new Blob(['x'], { type: 'image/png' })),
      } as unknown as Response);
      // Flush the .then chain (blob() await + createObjectURL).
      await Promise.resolve();
      await Promise.resolve();
    });

    expect((URL as any).createObjectURL).toHaveBeenCalledTimes(1);
    expect((URL as any).revokeObjectURL).toHaveBeenCalledWith('blob:fake-url');
  });
});
