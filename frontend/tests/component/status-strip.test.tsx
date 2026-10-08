import { afterEach, describe, it, expect } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { StatusStrip } from '@/components/models/settings/status-strip';

afterEach(cleanup);

// Plan 2026-10-04-settings-redesign §2, §6, §7 — ONE line answering "what can
// I change right now?". Loaded and 409 are the page's single role=alert; the
// not-loaded strip is a polite role=status so it never competes with a field
// error for the alert slot.

describe('StatusStrip', () => {
  it('loaded: green alert, says unloaded + before editing, links to the model page', () => {
    render(<StatusStrip status="loaded" conflict={false} modelId="qwen3-4b" />);
    const strip = screen.getByRole('alert');
    expect(strip).toHaveAttribute('data-testid', 'settings-loaded-banner');
    expect(strip).toHaveAttribute('data-variant', 'loaded');
    expect(strip.textContent).toMatch(/unloaded/);
    expect(strip.textContent).toMatch(/before editing/);
    expect(strip.textContent).toMatch(/Live changes only\./);
    expect(screen.getByRole('link', { name: /unload on the model page/i })).toHaveAttribute(
      'href',
      '/models/qwen3-4b',
    );
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('not loaded: neutral status, no alert, links to load', () => {
    render(<StatusStrip status="pulled" conflict={false} modelId="m1" />);
    const strip = screen.getByRole('status');
    expect(strip.textContent).toMatch(/Everything is editable\./);
    expect(strip).not.toHaveAttribute('data-testid', 'settings-loaded-banner');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('link', { name: /load on the model page/i })).toHaveAttribute(
      'href',
      '/models/m1',
    );
  });

  it.each(['registered', 'loading', 'unloading', 'pulling', 'failed'])(
    '%s counts as not loaded',
    (status) => {
      render(<StatusStrip status={status} conflict={false} modelId="m1" />);
      expect(screen.getByRole('status')).toBeInTheDocument();
      expect(screen.queryByRole('alert')).toBeNull();
    },
  );

  it('409 conflict: amber alert with the unload copy, wins over loaded', () => {
    const { rerender } = render(<StatusStrip status="pulled" conflict modelId="m1" />);
    let strip = screen.getByRole('alert');
    expect(strip).toHaveAttribute('data-testid', 'settings-loaded-banner');
    expect(strip).toHaveAttribute('data-variant', 'conflict');
    expect(strip.textContent).toMatch(/unloaded/);
    expect(strip.textContent).toMatch(/before editing/);
    expect(screen.queryByRole('status')).toBeNull();

    rerender(<StatusStrip status="loaded" conflict modelId="m1" />);
    expect(screen.getAllByRole('alert')).toHaveLength(1);
    strip = screen.getByRole('alert');
    expect(strip).toHaveAttribute('data-variant', 'conflict');
  });

  it('encodes the model id in the link', () => {
    render(<StatusStrip status="pulled" conflict={false} modelId="a/b c" />);
    expect(screen.getByRole('link')).toHaveAttribute('href', '/models/a%2Fb%20c');
  });
});
