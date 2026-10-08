import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { PullProgress } from '@/components/models/pull-progress';

describe('PullProgress', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders a progressbar with computed percent when total is known', () => {
    render(
      <PullProgress status="pulling" pulledBytes={50 * 1024 * 1024} pulledTotal={200 * 1024 * 1024} />,
    );
    const bar = screen.getByRole('progressbar');
    expect(bar).toHaveAttribute('aria-valuenow', '25');
    expect(bar).toHaveAttribute('aria-valuemax', '100');
    // Numeric percent must be surfaced as visible text so headless monitoring
    // (or a colour-blind operator) can read it without relying on the bar.
    expect(screen.getByText(/25%/)).toBeInTheDocument();
  });

  it('renders an indeterminate state with no aria-valuenow when total is unknown', () => {
    render(<PullProgress status="pulling" pulledBytes={1024} pulledTotal={null} />);
    const bar = screen.getByRole('progressbar');
    // ARIA pattern: omit aria-valuenow on indeterminate bars so AT announces
    // "loading" rather than a misleading 0%.
    expect(bar).not.toHaveAttribute('aria-valuenow');
    // Should still show a byte readout for the partial total.
    expect(screen.getByText(/1\.0 KiB/)).toBeInTheDocument();
  });

  it('renders nothing for terminal statuses (loaded, failed, etc.)', () => {
    const { container } = render(
      <PullProgress status="loaded" pulledBytes={123} pulledTotal={123} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('still renders for the registered (pre-pull) status', () => {
    // The detail page mounts this component for both `registered` and
    // `pulling` so the operator sees a 0-of-? bar while the worker is
    // booting. Pinning that here so a future refactor doesn't lose the
    // pre-pull placeholder.
    render(<PullProgress status="registered" pulledBytes={0} pulledTotal={null} />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
