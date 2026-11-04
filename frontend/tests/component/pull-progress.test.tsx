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

  // lmwarden.com, 2026-11-04: a restart killed a pull and the row went back to
  // `registered` with total 0. The bar kept pulsing at "0 B" and read as a
  // live download that had stalled. A `registered` row is NOT downloading.
  it('does not animate the bar for a registered row', () => {
    const { container } = render(
      <PullProgress status="registered" pulledBytes={0} pulledTotal={0} />,
    );
    expect(container.querySelector('.animate-pulse')).toBeNull();
    expect(screen.getByText(/not downloading/i)).toBeInTheDocument();
  });

  it('shows why a registered row is not downloading', () => {
    render(
      <PullProgress
        status="registered"
        pulledBytes={1288490188}
        pulledTotal={55579322990}
        lastError="pull interrupted by a restart; resuming"
      />,
    );
    expect(screen.getByText(/interrupted by a restart/)).toBeInTheDocument();
    // Bytes already finished stay visible.
    expect(screen.getByText(/1\.2 GiB/)).toBeInTheDocument();
  });

  it('still pulses while pulling with no total yet', () => {
    const { container } = render(
      <PullProgress status="pulling" pulledBytes={0} pulledTotal={null} />,
    );
    expect(container.querySelector('.animate-pulse')).not.toBeNull();
    expect(screen.queryByText(/not downloading/i)).toBeNull();
  });
});
