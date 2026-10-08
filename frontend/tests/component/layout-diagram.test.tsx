import { afterEach, describe, it, expect } from 'vitest';
import { cleanup, render, screen, within } from '@testing-library/react';
import { LayoutDiagram } from '@/components/models/settings/layout-diagram';

afterEach(cleanup);

// Plan 2026-10-04-settings-redesign §4.3 — read-only picture of how the
// selected GPUs fall into replicas. role=img; its accessible name is the
// equation sentence.

function boxes() {
  return screen.getAllByTestId('layout-replica');
}

describe('LayoutDiagram', () => {
  it('4 GPUs / dp 4 / tp 1 → four boxes of one GPU', () => {
    render(<LayoutDiagram gpuIndices={[0, 1, 2, 3]} dp={4} tp={1} />);
    const img = screen.getByRole('img', {
      name: '4 GPUs = 4 replicas × 1 GPU each (tensor-parallel 1)',
    });
    expect(img).toHaveAttribute('data-invalid', 'false');
    const b = boxes();
    expect(b).toHaveLength(4);
    expect(b.map((x) => x.textContent)).toEqual([
      'Replica 1GPU 0',
      'Replica 2GPU 1',
      'Replica 3GPU 2',
      'Replica 4GPU 3',
    ]);
  });

  it('4 GPUs / dp 2 / tp 2 → two pairs', () => {
    render(<LayoutDiagram gpuIndices={[0, 1, 2, 3]} dp={2} tp={2} />);
    const b = boxes();
    expect(b).toHaveLength(2);
    expect(within(b[0]).getAllByText(/^GPU \d$/).map((x) => x.textContent)).toEqual(['GPU 0', 'GPU 1']);
    expect(within(b[1]).getAllByText(/^GPU \d$/).map((x) => x.textContent)).toEqual(['GPU 2', 'GPU 3']);
  });

  it('6 GPUs / dp 3 / tp 2 → three pairs in selection order', () => {
    render(<LayoutDiagram gpuIndices={[0, 1, 2, 3, 4, 5]} dp={3} tp={2} />);
    const b = boxes();
    expect(b).toHaveLength(3);
    expect(within(b[2]).getAllByText(/^GPU \d$/).map((x) => x.textContent)).toEqual(['GPU 4', 'GPU 5']);
  });

  it('dp 1 labels the box "One replica"', () => {
    render(<LayoutDiagram gpuIndices={[0, 1]} dp={1} tp={2} />);
    expect(boxes()).toHaveLength(1);
    expect(screen.getByText('One replica')).toBeInTheDocument();
  });

  it('invalid split: red state, the sentence, one Unassigned box', () => {
    render(<LayoutDiagram gpuIndices={[0, 1, 2, 3]} dp={3} tp={1} />);
    const img = screen.getByRole('img', { name: "4 GPUs can't split into 3 equal replicas" });
    expect(img).toHaveAttribute('data-invalid', 'true');
    const b = boxes();
    expect(b).toHaveLength(1);
    expect(within(b[0]).getByText('Unassigned')).toBeInTheDocument();
    expect(within(b[0]).getAllByText(/^GPU \d$/)).toHaveLength(4);
    // The picture is not an alert — the field error under Replicas is.
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('no GPUs selected', () => {
    render(<LayoutDiagram gpuIndices={[]} dp={1} tp={1} />);
    expect(screen.getByRole('img', { name: 'No GPUs selected' })).toHaveAttribute('data-invalid', 'true');
  });

  it('llama.cpp: one box, layers split across the GPUs', () => {
    render(<LayoutDiagram gpuIndices={[0, 1]} dp={1} tp={1} backend="llamacpp" />);
    screen.getByRole('img', { name: '2 GPUs · layers split across them' });
    expect(boxes()).toHaveLength(1);
    expect(screen.queryByText(/replica/i)).toBeNull();
  });

  it('never exposes "parallelism" or "GPU layers" as a label', () => {
    const { rerender } = render(<LayoutDiagram gpuIndices={[0, 1, 2, 3]} dp={2} tp={2} />);
    expect(screen.queryByLabelText(/parallelism/i)).toBeNull();
    expect(screen.queryByLabelText(/gpu layers/i)).toBeNull();
    rerender(<LayoutDiagram gpuIndices={[0, 1]} dp={1} tp={1} backend="llamacpp" />);
    expect(screen.queryByLabelText(/parallelism/i)).toBeNull();
    expect(screen.queryByLabelText(/gpu layers/i)).toBeNull();
  });
});
