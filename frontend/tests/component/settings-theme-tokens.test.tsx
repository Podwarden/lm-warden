import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { LayoutNotice } from '@/components/models/layout-notice';
import { GpuChecklist } from '@/components/gpu/gpu-checklist';

// T5 real-page check: LayoutNotice and GpuChecklist used fixed dark palette
// classes (sky-100 / amber-100 text, slate-500 "GiB free"), which measured
// 1.6:1 on the light theme and 3.96:1 on retro-dark. They must use the theme
// tokens, which resolve per theme.
const PALETTE = /\b(?:text|bg|border)-(?:slate|sky|amber|red|blue|emerald)-\d{2,3}\b/;

afterEach(cleanup);

describe('theme tokens on the settings page chrome', () => {
  it.each(['info', 'warning'] as const)('LayoutNotice %s uses tokens, not a fixed palette', (level) => {
    const { container } = render(
      <LayoutNotice modelId="m" level={level} message="Started with 2 replicas." onDismissed={vi.fn()} />,
    );
    expect(container.innerHTML).not.toMatch(PALETTE);
  });

  it('GpuChecklist rows use tokens, not a fixed palette', () => {
    const { container } = render(
      <GpuChecklist
        gpus={[{ index: 0, name: 'NVIDIA RTX A4000', memory_total_mib: 16376, memory_used_mib: 400, utilization_pct: 0 }]}
        selected={[0, 3]}
        onChange={vi.fn()}
      />,
    );
    expect(container.innerHTML).not.toMatch(PALETTE);
  });
});
