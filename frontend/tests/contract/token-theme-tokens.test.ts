import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { VW_TOKENS, btn, badge, prioTone } from '@/components/tokens/detail/styles';

const read = (p: string) => readFileSync(path.resolve(__dirname, '../..', p), 'utf8');

// Spec §4.6: "Where no token exists, add one to globals.css for every theme
// rather than hard-coding a hex." A token defined for retro-dark only would
// make the light theme fall back to nothing (transparent) silently.
describe('token details theme tokens', () => {
  const css = read('src/app/globals.css');
  const dark = css.slice(css.indexOf('/* vw-tokens: dark */'), css.indexOf('/* vw-tokens: light */'));
  const light = css.slice(css.indexOf('/* vw-tokens: light */'), css.indexOf('/* vw-tokens: end */'));
  const tw = read('tailwind.config.ts');

  it.each(VW_TOKENS)('--vw-%s is defined for retro-dark, retro and in tailwind', (name) => {
    expect(dark).toMatch(new RegExp(`--vw-${name}:\\s*\\d+ \\d+ \\d+;`));
    expect(light).toMatch(new RegExp(`--vw-${name}:\\s*\\d+ \\d+ \\d+;`));
    expect(tw).toContain(`'${name}': 'rgb(var(--vw-${name}) / <alpha-value>)'`);
  });

  it('retro-dark values are the mockup values', () => {
    expect(dark).toContain('--vw-btn: 180 83 9;');
    expect(dark).toContain('--vw-btn-hover: 194 97 15;');
    expect(dark).toContain('--vw-rule-soft: 92 69 32;');
    expect(dark).toContain('--vw-dock: 21 15 6;');
    expect(dark).toContain('--vw-prompt: 167 139 250;');
  });

  it('helpers compose the mockup classes', () => {
    expect(btn('primary')).toContain('bg-vw-btn');
    expect(btn('primary')).toContain('px-3.5 py-[7px] text-[13px]');
    expect(badge('paused')).toContain('bg-vw-amber-bg/60');
    expect(prioTone(9).fg).toBe('rgb(var(--vw-danger-fg))');
    expect(prioTone(3).bg).toBe('rgb(var(--vw-info-bg) / .5)');
    expect(prioTone(2).bg).toBe('rgb(var(--vw-low-bg) / .6)');
  });
});
