import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { KeepCapturesCard } from '@/components/tokens/detail/keep-captures-card';

// Prompt Lens keep captures (G-R4): the admin key page's switch and its note.
describe('keep captures card', () => {
  afterEach(() => { cleanup(); });

  it('says it is off and turning it on calls onToggle(true)', () => {
    const onToggle = vi.fn().mockResolvedValue(true);
    render(<KeepCapturesCard on={false} onToggle={onToggle} />);
    const sw = screen.getByRole('switch', { name: 'Keep captures' });
    expect(sw).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByText('off')).toBeInTheDocument();
    fireEvent.click(sw);
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it('when on, turning it off calls onToggle(false)', () => {
    const onToggle = vi.fn().mockResolvedValue(true);
    render(<KeepCapturesCard on onToggle={onToggle} />);
    const sw = screen.getByRole('switch', { name: 'Keep captures' });
    expect(sw).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(sw);
    expect(onToggle).toHaveBeenCalledWith(false);
  });

  it('notes that prompts are stored on disk and that off deletes them, as Prompt Lens', () => {
    render(<KeepCapturesCard on={false} onToggle={vi.fn()} />);
    const note = screen.getByTestId('keep-captures-note').textContent ?? '';
    expect(note).toMatch(/stored in this server's database/);
    expect(note).toMatch(/on disk/);
    expect(note).toMatch(/Turning\s+this off deletes what was kept/);
    expect(screen.getByRole('heading', { name: 'Prompt Lens' })).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/god[ -]?mode/i);
  });

  it('is disabled while busy and shows an error', () => {
    render(<KeepCapturesCard on={false} onToggle={vi.fn()} busy error="nope" />);
    expect(screen.getByRole('switch', { name: 'Keep captures' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('nope');
  });

  it('says when Prompt Lens is off on the server that nothing is kept, the admin can still open kept ones and the holder cannot', () => {
    render(<KeepCapturesCard on onToggle={vi.fn()} platformOn={false} />);
    const off = screen.getByTestId('keep-captures-off').textContent ?? '';
    expect(off).toMatch(/nothing is captured or\s+kept now/);
    expect(off).toMatch(/you can still open them in Prompt Lens\s+\(read-only\)/);
    expect(off).toMatch(/holder cannot until it is on again/);
    cleanup();
    render(<KeepCapturesCard on onToggle={vi.fn()} />);
    expect(screen.queryByTestId('keep-captures-off')).toBeNull();
  });
});
