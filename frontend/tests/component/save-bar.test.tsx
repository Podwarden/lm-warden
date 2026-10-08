import { afterEach, describe, it, expect, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { SaveBar, type SaveBarRow, type SaveBarProps } from '@/components/models/settings/save-bar';

afterEach(cleanup);

// Plan 2026-10-04-settings-redesign §4.1 — the sticky save bar replaces the
// top Save/Reset. Page tests rely on EXACTLY ONE button matching /save/i at
// all times, so the "N unsaved" summary must never be (inside) a button.

const ROWS: SaveBarRow[] = [
  { key: 'data_parallel_size', label: 'Data-parallel replicas', before: '4', after: '2', live: false },
  { key: 'tensor_parallel_size', label: 'Tensor-parallel size', before: '1', after: '2', live: false },
  { key: 'max_model_len', label: 'Max model length', before: '32768', after: '16384', live: false },
  { key: 'supports_vision', label: 'Vision', before: 'Auto', after: 'Yes', live: true },
];

function setup(over: Partial<SaveBarProps> = {}) {
  const onSave = vi.fn();
  const onReset = vi.fn();
  const props: SaveBarProps = {
    rows: [],
    invalidReason: null,
    saving: false,
    canSave: false,
    isLoaded: false,
    saveError: null,
    onSave,
    onReset,
    ...over,
  };
  const utils = render(<SaveBar {...props} />);
  return { ...utils, onSave, onReset, props };
}

function summary() {
  return screen.getByTestId('settings-save-summary');
}

describe('SaveBar', () => {
  it('clean: "No unsaved changes", Save and Reset disabled, no Review', () => {
    setup();
    expect(summary()).toHaveTextContent('No unsaved changes');
    expect(summary()).toHaveAttribute('aria-live', 'polite');
    expect(screen.getByTestId('settings-save')).toBeDisabled();
    expect(screen.getByTestId('settings-reset')).toBeDisabled();
    expect(screen.queryByRole('button', { name: /review/i })).toBeNull();
  });

  it('dirty: total, "k on next load" and "m live" tags', () => {
    setup({ rows: ROWS, canSave: true });
    expect(summary()).toHaveTextContent('4 unsaved changes');
    expect(screen.getByTestId('save-bar-tag-next')).toHaveTextContent('3 on next load');
    expect(screen.getByTestId('save-bar-tag-live')).toHaveTextContent('1 live');
    expect(screen.getByTestId('settings-save')).toBeEnabled();
    expect(screen.getByTestId('settings-reset')).toBeEnabled();
  });

  it('singular and tag omission', () => {
    setup({ rows: [ROWS[3]], canSave: true });
    expect(summary()).toHaveTextContent('1 unsaved change');
    expect(summary()).not.toHaveTextContent('1 unsaved changes');
    expect(screen.queryByTestId('save-bar-tag-next')).toBeNull();
    expect(screen.getByTestId('save-bar-tag-live')).toHaveTextContent('1 live');
  });

  it('exactly one button matches /save/i in every state', () => {
    const states: Partial<SaveBarProps>[] = [
      {},
      { rows: ROWS, canSave: true },
      { rows: ROWS, invalidReason: 'replicas must divide the 4 selected GPUs', fixTargetId: 'dp' },
      { rows: [ROWS[3]], isLoaded: true, canSave: true },
      { rows: ROWS, saveError: 'Save failed: HTTP 500' },
    ];
    for (const s of states) {
      const { unmount } = setup(s);
      // Open Review too, if present, so the diff list is in the a11y tree.
      const review = screen.queryByRole('button', { name: /review/i });
      if (review) fireEvent.click(review);
      expect(screen.getAllByRole('button', { name: /save/i })).toHaveLength(1);
      unmount();
    }
  });

  it('the unsaved summary is not inside a button', () => {
    setup({ rows: ROWS, canSave: true });
    expect(screen.getByText('4 unsaved changes').closest('button')).toBeNull();
  });

  it('invalid: red reason with a Fix button that focuses the target input', () => {
    const { container } = render(<input id="dp" aria-label="Data-parallel replicas" />);
    void container;
    setup({ rows: ROWS, invalidReason: 'replicas must divide the 4 selected GPUs', fixTargetId: 'dp' });
    expect(summary()).toHaveTextContent("Can't save: replicas must divide the 4 selected GPUs.");
    expect(screen.getByTestId('settings-save')).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Fix' }));
    expect(document.activeElement).toBe(document.getElementById('dp'));
  });

  it('invalid with no fix target renders no Fix button', () => {
    setup({ rows: ROWS, invalidReason: 'select at least one GPU' });
    expect(summary()).toHaveTextContent("Can't save: select at least one GPU.");
    expect(screen.queryByRole('button', { name: 'Fix' })).toBeNull();
  });

  it('Review toggles the diff with aria-expanded / aria-controls, using labels', () => {
    setup({ rows: ROWS, canSave: true });
    const review = screen.getByRole('button', { name: 'Review' });
    expect(review).toHaveAttribute('aria-expanded', 'false');
    const panelId = review.getAttribute('aria-controls')!;
    const panel = document.getElementById(panelId)!;
    expect(panel).not.toBeVisible();

    fireEvent.click(review);
    expect(review).toHaveAttribute('aria-expanded', 'true');
    expect(panel).toBeVisible();
    const items = screen.getAllByTestId('save-bar-diff-row');
    expect(items).toHaveLength(4);
    expect(items[0]).toHaveAttribute('data-diff-key', 'data_parallel_size');
    expect(items[0]).toHaveTextContent('Data-parallel replicas');
    expect(items[0]).toHaveTextContent('4');
    expect(items[0]).toHaveTextContent('2');
    expect(items[0]).toHaveTextContent('next load');
    expect(items[3]).toHaveTextContent('live');

    fireEvent.click(review);
    expect(review).toHaveAttribute('aria-expanded', 'false');
    expect(panel).not.toBeVisible();
  });

  it('Save label: "Save", "Saving…", and "Save — applies now" on a loaded model', () => {
    const a = setup({ rows: ROWS, canSave: true });
    expect(screen.getByTestId('settings-save')).toHaveAccessibleName('Save');
    a.unmount();
    const b = setup({ rows: ROWS, saving: true });
    expect(screen.getByTestId('settings-save')).toHaveAccessibleName('Saving…');
    expect(summary()).toHaveTextContent('Saving…');
    b.unmount();
    setup({ rows: [ROWS[3]], isLoaded: true, canSave: true });
    expect(screen.getByTestId('settings-save')).toHaveAccessibleName('Save — applies now');
  });

  it('a loaded model with a next-load change never promises "applies now"', () => {
    setup({ rows: ROWS, isLoaded: true, canSave: false });
    expect(screen.getByTestId('settings-save')).toHaveAccessibleName('Save');
  });

  it('Fix runs onFix before focusing, so a collapsed target can be revealed first', () => {
    const order: string[] = [];
    const input = document.body.appendChild(document.createElement('input'));
    input.id = 'dp';
    input.addEventListener('focus', () => order.push('focus'));
    setup({
      rows: ROWS,
      invalidReason: 'replicas must divide the 4 selected GPUs',
      fixTargetId: 'dp',
      onFix: () => order.push('onFix'),
    });
    fireEvent.click(screen.getByRole('button', { name: 'Fix' }));
    expect(order).toEqual(['onFix', 'focus']);
    input.remove();
  });

  it('clicking Save / Reset calls the handlers', () => {
    const { onSave, onReset } = setup({ rows: ROWS, canSave: true });
    fireEvent.click(screen.getByTestId('settings-save'));
    fireEvent.click(screen.getByTestId('settings-reset'));
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it('Reset is disabled while saving', () => {
    setup({ rows: ROWS, saving: true });
    expect(screen.getByTestId('settings-reset')).toBeDisabled();
  });

  it('save error renders inside the bar as an alert', () => {
    setup({ rows: ROWS, canSave: true, saveError: 'max_model_len: too large' });
    const err = screen.getByTestId('settings-save-error');
    expect(err).toHaveAttribute('role', 'alert');
    expect(err).toHaveTextContent('max_model_len: too large');
    expect(screen.getByTestId('settings-save-bar')).toContainElement(err);
  });

  it('no alert at all without a save error (the page owns the single alert)', () => {
    setup({ rows: ROWS, invalidReason: 'select at least one GPU' });
    expect(screen.queryByRole('alert')).toBeNull();
  });

  describe('⌘S / Ctrl+S', () => {
    it('calls onSave when canSave and prevents the browser save dialog', () => {
      const { onSave } = setup({ rows: ROWS, canSave: true });
      const ev = new KeyboardEvent('keydown', { key: 's', metaKey: true, bubbles: true, cancelable: true });
      window.dispatchEvent(ev);
      expect(ev.defaultPrevented).toBe(true);
      expect(onSave).toHaveBeenCalledTimes(1);

      const ev2 = new KeyboardEvent('keydown', { key: 'S', ctrlKey: true, bubbles: true, cancelable: true });
      document.body.dispatchEvent(ev2);
      expect(ev2.defaultPrevented).toBe(true);
      expect(onSave).toHaveBeenCalledTimes(2);
    });

    it('prevents default but does not save when !canSave', () => {
      const { onSave } = setup({ rows: ROWS, canSave: false });
      const ev = new KeyboardEvent('keydown', { key: 's', ctrlKey: true, bubbles: true, cancelable: true });
      window.dispatchEvent(ev);
      expect(ev.defaultPrevented).toBe(true);
      expect(onSave).not.toHaveBeenCalled();
    });

    it('ignores plain "s" and other shortcuts', () => {
      const { onSave } = setup({ rows: ROWS, canSave: true });
      const plain = new KeyboardEvent('keydown', { key: 's', bubbles: true, cancelable: true });
      window.dispatchEvent(plain);
      const shifted = new KeyboardEvent('keydown', { key: 's', metaKey: true, shiftKey: true, bubbles: true, cancelable: true });
      window.dispatchEvent(shifted);
      expect(plain.defaultPrevented).toBe(false);
      expect(shifted.defaultPrevented).toBe(false);
      expect(onSave).not.toHaveBeenCalled();
    });

    it('uses the latest props without re-subscribing, and unsubscribes on unmount', () => {
      const add = vi.spyOn(window, 'addEventListener');
      const remove = vi.spyOn(window, 'removeEventListener');
      const first = vi.fn();
      const second = vi.fn();
      const base: SaveBarProps = {
        rows: ROWS, invalidReason: null, saving: false, canSave: false, isLoaded: false,
        saveError: null, onSave: first, onReset: vi.fn(),
      };
      const { rerender, unmount } = render(<SaveBar {...base} />);
      rerender(<SaveBar {...base} canSave onSave={second} />);
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', metaKey: true, cancelable: true }));
      expect(first).not.toHaveBeenCalled();
      expect(second).toHaveBeenCalledTimes(1);
      expect(add.mock.calls.filter((c) => c[0] === 'keydown')).toHaveLength(1);
      unmount();
      expect(remove.mock.calls.filter((c) => c[0] === 'keydown')).toHaveLength(1);
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', metaKey: true, cancelable: true }));
      expect(second).toHaveBeenCalledTimes(1);
      add.mockRestore();
      remove.mockRestore();
    });

    it('advertises the shortcut on the Save button', () => {
      setup({ rows: ROWS, canSave: true });
      expect(screen.getByTestId('settings-save')).toHaveAttribute('aria-keyshortcuts', 'Meta+S Control+S');
    });
  });
});
