import { describe, it, expect } from 'vitest';
import {
  splitManagedArgs,
  mergeManagedArgs,
  setManagedArg,
  LLAMACPP_MANAGED_FLAGS,
} from '@/lib/llamacpp-args';

// ---------------------------------------------------------------------------
// llama.cpp's important dials had no control — flash attention and the two KV
// cache types could only be reached by hand-typing into Extra args. They are
// now controls, and the controls store into `extra_args` rather than into new
// columns.
//
// The property that makes that safe, and the whole reason this module exists as
// a pure function with tests: a value set in the control must never DUPLICATE
// or CONFLICT with the same flag typed by hand. There is exactly one place each
// managed flag can live, because the managed flags are extracted out of the
// free-text list on load.
// ---------------------------------------------------------------------------

describe('splitManagedArgs', () => {
  it('leaves an unmanaged flag alone', () => {
    expect(splitManagedArgs(['--threads', '8'])).toEqual({
      managed: {},
      rest: ['--threads', '8'],
    });
  });

  it('extracts a two-token managed flag', () => {
    expect(splitManagedArgs(['--flash-attn', 'on'])).toEqual({
      managed: { flash_attn: 'on' },
      rest: [],
    });
  });

  it('extracts the short alias too', () => {
    // `-fa` and `-ctk` are what an operator following upstream docs will type.
    // Reading only the long form would leave the flag in the box AND let the
    // control append a second copy.
    expect(splitManagedArgs(['-fa', 'off']).managed).toEqual({ flash_attn: 'off' });
    expect(splitManagedArgs(['-ctk', 'q8_0']).managed).toEqual({
      cache_type_k: 'q8_0',
    });
  });

  it('extracts the --flag=value spelling', () => {
    expect(splitManagedArgs(['--cache-type-v=q4_0'])).toEqual({
      managed: { cache_type_v: 'q4_0' },
      rest: [],
    });
  });

  it('keeps unmanaged args in their original order', () => {
    const { rest } = splitManagedArgs([
      '--threads', '8', '--flash-attn', 'on', '--ubatch-size', '256',
    ]);
    expect(rest).toEqual(['--threads', '8', '--ubatch-size', '256']);
  });

  it('leaves a flag with an unrecognised value exactly where it was', () => {
    // The control offers a fixed list. Adopting a value it cannot display
    // would mean either dropping the operator's setting or rendering a select
    // with no matching option -- both worse than not touching a flag we do not
    // understand.
    const args = ['--cache-type-k', 'q3_K_M'];
    expect(splitManagedArgs(args)).toEqual({ managed: {}, rest: args });
  });

  it('leaves a dangling flag with no value alone', () => {
    expect(splitManagedArgs(['--flash-attn'])).toEqual({
      managed: {},
      rest: ['--flash-attn'],
    });
  });

  it('reads the LAST of a repeated flag, matching what llama-server honours', () => {
    expect(splitManagedArgs(['--flash-attn', 'on', '-fa', 'off']).managed).toEqual({
      flash_attn: 'off',
    });
  });
});

describe('mergeManagedArgs', () => {
  it('omits a flag with no value, so "unset" stays reachable', () => {
    // Writing out the engine's own default instead would change the launch
    // command for every existing row, and a default you cannot return to is
    // precisely the settings bug this codebase keeps finding.
    expect(mergeManagedArgs(['--threads', '8'], {})).toEqual(['--threads', '8']);
  });

  it('appends managed flags after the free-text ones', () => {
    expect(mergeManagedArgs(['--threads', '8'], { flash_attn: 'on' })).toEqual([
      '--threads', '8', '--flash-attn', 'on',
    ]);
  });

  it('writes the long spelling even when the short one was typed', () => {
    const { managed, rest } = splitManagedArgs(['-fa', 'on']);
    expect(mergeManagedArgs(rest, managed)).toEqual(['--flash-attn', 'on']);
  });

  it('is stable, so a save with no edits produces no diff', () => {
    const args = ['--threads', '8', '--flash-attn', 'on', '--cache-type-k', 'q8_0'];
    const { managed, rest } = splitManagedArgs(args);
    const once = mergeManagedArgs(rest, managed);
    const twice = (() => {
      const s = splitManagedArgs(once);
      return mergeManagedArgs(s.rest, s.managed);
    })();
    expect(twice).toEqual(once);
  });

  it('drops a value the flag does not accept', () => {
    expect(mergeManagedArgs([], { cache_type_k: 'not-a-type' })).toEqual([]);
  });
});

describe('setManagedArg — no duplication, ever', () => {
  it('replaces rather than appends when the flag is already present', () => {
    // The bug this function exists to make impossible: a control that appended
    // `--flash-attn on` while `--flash-attn off` sat in the free-text box would
    // hand llama-server both, and it honours the last.
    const out = setManagedArg(['--flash-attn', 'off'], 'flash_attn', 'on');
    expect(out).toEqual(['--flash-attn', 'on']);
    expect(out.filter((t) => t === '--flash-attn')).toHaveLength(1);
  });

  it('replaces a hand-typed short alias with one canonical flag', () => {
    const out = setManagedArg(['-fa', 'off'], 'flash_attn', 'on');
    expect(out).toEqual(['--flash-attn', 'on']);
  });

  it('clearing a control removes the flag entirely', () => {
    expect(setManagedArg(['--flash-attn', 'on'], 'flash_attn', undefined)).toEqual(
      [],
    );
  });

  it('never touches the operator\'s other args', () => {
    const out = setManagedArg(
      ['--threads', '8', '--ubatch-size', '256'],
      'cache_type_k',
      'q8_0',
    );
    expect(out.slice(0, 4)).toEqual(['--threads', '8', '--ubatch-size', '256']);
    expect(out).toContain('--cache-type-k');
  });

  it('leaves extra_args untouched for a key it does not manage', () => {
    expect(setManagedArg(['--threads', '8'], 'not_a_flag', 'x')).toEqual([
      '--threads', '8',
    ]);
  });

  it('does not mutate its input', () => {
    const args = ['--flash-attn', 'off'];
    setManagedArg(args, 'flash_attn', 'on');
    expect(args).toEqual(['--flash-attn', 'off']);
  });
});

describe('the flag catalogue matches the shipped binary', () => {
  // Captured from `/opt/llamacpp/llama-server --help` in the api container on
  // `bonus` (b10731, 0eadefebd) -- the binary, not upstream docs, because
  // several of these spellings have changed there.
  it('treats --flash-attn as taking a value, not as a bare switch', () => {
    const fa = LLAMACPP_MANAGED_FLAGS.find((f) => f.key === 'flash_attn')!;
    expect(fa.values).toEqual(['on', 'off', 'auto']);
  });

  it('offers exactly the KV cache types the build accepts', () => {
    const expected = [
      'f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1',
    ];
    for (const key of ['cache_type_k', 'cache_type_v']) {
      const spec = LLAMACPP_MANAGED_FLAGS.find((f) => f.key === key)!;
      expect(spec.values).toEqual(expected);
    }
  });

  it('does not manage split-mode or tensor-split', () => {
    // Decision D4 keeps the GPU COUNT the invariant and lets plan() pick the
    // backend-appropriate flag; llama.cpp derives --split-mode and --main-gpu
    // from the GPU selection. A control here would be a second, competing
    // source for the same fact.
    const keys = LLAMACPP_MANAGED_FLAGS.map((f) => f.key);
    expect(keys).not.toContain('split_mode');
    expect(keys).not.toContain('tensor_split');
  });
});
