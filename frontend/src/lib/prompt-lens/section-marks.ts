// Per-section trimmed / cut / dropped marks for a Prompt Lens section box,
// read from Task 9's store. `trimmedOf` sums a request's clips; this splits
// them by section kind, as each section header shows its own.
//
// - trimmed: chars of the section the dock no longer holds. Text sections drop
//   their head past CLIENT_SECTION_CHARS (`Section.clipped`, which also counts
//   what the server had already dropped when a section was fetched); the
//   prompt drops its oldest messages (`promptClipped`); tool calls cut the
//   tail of their arguments (`toolCallsClipped`) and add what the server
//   dropped (`tool_call_chars_dropped`, G-R5).
// - elided: the server cut the capture (a fragment or message marked elided,
//   or tool calls dropped whole).
// - stripped: the dock dropped this request's text to stay under its total
//   memory bound.

import type { RequestView } from './store';

export type SectionKind = 'thinking' | 'reply' | 'tool_calls' | 'tool_results' | 'prompt' | 'parts';

export interface SectionMarkValues {
  trimmed: number;
  elided: boolean;
  stripped: boolean;
}

const isToolResult = (role: string) => role === 'tool' || role === 'function';

export function sectionMarks(r: RequestView, kind: SectionKind): SectionMarkValues {
  const stripped = r.stripped;
  const msgs = r.prompt?.messages ?? [];
  switch (kind) {
    case 'thinking':
      return { trimmed: r.thinking.clipped, elided: false, stripped };
    case 'reply':
      return { trimmed: r.reply.clipped, elided: false, stripped };
    case 'tool_calls':
      return {
        trimmed: r.toolCallsClipped + (r.end?.tool_call_chars_dropped ?? 0),
        elided: r.toolCalls.some((c) => c.elided) || (r.end?.tool_calls_dropped ?? 0) > 0,
        stripped,
      };
    case 'tool_results':
      return { trimmed: 0, elided: msgs.some((m) => isToolResult(m.role) && m.elided), stripped };
    case 'prompt':
      return {
        trimmed: r.promptClipped,
        elided: r.promptElided || !!r.prompt?.sys?.elided || msgs.some((m) => m.elided),
        stripped,
      };
    case 'parts':
      return { trimmed: 0, elided: false, stripped: false };
  }
}
