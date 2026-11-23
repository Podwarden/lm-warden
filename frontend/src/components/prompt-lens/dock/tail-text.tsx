"use client";

// Streaming text for a Prompt Lens section box, written to the DOM by hand so
// a delta costs O(delta), never a re-render of the whole section (a 200k-char
// thinking section at ~50 deltas/s). React renders an empty container once;
// a layout effect then appends the new tail and drops the clipped head.
//
// The text sits in blocks of about TAIL_CHUNK_CHARS, split only after a
// newline, so the browser re-lays out the last block on an append, not the
// whole section. A line with no newline stays in one block and wraps.
//
// `offset` is how many chars of the section were dropped before `text`
// (the store's `Section.clipped`). When the head is dropped, the reader's
// scroll position is moved up by the height removed (through the section
// box's FollowTailContext), so the text under the reader does not move.
//
// The update is incremental when the new text continues the old one: the
// same section moved forward by `offset` and grew at the end. A short sample
// at the old end is checked; anything else (a section fill, another request)
// rebuilds the blocks.

import { useContext, useLayoutEffect, useRef } from "react";

import { cn } from "@/lib/utils";
import { FollowTailContext } from "./use-follow-tail";

/** A block is closed at the first newline once it holds this many chars. */
export const TAIL_CHUNK_CHARS = 4096;
const SAMPLE = 64;

interface Shown {
  offset: number;
  len: number;
  /** The last SAMPLE chars shown. */
  tail: string;
}

function lastText(block: Node): Text {
  return block.firstChild as Text;
}

function newBlock(doc: Document): HTMLDivElement {
  const d = doc.createElement("div");
  d.appendChild(doc.createTextNode(""));
  return d;
}

/** Append `s` to the last block, opening new blocks after newlines. */
function append(root: HTMLElement, s: string) {
  const doc = root.ownerDocument;
  let rest = s;
  while (rest) {
    let block = root.lastChild;
    if (!block) {
      block = newBlock(doc);
      root.appendChild(block);
    }
    const t = lastText(block);
    const room = TAIL_CHUNK_CHARS - t.length;
    if (room > 0) {
      // Fill to the chunk size, then up to the next newline.
      const nl = rest.indexOf("\n", Math.min(room, rest.length));
      const take = nl === -1 ? rest.length : nl + 1;
      t.appendData(rest.slice(0, take));
      rest = rest.slice(take);
    } else if (t.data.endsWith("\n")) {
      root.appendChild(newBlock(doc));
    } else {
      const nl = rest.indexOf("\n");
      const take = nl === -1 ? rest.length : nl + 1;
      t.appendData(rest.slice(0, take));
      rest = rest.slice(take);
    }
  }
}

/** Remove `n` chars from the head. */
function dropHead(root: HTMLElement, n: number) {
  let left = n;
  while (left > 0 && root.firstChild) {
    const t = lastText(root.firstChild);
    if (t.length <= left) {
      left -= t.length;
      root.removeChild(root.firstChild);
    } else {
      t.deleteData(0, left);
      left = 0;
    }
  }
}

/** Does `text` (starting at global `offset`) hold `tail` (ending at global
 *  `prevEnd`) where it should, as far as both still overlap? */
function overlapMatches(text: string, offset: number, prevEnd: number, tail: string): boolean {
  const kept = Math.min(tail.length, prevEnd - offset);
  if (kept <= 0) return true;
  const at = prevEnd - offset - kept;
  return text.slice(at, at + kept) === tail.slice(tail.length - kept);
}

function scroller(el: HTMLElement): HTMLElement | null {
  let p = el.parentElement;
  while (p && p.dataset.scrollBox === undefined) p = p.parentElement;
  return p;
}

export interface TailTextProps {
  text: string;
  /** Chars of the section dropped before `text`. */
  offset?: number;
  className?: string;
}

export function TailText({ text, offset = 0, className }: TailTextProps) {
  const ref = useRef<HTMLDivElement>(null);
  const shown = useRef<Shown | null>(null);
  const follow = useContext(FollowTailContext);

  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    const prev = shown.current;
    const end = offset + text.length;
    const next: Shown = { offset, len: text.length, tail: text.slice(-SAMPLE) };
    if (prev && prev.offset === offset && prev.len === text.length && prev.tail === next.tail) return;

    const prevEnd = prev ? prev.offset + prev.len : 0;
    const continues =
      prev !== null &&
      offset >= prev.offset &&
      offset <= prevEnd &&
      end >= prevEnd &&
      overlapMatches(text, offset, prevEnd, prev.tail);

    if (!continues) {
      root.textContent = "";
      append(root, text);
    } else {
      const drop = offset - prev.offset;
      if (drop > 0) {
        const box = scroller(root);
        const before = box?.scrollHeight ?? 0;
        dropHead(root, drop);
        if (box) {
          const dy = box.scrollHeight - before;
          if (dy) {
            if (follow) follow.nudge(dy);
            else box.scrollTop += dy;
          }
        }
      }
      if (end > prevEnd) append(root, text.slice(prevEnd - offset));
    }
    shown.current = next;
  });

  return (
    <div
      ref={ref}
      data-testid="tail-text"
      className={cn("whitespace-pre-wrap [overflow-wrap:anywhere]", className)}
    />
  );
}
