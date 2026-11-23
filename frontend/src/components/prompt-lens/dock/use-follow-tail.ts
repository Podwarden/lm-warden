"use client";

// Follow-the-tail rule for Prompt Lens section boxes (spec 2026-11-22 §3.3,
// mockup states 8 and 9). While following, every content change pins the box
// to the bottom. The reader scrolling up (more than FOLLOW_SLACK_PX from the
// bottom) stops following; the box then stays where it is while text keeps
// arriving, and `newChars` counts what arrived. `jump()` or scrolling back to
// the bottom by hand resumes.
//
// What keeps it free of jitter at ~50 deltas/s:
// - Programmatic scrolls are marked with the scrollTop they produced, and the
//   scroll event that reports exactly that position is ignored. A position set
//   that changes nothing marks nothing (no event will come for it).
// - A scroll event only stops following when scrollTop went DOWN since the
//   last known position. Text landing between a pin and its event leaves the
//   box short of the bottom without moving scrollTop, so it never counts.
// - Wheel up, a finger pulling the text down and PageUp/ArrowUp/Home stop
//   following at once, before the browser's (later) scroll event, so a delta
//   that lands in between cannot pin the box back down under the reader.
// - A width change (the text re-wraps) re-pins while following, and otherwise
//   keeps the reader at the same fraction of the text.
//
// Not the viewer's older `shared/use-sticky-bottom.ts`: that one never
// resumes by hand and leaves the scrolling to Virtuoso.

import { createContext, useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";

/** Distance from the bottom, in px, that still counts as "at the bottom". */
export const FOLLOW_SLACK_PX = 24;

export interface FollowTail {
  following: boolean;
  /** Chars added while not following (reset when following resumes). */
  newChars: number;
  /** Resume following and scroll to the bottom. */
  jump: () => void;
  /** Stop following (the pop-up's current search hit). */
  stop: () => void;
  /** Report `n` chars added to the content. Counted only while not following. */
  noteChars: (n: number) => void;
  /** Move the content by `dy` px without it counting as the reader scrolling
   *  (a head trim above the reader). */
  nudge: (dy: number) => void;
}

/** Lets content inside a box (TailText) keep the reader in place when it
 *  removes text above them. Null outside a box. */
export const FollowTailContext = createContext<Pick<FollowTail, "nudge"> | null>(null);

const UP_KEYS = new Set(["PageUp", "ArrowUp", "Home"]);

export function useFollowTail(
  ref: RefObject<HTMLElement | null>,
  contentKey: unknown,
  live: boolean,
  resetKey: string,
): FollowTail {
  const [following, setFollowingState] = useState(true);
  const [newChars, setNewChars] = useState(0);
  const followingRef = useRef(true);
  const liveRef = useRef(live);
  liveRef.current = live;
  /** The scrollTop our last programmatic scroll produced, until its event. */
  const expected = useRef<number | null>(null);
  /** Last scrollTop seen, to tell a scroll up from content growing. */
  const lastTop = useRef(0);
  /** Last scrollHeight seen, for a width change (re-wrap). */
  const lastHeight = useRef(0);
  const lastWidth = useRef<number | null>(null);

  const setFollowing = useCallback((v: boolean) => {
    followingRef.current = v;
    setFollowingState(v);
    if (v) setNewChars(0);
  }, []);

  const setTop = useCallback(
    (el: HTMLElement, top: number) => {
      const before = el.scrollTop;
      el.scrollTop = top;
      const after = el.scrollTop;
      if (after !== before) expected.current = after;
      lastTop.current = after;
      lastHeight.current = el.scrollHeight;
    },
    [],
  );

  const pin = useCallback(() => {
    const el = ref.current;
    if (el) setTop(el, el.scrollHeight);
  }, [ref, setTop]);

  // A new request: following again, at the bottom.
  const firstReset = useRef(true);
  useLayoutEffect(() => {
    if (firstReset.current) {
      firstReset.current = false;
      return;
    }
    followingRef.current = true;
    setFollowingState(true);
    setNewChars(0);
    expected.current = null;
  }, [resetKey]);

  // After each content change (and on mount / reset), pin while following.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (followingRef.current) pin();
    else lastHeight.current = el.scrollHeight;
  }, [ref, contentKey, resetKey, pin]);

  const stop = useCallback(() => {
    if (followingRef.current) setFollowing(false);
  }, [setFollowing]);

  // Listeners. The element can change (a collapsed box remounts), so check
  // after every render and re-attach when it did.
  const attached = useRef<HTMLElement | null>(null);
  const detach = useRef<(() => void) | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (el === attached.current) return;
    detach.current?.();
    detach.current = null;
    attached.current = el;
    if (!el) return;
    lastTop.current = el.scrollTop;
    lastHeight.current = el.scrollHeight;

    const onScroll = () => {
      const top = el.scrollTop;
      const exp = expected.current;
      expected.current = null;
      const movedUp = top < lastTop.current - 0.5;
      lastTop.current = top;
      lastHeight.current = el.scrollHeight;
      if (exp !== null && Math.abs(top - exp) < 1) return;
      const gap = el.scrollHeight - el.clientHeight - top;
      if (followingRef.current) {
        if (movedUp && gap > FOLLOW_SLACK_PX) setFollowing(false);
      } else if (gap <= FOLLOW_SLACK_PX) {
        setFollowing(true);
      }
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) stop();
    };
    let touchY: number | null = null;
    const onTouchStart = (e: TouchEvent) => {
      touchY = e.touches[0]?.clientY ?? null;
    };
    const onTouchMove = (e: TouchEvent) => {
      const y = e.touches[0]?.clientY;
      if (touchY !== null && y !== undefined && y > touchY + 2) stop();
      if (y !== undefined) touchY = y;
    };
    const onKey = (e: KeyboardEvent) => {
      if (UP_KEYS.has(e.key)) stop();
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("keydown", onKey);

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      lastWidth.current = el.getBoundingClientRect().width;
      ro = new ResizeObserver(() => {
        const w = el.getBoundingClientRect().width;
        const prevW = lastWidth.current;
        lastWidth.current = w;
        if (prevW === null || Math.abs(w - prevW) < 0.5) return;
        if (followingRef.current) {
          setTop(el, el.scrollHeight);
        } else if (lastHeight.current > 0) {
          setTop(el, Math.round((lastTop.current / lastHeight.current) * el.scrollHeight));
        }
      });
      ro.observe(el);
    }
    detach.current = () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("keydown", onKey);
      ro?.disconnect();
    };
  });
  useEffect(
    () => () => {
      detach.current?.();
      detach.current = null;
      attached.current = null;
    },
    [],
  );

  const jump = useCallback(() => {
    setFollowing(true);
    pin();
  }, [setFollowing, pin]);

  const noteChars = useCallback((n: number) => {
    if (n > 0 && !followingRef.current && liveRef.current) setNewChars((c) => c + n);
  }, []);

  const nudge = useCallback(
    (dy: number) => {
      const el = ref.current;
      if (el && dy) setTop(el, el.scrollTop + dy);
    },
    [ref, setTop],
  );

  return { following, newChars, jump, stop, noteChars, nudge };
}
