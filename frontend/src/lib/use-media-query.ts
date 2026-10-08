import { useSyncExternalStore } from "react";

/** Subscribe to a CSS media query. False on the server and where matchMedia is
 *  missing, so the desktop layout is the default and nothing mismatches on
 *  hydration. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (notify) => {
      if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return () => {};
      }
      const mq = window.matchMedia(query);
      mq.addEventListener?.("change", notify);
      return () => mq.removeEventListener?.("change", notify);
    },
    () =>
      typeof window !== "undefined" && typeof window.matchMedia === "function"
        ? window.matchMedia(query).matches
        : false,
    () => false,
  );
}

/** Below Tailwind's `lg` breakpoint (1024px): the table needs ~56rem plus page padding, so tablets get cards too. */
export const MOBILE_QUERY = "(max-width: 1023px)";
