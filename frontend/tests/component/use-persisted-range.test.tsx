import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";
import { usePersistedRange } from "@/lib/use-persisted-range";

// localStorage is provided by jsdom but persists across tests within the
// same worker — wipe before each case so writes from one don't bleed.
beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
});

describe("usePersistedRange", () => {
  it("returns the fallback when storage is empty", () => {
    const { result } = renderHook(() => usePersistedRange("vw.test.range", "6h"));
    expect(result.current[0]).toBe("6h");
  });

  it("defaults the fallback to '1h' when omitted", () => {
    const { result } = renderHook(() => usePersistedRange("vw.test.range"));
    expect(result.current[0]).toBe("1h");
  });

  it("hydrates from localStorage when a valid value is stored", () => {
    window.localStorage.setItem("vw.test.range", "24h");
    const { result } = renderHook(() => usePersistedRange("vw.test.range", "1h"));
    // useEffect runs synchronously in jsdom under @testing-library, so
    // the hydrated value is available on the second render — renderHook
    // already flushes effects before returning.
    expect(result.current[0]).toBe("24h");
  });

  it("ignores invalid stored values and falls back", () => {
    // Someone tampered with DevTools, or an old key format leaked in.
    window.localStorage.setItem("vw.test.range", "lol-not-a-range");
    const { result } = renderHook(() => usePersistedRange("vw.test.range", "1h"));
    expect(result.current[0]).toBe("1h");
  });

  it("writes to localStorage when setValue is called", () => {
    const { result } = renderHook(() => usePersistedRange("vw.test.range", "1h"));
    act(() => {
      result.current[1]("7d");
    });
    expect(result.current[0]).toBe("7d");
    expect(window.localStorage.getItem("vw.test.range")).toBe("7d");
  });

  it("updates state on setValue even if storage write throws", () => {
    // Stub setItem to simulate quota-exceeded / privacy-mode rejection.
    const original = window.localStorage.setItem;
    window.localStorage.setItem = () => {
      throw new Error("quota exceeded");
    };
    try {
      const { result } = renderHook(() => usePersistedRange("vw.test.range", "1h"));
      act(() => {
        result.current[1]("6h");
      });
      // State must still flip — the user picked it, the UI must reflect it.
      expect(result.current[0]).toBe("6h");
    } finally {
      window.localStorage.setItem = original;
    }
  });

  it("re-hydrates when storageKey changes", () => {
    window.localStorage.setItem("vw.test.a", "1h");
    window.localStorage.setItem("vw.test.b", "7d");
    const { result, rerender } = renderHook(
      ({ key }: { key: string }) => usePersistedRange(key, "1h"),
      { initialProps: { key: "vw.test.a" } },
    );
    expect(result.current[0]).toBe("1h");
    rerender({ key: "vw.test.b" });
    expect(result.current[0]).toBe("7d");
  });
});
