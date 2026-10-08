/** The forest token store (Task 8): the expiry goes in first, a failed write leaves nothing, a missing expiry is corrupt. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { forestTokenState, readForestToken, storeForestToken } from "@/lib/forest/token";

/** A Map-backed sessionStorage whose setItem can fail for one key (jsdom's Storage cannot be spied on). */
function fakeStorage(failing?: string) {
  const m = new Map<string, string>();
  const order: string[] = [];
  const fake = {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => {
      order.push(k);
      if (k === failing) throw new DOMException("quota", "QuotaExceededError");
      m.set(k, String(v));
    },
    removeItem: (k: string) => void m.delete(k),
    clear: () => m.clear(),
  };
  vi.stubGlobal("sessionStorage", fake);
  return { m, order };
}

afterEach(() => {
  vi.unstubAllGlobals();
  sessionStorage.clear();
});

describe("storeForestToken", () => {
  it("writes the expiry first, then the token, and reports success", () => {
    const { order } = fakeStorage();
    expect(storeForestToken("jwt", 60)).toBe(true);
    expect(order).toEqual(["vw-forest-exp", "vw-forest-token"]);
    expect(readForestToken()).toBe("jwt");
  });

  it.each(["vw-forest-exp", "vw-forest-token"])("a failure writing %s leaves neither key", (failing) => {
    const { m } = fakeStorage(failing);
    expect(storeForestToken("jwt", 60)).toBe(false);
    expect([...m.keys()]).toEqual([]);
  });
});

describe("forestTokenState", () => {
  it("a token with no expiry is corrupt: expired, and cleared", () => {
    sessionStorage.setItem("vw-forest-token", "jwt");
    expect(forestTokenState()).toBe("expired");
    expect(sessionStorage.getItem("vw-forest-token")).toBeNull();
    expect(readForestToken()).toBeNull();
  });

  it("an unreadable or passed expiry is expired; a future one is valid", () => {
    sessionStorage.setItem("vw-forest-token", "jwt");
    sessionStorage.setItem("vw-forest-exp", "soon");
    expect(forestTokenState()).toBe("expired");
    sessionStorage.setItem("vw-forest-token", "jwt");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() - 1));
    expect(forestTokenState()).toBe("expired");
    sessionStorage.setItem("vw-forest-token", "jwt");
    sessionStorage.setItem("vw-forest-exp", String(Date.now() + 60_000));
    expect(forestTokenState()).toBe("valid");
  });
});
