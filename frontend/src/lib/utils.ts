import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Copy `text` to the system clipboard.
 *
 * Tries the modern async Clipboard API first (`navigator.clipboard.writeText`,
 * gated to secure contexts: HTTPS or `localhost`). When that is unavailable
 * — which is the common case for vllm-warden in production where operators
 * reach the UI over raw HTTP via a VPN or the LAN (e.g. `http://192.0.2.10`) —
 * falls back to the legacy hidden-`<textarea>` + `document.execCommand("copy")`
 * approach so the copy button still works.
 *
 * Issue #149: the previous version of this helper called `execCommand` but
 * silently swallowed its boolean return value (and any synchronous throws
 * from browsers that no longer support the legacy API). Callers therefore
 * had no way to distinguish "copied" from "neither path worked", so they
 * either always showed "copied" or always showed "select manually". The
 * helper now `throw`s when BOTH paths fail, so callers can render the
 * fallback hint only on real failure.
 */
export async function copyToClipboard(text: string): Promise<void> {
  // Prefer the async API when available AND we are in a secure context.
  // The secure-context check matters because Chromium exposes
  // `navigator.clipboard` on `http://` pages too, but `writeText()`
  // rejects with NotAllowedError — using the textarea fallback up front
  // gives a more reliable copy on a LAN-HTTP deployment.
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function" &&
    typeof window !== "undefined" &&
    window.isSecureContext
  ) {
    await navigator.clipboard.writeText(text);
    return;
  }

  // Legacy fallback. `document.execCommand("copy")` is deprecated but
  // remains the only cross-browser path for non-secure contexts. It
  // requires a focused/selected editable element in the DOM to read
  // from, hence the hidden textarea.
  if (typeof document === "undefined") {
    throw new Error("Clipboard not available — no document");
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  // Position offscreen so the textarea never visually flashes, and use
  // `readonly` + tiny opacity to keep mobile keyboards from popping up.
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "0";
  textarea.style.width = "1px";
  textarea.style.height = "1px";
  textarea.style.padding = "0";
  textarea.style.border = "0";
  textarea.style.outline = "0";
  textarea.style.boxShadow = "none";
  textarea.style.background = "transparent";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);

  let ok = false;
  try {
    textarea.focus();
    textarea.select();
    // Some browsers throw on `execCommand` in disallowed contexts
    // instead of returning false — catch both.
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  } finally {
    document.body.removeChild(textarea);
  }

  if (!ok) {
    throw new Error("Clipboard copy failed");
  }
}
