// Settings redesign T3 (plan §4.5): the effective command panel, extracted
// from the settings page. One flag per line, the operator's own extra args
// tinted, a stale note while the draft is dirty, and never a horizontal
// scroll.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, act, waitFor, fireEvent } from "@testing-library/react";
import { SWRConfig } from "swr";
import { EffectiveArgvPanel } from "@/components/models/settings/effective-argv-panel";
import { setAccessToken, setCsrfToken } from "@/lib/auth-fetch";

const ARGV = [
  "vllm",
  "serve",
  "--model",
  "Qwen/Qwen3-4B",
  "--port",
  "10000",
  "--enable-prefix-caching",
  "--max-num-seqs",
  "32",
];

function stubArgv(body: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

async function renderPanel(props: Partial<Parameters<typeof EffectiveArgvPanel>[0]> = {}) {
  const r = render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <EffectiveArgvPanel modelId="abc" {...props} />
    </SWRConfig>,
  );
  await act(async () => {
    await new Promise((res) => setTimeout(res, 0));
  });
  return r;
}

beforeEach(() => {
  setAccessToken("test-jwt");
  setCsrfToken("test-csrf");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("EffectiveArgvPanel", () => {
  it("puts each flag on its own line with its value", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel();
    const lines = (await screen.findAllByTestId("effective-argv-line")).map(
      (l) => l.textContent,
    );
    expect(lines).toEqual([
      "vllm serve",
      "--model Qwen/Qwen3-4B",
      "--port 10000",
      "--enable-prefix-caching",
      "--max-num-seqs 32",
    ]);
    // The pre carries real newlines so a copied selection keeps its shape.
    expect(screen.getByTestId("effective-argv-pre").textContent).toBe(
      lines.join("\n"),
    );
  });

  it("keeps one token span per argv entry", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel();
    await screen.findByTestId("effective-argv-pre");
    expect(screen.getAllByTestId("effective-argv-tok")).toHaveLength(ARGV.length);
  });

  it("classifies tokens: binary, flags, values", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel();
    const toks = await screen.findAllByTestId("effective-argv-tok");
    const kinds = toks.map((t) => t.getAttribute("data-kind"));
    expect(kinds.slice(0, 4)).toEqual(["bin", "bin", "flag", "val"]);
  });

  it("tints the trailing tokens that equal the saved extra args", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel({
      extraArgs: ["--enable-prefix-caching", "--max-num-seqs", "32"],
    });
    const toks = await screen.findAllByTestId("effective-argv-tok");
    const yours = toks
      .filter((t) => t.getAttribute("data-kind") === "yours")
      .map((t) => t.textContent);
    expect(yours).toEqual(["--enable-prefix-caching", "--max-num-seqs", "32"]);
    expect(screen.getByTestId("effective-argv-legend").textContent).toMatch(
      /your extra args/i,
    );
  });

  it("tints extra args stored as one 'flag value' string", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel({ extraArgs: ["--max-num-seqs 32"] });
    const toks = await screen.findAllByTestId("effective-argv-tok");
    expect(
      toks.filter((t) => t.getAttribute("data-kind") === "yours").map((t) => t.textContent),
    ).toEqual(["--max-num-seqs", "32"]);
  });

  it("does not tint the llama.cpp flags the engine fields own as 'yours'", async () => {
    // --flash-attn / --cache-type-* live in extra_args but are set by the
    // llama.cpp engine controls, so they are "from the fields".
    stubArgv({
      argv: ["llama-server", "--model", "/m.gguf", "--foo", "1", "--flash-attn", "on", "--cache-type-k", "q8_0"],
    });
    await renderPanel({ extraArgs: ["--foo", "1", "--flash-attn", "on", "--cache-type-k", "q8_0"] });
    const toks = await screen.findAllByTestId("effective-argv-tok");
    expect(
      toks.filter((t) => t.getAttribute("data-kind") === "yours").map((t) => t.textContent),
    ).toEqual(["--foo", "1"]);
    const fa = toks.find((t) => t.textContent === "--flash-attn")!;
    expect(fa.getAttribute("data-kind")).toBe("flag");
    expect(toks.find((t) => t.textContent === "on")!.getAttribute("data-kind")).toBe("val");
  });

  it("does not tint anything when the extra args are not the argv tail", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel({ extraArgs: ["--trust-remote-code"] });
    const toks = await screen.findAllByTestId("effective-argv-tok");
    expect(toks.some((t) => t.getAttribute("data-kind") === "yours")).toBe(false);
    expect(screen.queryByTestId("effective-argv-legend")).toBeNull();
  });

  it("shows the stale note only when the draft is dirty", async () => {
    stubArgv({ argv: ARGV });
    const { rerender } = await renderPanel();
    await screen.findByTestId("effective-argv-pre");
    expect(screen.queryByTestId("effective-argv-stale")).toBeNull();
    rerender(
      <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
        <EffectiveArgvPanel modelId="abc" stale />
      </SWRConfig>,
    );
    expect(
      (await screen.findByTestId("effective-argv-stale")).textContent,
    ).toMatch(/unsaved changes are not in this command yet/i);
  });

  it("wraps instead of scrolling horizontally", async () => {
    stubArgv({ argv: ARGV });
    await renderPanel();
    const pre = await screen.findByTestId("effective-argv-pre");
    expect(pre.className).not.toMatch(/overflow-x/);
    expect(pre.className).toMatch(/whitespace-pre-wrap/);
    expect(pre.className).toMatch(/\[overflow-wrap:anywhere\]/);
  });

  it("keeps the 'whole command' subtitle and the copy button", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn(async () => undefined) },
    });
    Object.defineProperty(window, "isSecureContext", {
      configurable: true,
      writable: true,
      value: true,
    });
    stubArgv({ argv: ARGV });
    await renderPanel();
    expect(screen.getByTestId("effective-argv-subtitle").textContent).toMatch(/whole/i);
    expect(screen.getByRole("heading", { name: "Effective command" })).toBeInTheDocument();
    const copy = screen.getByTestId("effective-argv-copy");
    await waitFor(() => expect(copy).not.toBeDisabled());
    fireEvent.click(copy);
    await waitFor(() => expect(copy.textContent).toMatch(/Copied/));
  });

  it("renders the error state", async () => {
    stubArgv({ detail: "engine offline" }, 500);
    await renderPanel();
    expect(await screen.findByTestId("effective-argv-error")).toBeInTheDocument();
  });
});
