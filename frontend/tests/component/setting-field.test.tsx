import { describe, it, expect, vi, afterEach } from "vitest";
import React from "react";
import { render, screen, fireEvent, cleanup, within } from "@testing-library/react";
import { SettingField } from "@/components/settings/setting-field";
import type { GpuInfo } from "@/components/gpu/gpu-checklist";
import { MODEL_HINTS, RUNTIME_HINTS, type FieldHint } from "@/lib/settings-hints";

afterEach(cleanup);

const GPUS: GpuInfo[] = [
  { index: 0, name: "RTX 4090", memory_total_mib: 24564, memory_used_mib: 1200, utilization_pct: 5 },
  { index: 1, name: "RTX 4090", memory_total_mib: 24564, memory_used_mib: 800, utilization_pct: 0 },
];

// SettingField renders its label/hint/restart chrome from a `field: FieldHint`,
// not a bare `label` prop — so the gpu-set tests supply a FieldHint whose
// `label` carries the visible text the spec asserts on.
const FIELD: FieldHint = {
  label: "Default GPU indices",
  hint: "Pre-selected when adding a new model. Comma-separated GPU IDs.",
  restart: "none",
};

describe("SettingField — gpu-set kind", () => {
  it("renders the label and one checkbox per present GPU", () => {
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0]} gpus={GPUS} onChange={() => {}} />,
    );
    expect(screen.getByText("Default GPU indices")).toBeInTheDocument();
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
  });

  it("emits a sorted number[] when a GPU is toggled on", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0]} gpus={GPUS} onChange={onChange} />,
    );
    fireEvent.click(screen.getByLabelText(/#1/));
    expect(onChange).toHaveBeenCalledWith([0, 1]);
  });

  it("renders a ghost row + alert for an absent configured index and removes it on uncheck", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="gpu-set" field={FIELD} value={[0, 5]} gpus={GPUS} onChange={onChange} />,
    );
    const ghost = screen.getByLabelText(/GPU 5 — not present/);
    expect(ghost).toBeInTheDocument();
    expect(ghost).toBeChecked();
    expect(screen.getByRole("alert")).toBeInTheDocument();
    fireEvent.click(ghost);
    expect(onChange).toHaveBeenCalledWith([0]);
  });
});

// ---------------------------------------------------------------------------
// Regression: freeform-typing branches (int-list / string-list / kv-map)
// must preserve in-progress text. The bug was that the displayed value was
// derived from the parsed result on every keystroke, wiping trailing
// commas/newlines and making multi-value entry impossible.
// ---------------------------------------------------------------------------

const FREEFORM_FIELD: FieldHint = {
  label: "L",
  hint: "h",
  restart: "none",
};

// Stateful harness exercises the full controlled round-trip: SettingField's
// parsed onChange feeds back into the `value` prop, so a buggy "display from
// parsed value" branch would snap the text back on the next render.
function Harness({ kind, initial }: { kind: "int-list" | "string-list" | "kv-map"; initial: unknown }) {
  const [v, setV] = React.useState(initial);
  return (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    <SettingField kind={kind as any} field={FREEFORM_FIELD} value={v as any} onChange={setV as any} />
  );
}

describe("SettingField — freeform typing preserves in-progress text", () => {
  it("int-list: trailing comma and second value survive", () => {
    render(<Harness kind="int-list" initial={[]} />);
    const input = screen.getByRole("textbox") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "0," } });
    expect(input.value).toBe("0,");

    fireEvent.change(input, { target: { value: "0,1" } });
    expect(input.value).toBe("0,1");
  });

  it("string-list: trailing newline and second line survive", () => {
    render(<Harness kind="string-list" initial={[]} />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;

    fireEvent.change(input, { target: { value: "--foo\n" } });
    expect(input.value).toBe("--foo\n");

    fireEvent.change(input, { target: { value: "--foo\n--bar" } });
    expect(input.value).toBe("--foo\n--bar");
  });

  it("kv-map: Enter then a second KEY=value survive", () => {
    render(<Harness kind="kv-map" initial={{}} />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;

    fireEvent.change(input, { target: { value: "A=1\n" } });
    expect(input.value).toBe("A=1\n");

    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(input.value).toBe("A=1\nB=2");
  });

  it("int-list: still emits the parsed number[] upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="int-list" field={FREEFORM_FIELD} value={[]} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "0,1" } });
    expect(onChange).toHaveBeenLastCalledWith([0, 1]);
  });

  it("string-list: still emits the parsed string[] upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="string-list" field={FREEFORM_FIELD} value={[]} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "--foo\n--bar" } });
    expect(onChange).toHaveBeenLastCalledWith(["--foo", "--bar"]);
  });

  it("kv-map: still emits the parsed Record upward", () => {
    const onChange = vi.fn();
    render(
      <SettingField kind="kv-map" field={FREEFORM_FIELD} value={{}} onChange={onChange} />,
    );
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(onChange).toHaveBeenLastCalledWith({ A: "1", B: "2" });
  });

  it("int-list: adopts an external value reset (e.g. model switch)", () => {
    function ResetHarness() {
      const [v, setV] = React.useState<number[]>([]);
      return (
        <div>
          <button onClick={() => setV([3, 4])}>reset</button>
          <SettingField kind="int-list" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ResetHarness />);
    const input = screen.getByRole("textbox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "0," } });
    expect(input.value).toBe("0,");
    fireEvent.click(screen.getByText("reset"));
    expect(input.value).toBe("3,4");
  });

  it("kv-map: adopts an external value reset to different contents", () => {
    function ResetHarness() {
      const [v, setV] = React.useState<Record<string, string>>({ A: "1" });
      return (
        <div>
          <button onClick={() => setV({ C: "3" })}>reset</button>
          <SettingField kind="kv-map" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ResetHarness />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(input.value).toBe("A=1");
    fireEvent.click(screen.getByText("reset"));
    expect(input.value).toBe("C=3");
  });

  it("kv-map: a reordered-but-equal prop update does NOT clobber typed text", () => {
    // Regression for the order-sensitive guard: after a save→refresh the
    // server may echo the same map in a different key order. With a plain
    // string comparison the effect would fire and snap the textarea to the
    // server's order, resetting the cursor. The canonical (sorted) guard
    // must treat reordered-equal contents as a no-op.
    function ReorderHarness() {
      const [v, setV] = React.useState<Record<string, string>>({});
      return (
        <div>
          <button onClick={() => setV({ B: "2", A: "1" })}>echo</button>
          <SettingField kind="kv-map" field={FREEFORM_FIELD} value={v} onChange={setV} />
        </div>
      );
    }
    render(<ReorderHarness />);
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    // User types A then B; onChange emits {A:"1",B:"2"} into the prop.
    fireEvent.change(input, { target: { value: "A=1\nB=2" } });
    expect(input.value).toBe("A=1\nB=2");
    // Server echoes the same contents in reversed key order.
    fireEvent.click(screen.getByText("echo"));
    // Display must stay as the user typed — canonical forms match, no resync.
    expect(input.value).toBe("A=1\nB=2");
  });
});

// ---------------------------------------------------------------------------
// Settings redesign T1 — opt-in props. Every one defaults off, so the global
// settings tabs keep rendering exactly as before.
// ---------------------------------------------------------------------------

const RELOAD_FIELD: FieldHint = {
  label: "dtype",
  hint: "One of `auto`, `float16`, `bfloat16`, `float32`. `auto` follows the model's config.",
  short: "Default follows the model's config.",
  restart: "model-reload",
};

describe("SettingField — default chrome is unchanged", () => {
  it("still shows the restart badge, the long hint and slate label classes", () => {
    render(<SettingField kind="text" field={RELOAD_FIELD} value="" onChange={() => {}} />);
    expect(screen.getByLabelText("requires model-reload")).toBeInTheDocument();
    expect(screen.getByText(RELOAD_FIELD.hint)).toBeInTheDocument();
    expect(screen.queryByText(RELOAD_FIELD.short!)).toBeNull();
    expect(screen.queryByRole("button", { name: /more/i })).toBeNull();
    const label = screen.getByText("dtype");
    expect(label.tagName).toBe("LABEL");
    expect(label.className).toBe("font-medium text-slate-200");
    expect(screen.getByText(RELOAD_FIELD.hint).className).toBe("text-xs text-slate-500");
  });

  it("boolean without a variant keeps the enabled/disabled word and checkbox role", () => {
    const f: FieldHint = { label: "Replica affinity", hint: "h", restart: "none" };
    render(<SettingField kind="boolean" field={f} value={false} onChange={() => {}} />);
    expect(screen.getByRole("checkbox")).toBeInTheDocument();
    expect(screen.queryByRole("switch")).toBeNull();
    expect(screen.getByText("disabled")).toBeInTheDocument();
  });
});

describe("SettingField — chrome=\"token\"", () => {
  it("hides the 'requires model-reload' badge", () => {
    render(
      <SettingField kind="text" field={RELOAD_FIELD} value="" onChange={() => {}} chrome="token" />,
    );
    expect(screen.queryByLabelText(/requires/i)).toBeNull();
    expect(screen.queryByText(/requires model-reload/i)).toBeNull();
  });
});

describe("SettingField — token hint", () => {
  it("renders the short hint and a More toggle that reveals the long one", () => {
    render(
      <SettingField kind="text" field={RELOAD_FIELD} value="" onChange={() => {}} chrome="token" />,
    );
    const input = screen.getByLabelText(/^dtype/i);
    const hintId = input.getAttribute("aria-describedby")!;
    const hint = document.getElementById(hintId)!;
    expect(hint).toHaveTextContent(RELOAD_FIELD.short!);
    expect(screen.getByText(RELOAD_FIELD.hint)).not.toBeVisible();

    const more = screen.getByRole("button", { name: "More" });
    expect(more).toHaveAttribute("aria-expanded", "false");
    // The toggle sits beside the described-by text, never inside it, so a
    // screen reader does not end every description with "More".
    expect(hint).not.toContainElement(more);
    expect(input).toHaveAccessibleDescription(RELOAD_FIELD.short!);

    fireEvent.click(more);
    expect(more).toHaveAttribute("aria-expanded", "true");
    expect(more).toHaveTextContent("Less");
    const long = screen.getByText(RELOAD_FIELD.hint);
    expect(long).toBeVisible();
    expect(more).toHaveAttribute("aria-controls", long.id);

    fireEvent.click(more);
    expect(more).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText(RELOAD_FIELD.hint)).not.toBeVisible();
  });

  it("shows no More toggle when there is no separate short hint", () => {
    const f: FieldHint = { label: "L", hint: "Only one hint.", restart: "none" };
    render(<SettingField kind="text" field={f} value="" onChange={() => {}} chrome="token" />);
    expect(screen.getByText("Only one hint.")).toBeVisible();
    expect(screen.queryByRole("button", { name: /more/i })).toBeNull();
  });

  it("uses token typography instead of slate classes", () => {
    const { container } = render(
      <SettingField kind="text" field={RELOAD_FIELD} value="" onChange={() => {}} chrome="token" />,
    );
    const label = screen.getByText("dtype");
    expect(label.className).toContain("text-chat-fg");
    const hint = document.getElementById(
      screen.getByLabelText(/^dtype/i).getAttribute("aria-describedby")!,
    )!;
    expect(hint.className).toContain("text-chat-muted");
    // The chrome this component owns is slate-free (the shared Input
    // primitive is not this component's to restyle).
    expect(label.outerHTML + hint.outerHTML).not.toMatch(/slate-/);
    expect(container.innerHTML).not.toMatch(/text-chat-dim/);
  });
});

describe("SettingField — dirty", () => {
  it("adds an aria-hidden dot without changing the label's accessible name", () => {
    render(
      <SettingField kind="text" field={RELOAD_FIELD} value="x" onChange={() => {}} dirty chrome="token" />,
    );
    const dot = screen.getByTestId("setting-field-dirty-dot");
    expect(dot).toHaveAttribute("aria-hidden", "true");
    const input = screen.getByLabelText(/^dtype$/i);
    expect(input).toHaveAccessibleName("dtype");
    // Screen readers hear "(unsaved)" through the hint, not the label.
    const hint = document.getElementById(input.getAttribute("aria-describedby")!)!;
    expect(within(hint).getByText("(unsaved)")).toHaveClass("sr-only");
    expect(input).toHaveAccessibleDescription(expect.stringContaining("(unsaved)"));
  });

  it("renders no dot when clean", () => {
    render(<SettingField kind="text" field={RELOAD_FIELD} value="" onChange={() => {}} dirty={false} />);
    expect(screen.queryByTestId("setting-field-dirty-dot")).toBeNull();
    expect(screen.queryByText("(unsaved)")).toBeNull();
  });
});

describe("SettingField — boolean variant=switch", () => {
  const AFF: FieldHint = {
    label: "Replica affinity",
    hint: "long",
    short: "Keep each conversation on one replica.",
    restart: "none",
  };

  function SwitchHarness({ onChange }: { onChange: (v: boolean) => void }) {
    const [v, setV] = React.useState(false);
    return (
      <SettingField
        kind="boolean"
        variant="switch"
        field={AFF}
        value={v}
        onChange={(next) => {
          setV(next);
          onChange(next);
        }}
        chrome="token"
      />
    );
  }

  it("is found by its label, has role=switch and toggles", () => {
    const onChange = vi.fn();
    render(<SwitchHarness onChange={onChange} />);
    const sw = screen.getByLabelText(/replica affinity/i);
    expect(sw).toHaveAttribute("role", "switch");
    expect(screen.getByRole("switch", { name: /replica affinity/i })).toBe(sw);
    expect(sw).toHaveAttribute("aria-checked", "false");
    // The state word is gone; aria-checked carries the state.
    expect(screen.queryByText("enabled")).toBeNull();
    expect(screen.queryByText("disabled")).toBeNull();

    fireEvent.click(sw);
    expect(onChange).toHaveBeenLastCalledWith(true);
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(sw).toBeChecked();
  });

  it("keeps the hint linked via aria-describedby", () => {
    render(<SwitchHarness onChange={() => {}} />);
    expect(screen.getByRole("switch")).toHaveAccessibleDescription(
      expect.stringContaining("Keep each conversation on one replica."),
    );
  });
});

describe("MODEL_HINTS — short hints for the model settings page", () => {
  // Every key the model settings page draws a control for.
  const PAGE_KEYS = [
    "served_model_name", "hf_repo", "hf_revision",
    "gpu_indices", "tensor_parallel_size", "data_parallel_size",
    "dp_affinity_enabled", "dp_spill_threshold",
    "gpu_memory_utilization", "max_model_len", "dtype", "n_gpu_layers",
    "flash_attn", "cache_type_k", "cache_type_v",
    "supports_vision", "supports_tools", "supports_reasoning",
    "trust_remote_code", "extra_args", "extra_env", "mmproj_filename",
  ];

  it.each(PAGE_KEYS)("%s has a short hint distinct from and shorter than the long one", (k) => {
    const h = MODEL_HINTS[k];
    expect(h.short, k).toBeTruthy();
    expect(h.short!.length, k).toBeLessThanOrEqual(110);
    expect(h.short!.length, k).toBeLessThan(h.hint.length);
  });

  it("the TP short hint matches the plan's copy", () => {
    expect(MODEL_HINTS.tensor_parallel_size.short).toBe("GPUs per replica. Follows GPUs ÷ replicas.");
  });

  it("runtime (global settings) hints are untouched — no short copy", () => {
    for (const h of Object.values(RUNTIME_HINTS)) expect(h.short).toBeUndefined();
  });
});

// Settings redesign Task 4: the page needs a stable input id (the save bar's
// "Fix" focuses `#dp`) and a readout between the control and the hint that is
// linked to the control as a description (the GiB readout under GPU memory
// utilization). All three props are opt-in.
describe("SettingField — id, describedBy, afterControl", () => {
  const NUM: FieldHint = { label: "Replicas", hint: "Long hint.", short: "Short.", restart: "none" };

  it("uses the given id for the input so a label and `#id` both reach it", () => {
    render(<SettingField kind="number" field={NUM} value={2} onChange={() => {}} id="dp" chrome="token" />);
    const input = screen.getByLabelText("Replicas");
    expect(input.id).toBe("dp");
    expect(document.getElementById("dp")).toBe(input);
  });

  it("prepends describedBy ids to the control's aria-describedby", () => {
    render(
      <>
        <span id="readout">≈ 14.4 GiB</span>
        <SettingField kind="number" field={NUM} value={0.9} onChange={() => {}} describedBy="readout" chrome="token" />
      </>,
    );
    const ids = (screen.getByLabelText("Replicas").getAttribute("aria-describedby") ?? "").split(" ");
    expect(ids[0]).toBe("readout");
    expect(ids).toHaveLength(2);
  });

  it("renders afterControl between the control and the hint", () => {
    render(
      <SettingField
        kind="number"
        field={NUM}
        value={0.9}
        onChange={() => {}}
        chrome="token"
        afterControl={<span data-testid="after">readout</span>}
      />,
    );
    const input = screen.getByLabelText("Replicas");
    const after = screen.getByTestId("after");
    const hint = screen.getByText("Short.");
    expect(input.compareDocumentPosition(after) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(after.compareDocumentPosition(hint) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
