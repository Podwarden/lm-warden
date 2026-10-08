// TemplateList (#162) — the presentational grid on the Templates manager page.
//
// Contract pinned here:
//   - the human label renders
//   - built-in templates carry a "built-in" badge and expose NO delete control
//   - user templates expose a delete control wired to onDelete(id)
//
// The page-level data fetch (SWR) + DELETE round-trip live in the container
// page (templates/page.tsx) — this component is pure props in / callback out,
// so the test drives it directly without a fetch stub.
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { TemplateList, type TemplateDTO } from "@/components/templates/template-list";

const TEMPLATES: TemplateDTO[] = [
  {
    id: "gpt-oss-20b",
    label: "GPT-OSS 20B",
    source: "builtin",
    hf_repo: "openai/gpt-oss-20b",
    engine: { channel: "cuda-stable", vllm_version: "0.20.0", image: null },
  },
  {
    id: "my-mistral",
    label: "My Mistral",
    source: "user",
    hf_repo: "mistralai/Mistral-7B",
    engine: { channel: "cuda-stable", vllm_version: "0.20.0", image: null },
  },
];

describe("TemplateList", () => {
  afterEach(cleanup);

  it("renders builtin badge and a user-only delete control", () => {
    render(<TemplateList templates={TEMPLATES} onDelete={() => {}} />);
    expect(screen.getByText("GPT-OSS 20B")).toBeInTheDocument();
    expect(screen.getByText("My Mistral")).toBeInTheDocument();
    // builtin badge present (matches "built-in" / "builtin")
    expect(screen.getByText(/built-?in/i)).toBeInTheDocument();
    // user template exposes a delete control; builtin does not
    expect(screen.getByTestId("delete-my-mistral")).toBeInTheDocument();
    expect(screen.queryByTestId("delete-gpt-oss-20b")).toBeNull();
  });

  it("calls onDelete with the template id when the delete control is clicked", () => {
    const onDelete = vi.fn();
    render(<TemplateList templates={TEMPLATES} onDelete={onDelete} />);
    fireEvent.click(screen.getByTestId("delete-my-mistral"));
    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onDelete).toHaveBeenCalledWith("my-mistral");
  });

  it("renders the engine combo when present and tolerates a null engine", () => {
    render(
      <TemplateList
        templates={[
          { id: "no-engine", label: "No Engine", source: "user", hf_repo: "a/b", engine: null },
          TEMPLATES[0],
        ]}
        onDelete={() => {}}
      />,
    );
    expect(screen.getByText(/cuda-stable/)).toBeInTheDocument();
    expect(screen.getByText(/vLLM 0\.20\.0/)).toBeInTheDocument();
    // No engine row → no crash, label still renders.
    expect(screen.getByText("No Engine")).toBeInTheDocument();
  });

  it("renders an empty-state hint when there are no templates", () => {
    render(<TemplateList templates={[]} onDelete={() => {}} />);
    expect(screen.getByText(/no templates/i)).toBeInTheDocument();
  });
});
