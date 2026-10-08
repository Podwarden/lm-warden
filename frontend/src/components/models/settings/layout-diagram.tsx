import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { planLayout } from "@/lib/settings-layout";

function B({ children }: { children: ReactNode }) {
  return <b className="font-semibold text-chat-fg">{children}</b>;
}

/**
 * Read-only picture of how the selected GPUs fall into replicas
 * (plan 2026-10-04-settings-redesign §4.3). Sits at the top of the Layout
 * section, directly above the GPU / Replicas / Tensor-parallel controls it
 * reflects. `role="img"`: the accessible name is the equation sentence; the
 * boxes are decoration for sighted users. It is never an alert — the field
 * error under Replicas (`dp-invalid-error`) carries that.
 */
export function LayoutDiagram({
  gpuIndices,
  dp,
  tp,
  backend,
}: {
  gpuIndices: readonly number[];
  dp: number;
  tp: number;
  backend?: string | null;
}) {
  const plan = planLayout({ gpuIndices, dp, tp, backend });
  const n = plan.gpuCount;

  let equation: ReactNode;
  if (n === 0) {
    equation = (
      <>
        <B>No GPUs selected.</B> Pick at least one.
      </>
    );
  } else if (plan.llamacpp) {
    equation =
      n === 1 ? (
        <>
          <B>1 GPU</B> · all layers on it
        </>
      ) : (
        <>
          <B>{n} GPUs</B> · layers split across them
        </>
      );
  } else if (!plan.valid && n % plan.dp !== 0) {
    equation = (
      <>
        <B>{n === 1 ? "1 GPU" : `${n} GPUs`}</B> can&apos;t split into <B>{plan.dp}</B> equal
        replicas.
      </>
    );
  } else {
    const per = plan.tp;
    equation = (
      <>
        <B>{n === 1 ? "1 GPU" : `${n} GPUs`}</B> {plan.valid ? "=" : "≠"} <B>{plan.dp}</B>{" "}
        {plan.dp === 1 ? "replica" : "replicas"} × <B>{per}</B> {per === 1 ? "GPU" : "GPUs"} each{" "}
        <span className="text-chat-muted">(tensor-parallel {plan.tp})</span>
        {!plan.valid && (
          <>
            {" "}
            — tensor-parallel should be <B>{n / plan.dp}</B>.
          </>
        )}
      </>
    );
  }

  const boxes: { name: string; gpus: number[] }[] = plan.valid
    ? plan.replicas.map((g, r) => ({
        name: plan.llamacpp
          ? n === 1
            ? "One GPU"
            : "Layers split"
          : plan.dp > 1
            ? `Replica ${r + 1}`
            : "One replica",
        gpus: g,
      }))
    : plan.unassigned.length > 0
      ? [{ name: "Unassigned", gpus: plan.unassigned }]
      : [];

  return (
    <div
      role="img"
      aria-label={plan.sentence}
      data-testid="layout-diagram"
      data-invalid={String(!plan.valid)}
      className={cn(
        "rounded-[10px] border border-dashed bg-chat-page/35 p-3",
        plan.valid ? "border-vw-rule-soft" : "border-chat-negative/80",
      )}
    >
      <p className="mb-2.5 text-[13.5px] leading-snug text-chat-muted tabular-nums [overflow-wrap:anywhere]">
        {equation}
      </p>
      {boxes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {boxes.map((box) => (
            <div
              key={box.name}
              data-testid="layout-replica"
              className={cn(
                "flex min-w-[92px] flex-col gap-1 rounded-lg border bg-chat-surface/70 p-1.5",
                plan.valid ? "border-vw-rule-soft" : "border-chat-negative/60",
              )}
            >
              <span className="px-0.5 text-[11px] font-semibold uppercase tracking-[0.06em] text-chat-muted">
                {box.name}
              </span>
              <span className="flex flex-wrap gap-1">
                {box.gpus.map((i) => (
                  <span
                    key={i}
                    className="whitespace-nowrap rounded-[5px] bg-chat-surface-2 px-[7px] py-[3px] text-xs text-chat-fg tabular-nums"
                  >
                    GPU {i}
                  </span>
                ))}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
