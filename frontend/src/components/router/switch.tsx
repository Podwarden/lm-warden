"use client";

// Router-local switch (plan §4.2, §5): a real checkbox with role="switch", so
// keyboard, form semantics and `aria-checked` come for free. Same look as the
// model-settings redesign's `SettingField variant="switch"` (mockup .switch /
// .switch.sm). Follow-up (plan "Out of scope"): unify both into
// components/ui/switch.tsx once the two redesign branches land.

import { cn } from "@/lib/utils";

export interface RouterSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  /** Accessible name (aria-label), e.g. "Routing" or "Enable rule claude-haiku*". */
  label: string;
  /** id of the element holding the visible state text. */
  describedBy?: string;
  disabled?: boolean;
  /** md = 44x24 (master switch), sm = 36x20 (row switch). */
  size?: "md" | "sm";
  testId?: string;
  id?: string;
}

// .switch { 44x24, radius 999, surface-2 bg, rule-soft border }
// .switch:checked { ok-bg bg, live/.7 border }
const TRACK =
  "peer m-0 flex-none cursor-pointer appearance-none rounded-full border border-vw-rule-soft bg-chat-surface-2 " +
  "transition-colors checked:border-vw-live/70 checked:bg-vw-ok-bg " +
  "focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-chat-accent " +
  "disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transition-none";

// .switch::after { top/left 3px, muted knob }  :checked::after { live, translateX }
const KNOB =
  "pointer-events-none absolute left-[4px] top-[4px] rounded-full bg-chat-muted transition-transform " +
  "peer-checked:bg-vw-live peer-disabled:opacity-50 motion-reduce:transition-none";

const SIZE = {
  md: { track: "h-6 w-11", knob: "h-4 w-4 peer-checked:translate-x-5" },
  sm: { track: "h-5 w-9", knob: "h-3 w-3 peer-checked:translate-x-4" },
} as const;

export function RouterSwitch({
  checked,
  onChange,
  label,
  describedBy,
  disabled,
  size = "md",
  testId,
  id,
}: RouterSwitchProps) {
  const s = SIZE[size];
  return (
    <span className="relative inline-flex flex-none items-center">
      <input
        id={id}
        type="checkbox"
        role="switch"
        aria-label={label}
        aria-checked={checked}
        aria-describedby={describedBy}
        data-testid={testId}
        className={cn(TRACK, s.track)}
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span aria-hidden="true" className={cn(KNOB, s.knob)} />
    </span>
  );
}
