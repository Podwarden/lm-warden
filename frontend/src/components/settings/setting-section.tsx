"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// ---------------------------------------------------------------------------
// SettingSection — Card-wrapped section primitive used across the four
// runtime sub-tabs (General / Networking / Sessions & Tokens / Maintenance).
//
// Pure presentation. No state, no hooks. Renders a titled card with an
// optional subtitle, an optional restart chip (e.g. "Some fields here
// require model reload"), and arbitrary children — which the tab fills
// with one or more `<SettingField>` rows.
//
// The restart-chip slot is per-section, not per-field — the per-field
// chip stays on SettingField. The section chip is the operator's
// "heads up — saving anything in this card will require X" cue, mirroring
// the wireframes in the #154 spec.
// ---------------------------------------------------------------------------

type SectionRestartKind = "model-reload" | "warden-restart";

interface SettingSectionProps {
  title: string;
  subtitle?: string;
  /** Optional per-section restart chip. Same `restart` values the
   *  per-field FieldHint uses, minus `"none"` (no chip when none). */
  restartChip?: SectionRestartKind;
  children: React.ReactNode;
}

const RESTART_CHIP_COPY: Record<SectionRestartKind, string> = {
  "model-reload": "Some fields here require model reload",
  "warden-restart": "Some fields here require warden restart",
};

export function SettingSection({
  title,
  subtitle,
  restartChip,
  children,
}: SettingSectionProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <CardTitle>{title}</CardTitle>
          {restartChip && (
            <Badge
              variant="info"
              aria-label={`section requires ${restartChip}`}
            >
              {RESTART_CHIP_COPY[restartChip]}
            </Badge>
          )}
        </div>
        {subtitle && (
          <p className="text-xs text-slate-500">{subtitle}</p>
        )}
      </CardHeader>
      <CardContent className="space-y-6">{children}</CardContent>
    </Card>
  );
}
