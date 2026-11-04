"use client";

// The sessions pages in key mode sit inside the key holder's area: the same header, nav and Sign out as /key/*.

import { KeyShell } from "@/components/key-area/key-shell";

export function SessionsShell({ children }: { children: React.ReactNode }) {
  return <KeyShell>{children}</KeyShell>;
}
