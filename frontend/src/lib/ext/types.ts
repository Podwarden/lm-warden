import type { ComponentType } from "react";
import type { LucideIcon } from "lucide-react";

// Editions split: the shapes an Enterprise build plugs into the core UI.
// The community build resolves `@ee` to src/ee-stub, where every field is empty.
export type EeNavItem = { href: string; label: string; icon: LucideIcon; feature: string };
export type EeSettingsTab = {
  id: string;
  label: string;
  feature: string;
  Component: ComponentType;
};
export type EeModule = {
  navItems: EeNavItem[];
  settingsTabs: EeSettingsTab[];
  // rendered at /ui/enterprise?view=<id>
  views: Record<string, ComponentType>;
  // key detail page slots
  tokenPanels: Record<string, ComponentType<{ tokenId: string }>>;
  loginProviders: { id: string; label: string; href: string }[];
};
