"use client";

import { useState } from "react";
import { Tabs } from "@/components/ui/tabs";
import { GeneralTab } from "@/components/settings/general-tab";
import { NetworkingTab } from "@/components/settings/networking-tab";
import { SessionsTab } from "@/components/settings/sessions-tab";
import { MaintenanceTab } from "@/components/settings/maintenance-tab";
import { ModelTab } from "@/components/settings/model-tab";
import { AdminTokensTab } from "@/components/settings/admin-tokens-tab";
import { GuardPlatformTab } from "@/components/settings/guard-tab";
import ee from "@/lib/extensions";
import { useEdition, hasFeature } from "@/lib/edition";

// ---------------------------------------------------------------------------
// Settings — six-tab shell (#154)
// ---------------------------------------------------------------------------
//
// Pre-redesign: a single 581-line Runtime tab + Model tab. The new IA
// splits Runtime by user task — what an operator is actually trying to
// do — so each pane stays under one screen-height.
//
// Tab order is the spec's:
//   General → Networking → Sessions & Tokens → Maintenance → Model → Access Tokens
//   → Sensitive data (the guard's platform default)
//
// Each of the four runtime sub-tabs owns its own useRuntimeSettings()
// instance — SWR dedupes the GET across tabs, so we still issue exactly
// one /api/settings/runtime fetch per page load.
// ---------------------------------------------------------------------------

const CORE_TABS: { id: string; label: string }[] = [
  { id: "general", label: "General" },
  { id: "networking", label: "Networking" },
  { id: "sessions", label: "Sessions & Tokens" },
  { id: "maintenance", label: "Maintenance" },
  { id: "model", label: "Model" },
  { id: "admin-tokens", label: "Access Tokens" },
  { id: "guard", label: "Sensitive data" },
];

export default function SettingsPage() {
  // Default to General — first port of call for fresh installs
  // (admin credentials, HF token, default GPU indices).
  const [activeTab, setActiveTab] = useState<string>("general");
  const edition = useEdition();
  const eeTabs = ee.settingsTabs.filter((t) => hasFeature(edition, t.feature));
  const tabs = [...CORE_TABS, ...eeTabs];
  const EeTab = eeTabs.find((t) => t.id === activeTab)?.Component;

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <Tabs
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={(id) => setActiveTab(id)}
      >
        {activeTab === "general" && <GeneralTab />}
        {activeTab === "networking" && <NetworkingTab />}
        {activeTab === "sessions" && <SessionsTab />}
        {activeTab === "maintenance" && <MaintenanceTab />}
        {activeTab === "model" && <ModelTab />}
        {activeTab === "admin-tokens" && <AdminTokensTab />}
        {activeTab === "guard" && <GuardPlatformTab />}
        {EeTab && <EeTab />}
      </Tabs>
    </div>
  );
}
