import type { EeModule } from "@/lib/ext/types";

// Community build: every extension slot is empty.
const ee: EeModule = {
  navItems: [],
  settingsTabs: [],
  views: {},
  tokenPanels: {},
  loginProviders: [],
};

export default ee;
