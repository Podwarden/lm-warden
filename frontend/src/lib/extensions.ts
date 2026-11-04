// The ONLY file allowed to import `@ee` (enforced by ESLint and
// tests/unit/no-ee-imports.test.ts). `@ee` resolves to src/ee/index.ts in an
// Enterprise build (LM_EDITION=enterprise) and to src/ee-stub/index.ts otherwise.
import ee from "@ee";
import type { EeModule } from "@/lib/ext/types";

export default ee as EeModule;
