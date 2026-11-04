// The type of the `@ee` module, whichever tree the bundler picked.
//
// `@ee` is resolved by the bundlers alone (next.config.ts for webpack and
// turbopack, vitest.config.ts): src/ee/index.ts under LM_EDITION=enterprise,
// src/ee-stub/index.ts otherwise. It must NOT be a tsconfig `paths` entry:
// Next.js turns tsconfig paths into a webpack resolver that wins over
// `resolve.alias`, so a `paths` mapping to the stub silently built the stub
// into the Enterprise UI (no ee code, no canary in .next; found 2026-11-04).
// Both trees satisfy this contract: each declares `const ee: EeModule`.
//
// Lives outside src/ so tests/unit/no-ee-imports.test.ts (no file in src/
// but lib/extensions.ts names `@ee`) stays exact.
declare module "@ee" {
  import type { EeModule } from "@/lib/ext/types";
  const ee: EeModule;
  export default ee;
}
