// Community-build leak check. src/ee/ (Task 7) embeds the string below in a
// module-level constant; a community build must never contain it. Usage:
//   LM_EDITION=community npx next build && node scripts/check-ee-canary.mjs [dir ...]
// Exit 1 and list the files when the canary is found. Default dirs: .next, out.
//
// --present inverts it for the Enterprise build, which must carry the canary
// (proof that @ee resolved to src/ee and not to the stub): exit 0 when found,
// exit 1 when absent. frontend/Dockerfile runs one mode or the other.
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export const EE_CANARY = 'LMW-EE-CANARY';

// Files of 20 MB or more are skipped: emitted JS/CSS/HTML never gets near that,
// and a huge binary or source map is not worth reading whole.
const MAX_BYTES = 20_000_000;

export function findCanary(dir) {
  const hits = [];
  const walk = (d) => {
    for (const n of readdirSync(d)) {
      const p = path.join(d, n);
      const st = statSync(p);
      if (st.isDirectory()) {
        // Skip `cache` dirs: the webpack/Next build cache can hold stale output
        // of an earlier (e.g. Enterprise) build and would raise a false hit.
        if (n === 'cache') continue;
        walk(p);
      } else if (st.size < MAX_BYTES && readFileSync(p).includes(EE_CANARY)) hits.push(p);
    }
  };
  if (existsSync(dir)) walk(dir);
  return hits;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const present = args.includes('--present');
  const dirs = args.filter((a) => a !== '--present');
  const targets = (dirs.length ? dirs : ['.next', 'out']).filter(existsSync);
  if (!targets.length) {
    console.error('check-ee-canary: no build output found');
    process.exit(2);
  }
  const hits = targets.flatMap(findCanary);
  if (present) {
    if (!hits.length) {
      console.error(`check-ee-canary: ${EE_CANARY} is missing from an Enterprise build (${targets.join(', ')})`);
      process.exit(1);
    }
    console.log(`check-ee-canary: Enterprise build carries the canary (${hits.length} file(s))`);
    process.exit(0);
  }
  if (hits.length) {
    console.error(`check-ee-canary: ${EE_CANARY} found in a community build:\n${hits.join('\n')}`);
    process.exit(1);
  }
  console.log(`check-ee-canary: clean (${targets.join(', ')})`);
}
