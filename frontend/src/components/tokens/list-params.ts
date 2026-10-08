// The token list's sort / page / search / filter state and its two
// serialisations: the page URL (`/tokens?sort=&dir=&page=&size=&q=&expiring=1`,
// defaults omitted so a plain /tokens stays plain) and the API request
// (`GET /api/tokens?sort=&dir=&limit=&offset=&q=&near_expiry=1`). Pure functions so both round-trips
// are unit-testable without a router.

import type { operations } from "@/lib/api-types.generated";

type ListQuery = NonNullable<operations["list_tokens_api_tokens_get"]["parameters"]["query"]>;

/** The server's sort columns (app/tokens/routes_api.py list_tokens). */
export type TokenSort = NonNullable<ListQuery["sort"]>;
export type SortDir = NonNullable<ListQuery["dir"]>;

export const TOKEN_SORTS: readonly TokenSort[] = [
  "name", "prefix", "created", "expires", "last_used", "priority", "usage_24h", "status",
];

/** The direction a column sorts in when its header is first clicked: the
 *  order an operator reaches for -- A→Z, soonest expiry, Paused first, but
 *  newest / most-used / highest-priority first. Clicking again flips it. */
export const FIRST_DIR: Record<TokenSort, SortDir> = {
  name: "asc",
  prefix: "asc",
  created: "desc",
  expires: "asc",
  last_used: "desc",
  priority: "desc",
  usage_24h: "desc",
  status: "asc",
};

export const PAGE_SIZES = [25, 50, 100] as const;

/** Typing is debounced before it reaches the URL and the API: one request
 *  per pause, not one per keystroke. */
export const SEARCH_DEBOUNCE_MS = 300;

/** Mirrors the API's `q` max_length. */
export const SEARCH_MAX = 64;

export interface ListParams {
  sort: TokenSort;
  dir: SortDir;
  /** 1-based. */
  page: number;
  size: number;
  /** Trimmed; "" = no filter. */
  q: string;
  /** Only keys expiring within 30 days (the API's `near_expiry=1`). */
  expiring: boolean;
}

export const DEFAULT_LIST_PARAMS: ListParams = {
  sort: "created",
  dir: "desc",
  page: 1,
  size: 50,
  q: "",
  expiring: false,
};

interface Readable {
  get(name: string): string | null;
}

/** URL → params. Anything unrecognised falls back to its default rather than
 *  producing a request the server would 422. */
export function parseListParams(sp: Readable): ListParams {
  const d = DEFAULT_LIST_PARAMS;
  const sort = sp.get("sort");
  const dir = sp.get("dir");
  const page = Number(sp.get("page"));
  const size = Number(sp.get("size"));
  return {
    sort: TOKEN_SORTS.includes(sort as TokenSort) ? (sort as TokenSort) : d.sort,
    dir: dir === "asc" || dir === "desc" ? dir : d.dir,
    page: Number.isSafeInteger(page) && page >= 1 ? page : d.page,
    size: (PAGE_SIZES as readonly number[]).includes(size) ? size : d.size,
    q: (sp.get("q") ?? "").trim().slice(0, SEARCH_MAX),
    expiring: sp.get("expiring") === "1",
  };
}

/** Params → the page's query string, every default left out ("" for all
 *  defaults). */
export function listParamsToSearch(p: ListParams): string {
  const d = DEFAULT_LIST_PARAMS;
  const out = new URLSearchParams();
  if (p.sort !== d.sort) out.set("sort", p.sort);
  if (p.dir !== d.dir) out.set("dir", p.dir);
  if (p.page !== d.page) out.set("page", String(p.page));
  if (p.size !== d.size) out.set("size", String(p.size));
  if (p.q !== d.q) out.set("q", p.q);
  if (p.expiring) out.set("expiring", "1");
  return out.toString();
}

/** Params → the SWR key / API URL. */
export function listApiUrl(p: ListParams): string {
  const qs = new URLSearchParams({
    sort: p.sort,
    dir: p.dir,
    limit: String(p.size),
    offset: String((p.page - 1) * p.size),
  });
  if (p.q) qs.set("q", p.q);
  if (p.expiring) qs.set("near_expiry", "1");
  return `/api/tokens?${qs.toString()}`;
}
