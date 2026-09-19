"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { ExpiryBanner } from "@/components/tokens/expiry-banner";
import { TokenRow } from "@/components/tokens/token-row";
import { CreateTokenDialog } from "@/components/tokens/create-token-dialog";
import {
  FIRST_DIR,
  PAGE_SIZES,
  SEARCH_DEBOUNCE_MS,
  SEARCH_MAX,
  listApiUrl,
  listParamsToSearch,
  parseListParams,
  type ListParams,
  type TokenSort,
} from "@/components/tokens/list-params";
import type { components } from "@/lib/api-types.generated";

// GET /api/tokens: one sorted page plus the list-wide counts
// (app/tokens/routes_api.py list_tokens, response model TokenListPage).
type TokensResponse = components["schemas"]["TokenListPage"];

interface Column {
  sort: TokenSort;
  label: string;
  className?: string;
  title?: string;
}

const COLUMNS: Column[] = [
  { sort: "name", label: "Name" },
  { sort: "prefix", label: "Prefix" },
  { sort: "created", label: "Created" },
  { sort: "expires", label: "Expires" },
  { sort: "last_used", label: "Last used" },
  {
    sort: "priority",
    label: "Priority",
    title:
      "STRICT priority scheduler — higher priority is always " +
      "served first. See docs/operating.md for starvation semantics.",
  },
  { sort: "usage_24h", label: "Last 24h", className: "text-right" },
  { sort: "status", label: "Status" },
];

const SIZE_OPTIONS = PAGE_SIZES.map((n) => ({ value: String(n), label: String(n) }));

export default function TokensPage() {
  // useSearchParams opts the tree into client-side rendering; Suspense is
  // required around it (same as tokens/[id]/page.tsx).
  return (
    <Suspense fallback={null}>
      <TokensView />
    </Suspense>
  );
}

function TokensView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // The URL seeds the state; the state is the source and the URL mirrors it
  // (replace, not push -- a sort click is not a history step).
  const [params, setParams] = useState<ListParams>(() => parseListParams(searchParams));
  const [qInput, setQInput] = useState(params.q);
  const [createOpen, setCreateOpen] = useState(false);

  // The query string we last wrote (or started from). A URL that differs
  // from it did not come from us -- the nav's plain "Tokens" link while the
  // list is sorted, back/forward -- so the state follows the URL instead.
  const written = useRef(searchParams.toString());
  const urlSearch = searchParams.toString();
  useEffect(() => {
    if (urlSearch === written.current) return;
    written.current = urlSearch;
    const next = parseListParams(new URLSearchParams(urlSearch));
    setParams(next);
    setQInput(next.q);
  }, [urlSearch]);

  useEffect(() => {
    const qs = listParamsToSearch(params);
    if (qs === written.current) return;
    written.current = qs;
    router.replace(qs ? `/tokens?${qs}` : "/tokens", { scroll: false });
  }, [params, router]);

  useEffect(() => {
    const next = qInput.trim().slice(0, SEARCH_MAX);
    if (next === params.q) return;
    const t = setTimeout(() => setParams((p) => ({ ...p, q: next, page: 1 })), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [qInput, params.q]);

  const { data, error, isLoading, isValidating, mutate } = useSWR<TokensResponse>(
    listApiUrl(params),
    authFetchJSON,
    {
      // Match the cadence of the rest of the operator UI. A number, not a
      // function returning 0 while the tab is hidden: SWR never schedules
      // another tick after a 0. It skips hidden tabs by itself
      // (refreshWhenHidden defaults to false).
      refreshInterval: 10_000,
      // Paging, re-sorting and searching keep the current rows up until the
      // next page lands, so the table and the pager never unmount (no layout
      // jump; keyboard focus stays on the button that was pressed).
      keepPreviousData: true,
    },
  );

  // A page that came back empty although rows exist -- the last row of the
  // last page was deleted, or the list shrank under us -- steps back to the
  // last page that has rows. Only for the response to THIS page: with
  // keepPreviousData, `data` can still be the previous page's.
  useEffect(() => {
    if (!data || data.items.length > 0 || params.page <= 1) return;
    if (data.limit !== params.size || data.offset !== (params.page - 1) * params.size) return;
    const last = Math.max(1, Math.ceil(data.total / params.size));
    setParams((p) => ({ ...p, page: Math.min(p.page - 1, last) }));
  }, [data, params.page, params.size]);

  // SWR's isLoading is true for any key it has no answer for yet -- every
  // page, sort and search change -- even while keepPreviousData shows the
  // old rows. Only the very first load gets the skeleton; later ones dim the
  // rows in place.
  const firstLoad = !data && !error;
  const stale = isLoading && !!data;
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const offset = data?.offset ?? 0;
  const filtered = params.q !== "" || params.expiring;

  function refresh() {
    // Revalidate the page in view, in place. The 10s poll catches any miss.
    mutate().catch(() => {});
  }

  function onCreateClose() {
    setCreateOpen(false);
    refresh();
  }

  function onSort(col: TokenSort) {
    setParams((p) =>
      p.sort === col
        ? { ...p, dir: p.dir === "asc" ? "desc" : "asc", page: 1 }
        : { ...p, sort: col, dir: FIRST_DIR[col], page: 1 },
    );
  }

  function onSize(value: string) {
    const size = Number(value);
    // Keep the first row in view on the new page size.
    setParams((p) => ({ ...p, size, page: Math.floor(((p.page - 1) * p.size) / size) + 1 }));
  }

  function clearSearch() {
    setQInput("");
    setParams((p) => ({ ...p, q: "", page: 1 }));
  }

  function setExpiring(expiring: boolean) {
    setParams((p) => ({ ...p, expiring, page: 1 }));
  }

  function clearFilters() {
    setQInput("");
    setParams((p) => ({ ...p, q: "", expiring: false, page: 1 }));
  }

  const lastPage = Math.max(1, Math.ceil(total / params.size));

  return (
    <div className="space-y-4">
      <ExpiryBanner
        count={data?.near_expiry ?? 0}
        onShow={params.expiring ? undefined : () => setExpiring(true)}
      />

      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">API tokens</h1>
        <Button onClick={() => setCreateOpen(true)}>Create token</Button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="w-full max-w-sm">
          <label htmlFor="token-search" className="sr-only">
            Search tokens
          </label>
          <Input
            id="token-search"
            type="search"
            placeholder="Search by name"
            maxLength={SEARCH_MAX}
            value={qInput}
            onChange={(e) => setQInput(e.target.value)}
          />
        </div>
        {params.expiring && (
          <span
            data-testid="filter-chip"
            className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-100/10 py-0.5 pl-2.5 pr-1 text-xs text-amber-200"
          >
            Expiring within 30 days
            <button
              type="button"
              aria-label="Remove filter: expiring within 30 days"
              onClick={() => setExpiring(false)}
              className="rounded-full px-1 leading-none text-amber-200/80 hover:bg-amber-100/10 hover:text-amber-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
            >
              ×
            </button>
          </span>
        )}
      </div>

      {firstLoad && (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}

      {error && (
        <p className="text-sm text-red-500">
          Failed to load tokens{error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}

      {data && !error && total === 0 && !filtered && (
        <div className="rounded-md border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-slate-400">
          <p className="text-sm">No tokens yet — create one to authenticate API clients.</p>
        </div>
      )}

      {data && !error && total === 0 && filtered && (
        <div className="rounded-md border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-slate-400">
          <p className="text-sm">
            {params.q
              ? `No tokens match “${params.q}”${params.expiring ? " among those expiring within 30 days" : ""}.`
              : "No tokens expire within 30 days."}
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-3"
            onClick={params.q && !params.expiring ? clearSearch : clearFilters}
          >
            {params.q && !params.expiring ? "Clear search" : "Clear filters"}
          </Button>
        </div>
      )}

      {/* Kept mounted through page / sort / search changes and poll errors:
          stale rows stay (dimmed while the next page loads) rather than
          unmounting the pager under the operator's focus. */}
      {data && total > 0 && (
        <>
          <div
            aria-busy={isValidating}
            className={`overflow-x-auto rounded-md border border-slate-800 transition-opacity ${stale ? "opacity-60" : ""}`}
            data-testid="token-table"
          >
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 bg-slate-900/50 text-left text-xs uppercase text-slate-400">
                <tr>
                  {COLUMNS.map((col) => (
                    <SortHeader
                      key={col.sort}
                      column={col}
                      active={params.sort === col.sort}
                      dir={params.dir}
                      onSort={onSort}
                    />
                  ))}
                  <th className="px-2 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <TokenRow key={it.id} item={it} onChange={refresh} />
                ))}
              </tbody>
            </table>
          </div>

          <nav
            aria-label="Token list pages"
            className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400"
          >
            <p data-testid="token-page-range" className="tabular-nums">
              {items.length > 0
                ? `${(offset + 1).toLocaleString()}–${(offset + items.length).toLocaleString()} of ${total.toLocaleString()}`
                : `0 of ${total.toLocaleString()}`}
            </p>
            <div className="flex items-center gap-2">
              <span>Rows per page</span>
              <Select
                className="w-20"
                ariaLabel="Rows per page"
                options={SIZE_OPTIONS}
                value={String(params.size)}
                onChange={onSize}
              />
              <Button
                variant="outline"
                size="sm"
                disabled={params.page <= 1}
                onClick={() => setParams((p) => ({ ...p, page: p.page - 1 }))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={params.page >= lastPage}
                onClick={() => setParams((p) => ({ ...p, page: p.page + 1 }))}
              >
                Next
              </Button>
            </div>
          </nav>
        </>
      )}

      <CreateTokenDialog open={createOpen} onClose={onCreateClose} />
    </div>
  );
}

function SortHeader({
  column,
  active,
  dir,
  onSort,
}: {
  column: Column;
  active: boolean;
  dir: "asc" | "desc";
  onSort: (col: TokenSort) => void;
}) {
  return (
    <th
      className={`px-2 py-2 ${column.className ?? ""}`}
      title={column.title}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(column.sort)}
        className="inline-flex items-center gap-1 rounded-sm uppercase hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
      >
        {column.label}
        {active && (
          <span aria-hidden="true" data-testid="sort-indicator" className="text-[0.6rem] text-emerald-400/80">
            {dir === "asc" ? "▲" : "▼"}
          </span>
        )}
      </button>
    </th>
  );
}
