"use client";

interface ExpiryBannerProps {
  /** `near_expiry` from GET /api/tokens: visible keys expiring within 30
   *  days and not yet expired, counted over the whole list server-side --
   *  the page in hand holds at most one page of them. */
  count: number;
  /** Filters the list to exactly those keys (`near_expiry=1`). Omitted while
   *  that filter is already on. */
  onShow?: () => void;
}

export function ExpiryBanner({ count, onShow }: ExpiryBannerProps) {
  if (count <= 0) return null;

  return (
    <div
      role="alert"
      className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-100/10 p-3 text-amber-200"
    >
      <div>
        <p className="text-sm font-semibold">Tokens expiring soon ({count.toLocaleString()})</p>
        <p className="mt-1 text-xs text-amber-100/90">
          {count === 1 ? "1 token expires" : `${count.toLocaleString()} tokens expire`} within 30 days.
        </p>
      </div>
      {onShow && (
        <button
          type="button"
          onClick={onShow}
          className="rounded-md border border-amber-500/40 px-2.5 py-1 text-xs text-amber-200 hover:bg-amber-100/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
        >
          Show them
        </button>
      )}
    </div>
  );
}
