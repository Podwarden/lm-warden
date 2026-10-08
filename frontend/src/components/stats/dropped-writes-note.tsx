import type { StatsV2Bookkeeping } from "@/lib/stats-v2";

/** The dim "stats writes dropped" line (#294).
 *
 *  Renders nothing until the proxy has actually dropped a write: the counters
 *  are process-local, so "since start" is the warden's last restart. A null
 *  `requests_lost` is unknown and is left out rather than shown as 0. The
 *  tooltip breaks the count down by call site. */
export function DroppedWritesNote({ data }: { data: StatsV2Bookkeeping | undefined }) {
  const total = data?.total;
  if (!data || !total || total.count <= 0) return null;
  const writes = total.count === 1 ? "stats write" : "stats writes";
  const lost =
    total.requests_lost === null
      ? ""
      : ` (${total.requests_lost.toLocaleString()} ${
          total.requests_lost === 1 ? "request's" : "requests'"
        } accounting lost)`;
  const since = new Date(data.since_epoch * 1000).toLocaleString();
  const breakdown = Object.entries(data.sites)
    .map(
      ([site, v]) =>
        `${site}: ${v.count}` + (v.requests_lost === null ? "" : `, ${v.requests_lost} lost`),
    )
    .join("\n");
  return (
    <p
      data-testid="stats-dropped-writes"
      className="text-xs text-chat-dim"
      title={`Since ${since}\n${breakdown}`}
    >
      {total.count.toLocaleString()} {writes} dropped since start{lost}
    </p>
  );
}
