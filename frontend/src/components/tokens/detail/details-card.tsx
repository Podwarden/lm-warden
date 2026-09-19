// Details card (mockup `dl.facts`).
import type { TokenDetail } from "@/lib/token-series";
import { compact, fmtDateTime, parseSqliteUtc, relativeAgo } from "@/lib/token-format";
import { CARD, CARD_TITLE } from "./styles";

export function DetailsCard({ token, nowSec }: { token: TokenDetail; nowSec: number }) {
  const created = parseSqliteUtc(token.created_at);
  const expires = parseSqliteUtc(token.expires_at);
  const u = token.usage_24h;
  const rows: [string, string][] = [
    ["Last used", relativeAgo(parseSqliteUtc(token.last_used_at), nowSec)],
    ["Created", created == null ? "—" : fmtDateTime(created, nowSec)],
    ["Expires", expires == null ? "Never" : fmtDateTime(expires, nowSec)],
    ["Last 24h", u.requests === 0 ? "—" : `${compact(u.requests)} req · ${compact(u.prompt_tokens + u.completion_tokens)} tok`],
  ];
  return (
    <section className={CARD} aria-label="Details">
      <h2 className={CARD_TITLE}>Details</h2>
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3.5 gap-y-[7px] text-[13px]">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-chat-dim">{k}</dt>
            <dd className="m-0 text-right tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
