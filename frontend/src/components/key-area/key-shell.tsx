"use client";

// The key holder's chrome: the key's name and state, the area's nav and Sign out. Nothing of the admin app (no admin
// nav, no admin requests). Sign out ends the sign-in on the server (POST /api/forest/logout), not only in this tab.

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSWRConfig } from "swr";

import { Button } from "@/components/ui/button";
import { type MeKey, signOutKey, useMe } from "@/lib/key-area/api";
import { K } from "@/lib/key-area/strings";
import { KeyAuthError } from "@/lib/sessions/client";
import { isKeyCacheKey, SESSIONS_LOGIN } from "@/lib/sessions/key-token";
import { day, parseSqliteUtc } from "@/lib/token-format";
import { cn } from "@/lib/utils";

export function keyStatus(k: MeKey): keyof typeof K.shell.status {
  if (k.is_revoked) return "revoked";
  if (k.is_expired) return "expired";
  if (k.is_paused) return "paused";
  return "active";
}

/** Send the key holder back to the API key sign-in when the sign-in is gone. */
export function useKeyAuthRedirect(error: unknown) {
  const router = useRouter();
  useEffect(() => {
    if (error instanceof KeyAuthError) router.replace(SESSIONS_LOGIN);
  }, [error, router]);
}

export function useKeySignOut() {
  const router = useRouter();
  const { mutate } = useSWRConfig();
  return async () => {
    await signOutKey();
    await mutate(isKeyCacheKey, undefined, { revalidate: false });
    router.replace(SESSIONS_LOGIN);
  };
}

/** "Sep 12", or "Sep 12 2027" outside this year. */
function dayYear(sec: number): string {
  const y = new Date(sec * 1000).getFullYear();
  return y === new Date().getFullYear() ? day(sec) : `${day(sec)} ${y}`;
}

function Meta({ k }: { k: MeKey }) {
  const created = parseSqliteUtc(k.created_at);
  const expires = parseSqliteUtc(k.expires_at);
  const status = keyStatus(k);
  return (
    <span className="block text-xs text-slate-400">
      <span className="font-mono">{k.prefix}…</span>
      {created !== null && ` · ${K.shell.created} ${dayYear(created)}`}
      {` · ${expires !== null ? `${K.shell.expires} ${dayYear(expires)}` : K.shell.never} · `}
      <span
        className={cn(
          "rounded-full border px-2 py-px",
          status === "active" ? "border-emerald-800 text-emerald-400" : "border-red-900 text-red-400",
        )}
      >
        {K.shell.status[status]}
      </span>
    </span>
  );
}

export function KeyShell({ children }: { children: React.ReactNode }) {
  const path = usePathname() ?? "";
  const { data, error } = useMe<MeKey>("/api/me/key");
  const signOut = useKeySignOut();
  useKeyAuthRedirect(error);
  const here = path.replace(/\/$/, "") || "/";

  return (
    <div className="mx-auto max-w-6xl px-4 py-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/40">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
          <div>
            <h1 className="text-lg font-semibold">{data?.name ?? " "}</h1>
            {data && <Meta k={data} />}
          </div>
          <Button variant="outline" onClick={() => void signOut()}>
            {K.shell.signOut}
          </Button>
        </header>
        <nav aria-label="Key" className="flex flex-wrap gap-1 border-b border-slate-800 px-3 py-2">
          {K.shell.nav.map((item) => {
            const on = item.href === "/sessions" ? here.startsWith("/sessions") : here === item.href;
            return (
              <Link
                key={item.href}
                href={item.href === "/sessions" ? "/sessions?key=1" : item.href}
                aria-current={on ? "page" : undefined}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm",
                  on ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:text-slate-200",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="px-4 py-4">
          {error && !(error instanceof KeyAuthError) ? (
            <p role="alert" className="text-sm text-red-400">
              {K.shell.loadError}
            </p>
          ) : (
            children
          )}
        </div>
      </div>
    </div>
  );
}
