// App-wide 404.
//
// /ui/stats/live was removed and deliberately NOT redirected: there is one
// stats page now, and a silent redirect would leave people believing the old
// route still exists. The nav entry went with it, so the only way to reach
// this for that route is a bookmark or a tab left open — worth having the
// 404 name the page that replaced it rather than being a bare dead end.

import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-md py-16 text-center">
      <p className="font-mono text-sm text-chat-dim">404</p>
      <h1 className="mt-2 text-xl font-semibold">This page does not exist.</h1>
      <p className="mt-3 text-sm text-chat-muted">
        Looking for the live stats view? It merged into{" "}
        <Link href="/stats" className="text-chat-accent underline underline-offset-2">
          Stats
        </Link>
        {" "}— one page now carries the history charts, the live timeline and
        the request tables.
      </p>
    </div>
  );
}
