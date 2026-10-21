/**
 * Detail-page URLs (BACKLOG-macos P4).
 *
 * Model and token ids are runtime data, so a static export (`LMW_UI_EXPORT=1`,
 * `output: 'export'`) cannot pre-render a `/models/[id]` page per id. The
 * detail pages therefore live at fixed paths and read the id from the query:
 *
 *   /models/detail?id=<id>            — model detail
 *   /models/detail/settings?id=<id>   — model settings
 *   /tokens/detail?id=<id>            — token detail
 *
 * Every link to a detail page is built here, never by hand. Old bookmarks
 * (`/ui/models/<id>`, `/ui/models/<id>/settings`, `/ui/tokens/<id>`) are
 * redirected to these URLs server-side: by `redirects()` in next.config.ts on
 * the Linux standalone build, and by app/webui/redirects.py on the Mac.
 *
 * The breadcrumb registry and the back stack keep working on the old
 * path-shaped form (`/models/<id>/settings`) — the *logical* path: it is what
 * the trail is built from (one segment per crumb) and what the stack stores.
 * `logicalPath` maps a real URL to it and `detailHref` maps it back.
 *
 * Paths are app-router paths without the `/ui` basePath (Next adds it).
 */

export const MODEL_DETAIL_PATH = "/models/detail";
export const MODEL_SETTINGS_PATH = "/models/detail/settings";
export const TOKEN_DETAIL_PATH = "/tokens/detail";

const q = (id: string) => `?id=${encodeURIComponent(id)}`;

export function modelHref(id: string): string {
  return MODEL_DETAIL_PATH + q(id);
}

export function modelSettingsHref(id: string): string {
  return MODEL_SETTINGS_PATH + q(id);
}

export function tokenHref(id: string): string {
  return TOKEN_DETAIL_PATH + q(id);
}

/** Strip one trailing slash (the export builds with `trailingSlash: true`). */
function trimSlash(pathname: string): string {
  return pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
}

/** True when the page at `pathname` takes its id from the query string. */
export function isQueryDetailPath(pathname: string): boolean {
  const p = trimSlash(pathname);
  return p === MODEL_DETAIL_PATH || p === MODEL_SETTINGS_PATH || p === TOKEN_DETAIL_PATH;
}

/**
 * Real URL → logical path: `/models/detail` + `id=abc` → `/models/abc`.
 * Any other path, or a detail path without an id, is returned unchanged.
 */
export function logicalPath(pathname: string, search: string | URLSearchParams): string {
  if (!isQueryDetailPath(pathname)) return pathname;
  const id = (typeof search === "string" ? new URLSearchParams(search) : search).get("id");
  if (!id) return pathname;
  const seg = encodeURIComponent(id);
  switch (trimSlash(pathname)) {
    case MODEL_DETAIL_PATH:
      return `/models/${seg}`;
    case MODEL_SETTINGS_PATH:
      return `/models/${seg}/settings`;
    default:
      return `/tokens/${seg}`;
  }
}

function safeDecode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * Logical path → real URL: `/models/abc/settings` →
 * `/models/detail/settings?id=abc`. Anything that is not a logical detail
 * path (including the real detail paths themselves) is returned unchanged.
 */
export function detailHref(path: string): string {
  const segs = path.split("/").filter(Boolean);
  if (segs.length < 2 || segs[1] === "detail") return path;
  const id = safeDecode(segs[1]);
  if (segs[0] === "models" && segs.length === 2) return modelHref(id);
  if (segs[0] === "models" && segs.length === 3 && segs[2] === "settings") return modelSettingsHref(id);
  if (segs[0] === "tokens" && segs.length === 2) return tokenHref(id);
  return path;
}
