// God-mode media strip — inline image thumbnails for vision requests
// (spec 2026-08-03). Stored items are fetched through authFetch (the UI's
// JWT lives in a header, so a bare <img src> can't authenticate) and shown
// via object URLs; remote http(s) urls are hotlinked directly (user-chosen
// auto-load). Click any thumbnail -> full-size lightbox.

"use client";

import { useEffect, useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import type { MediaEntry } from "./godmode-viewer";

/** chars is base64 length; decoded bytes ≈ chars * 3/4. Display-only. */
export function mediaSizeLabel(chars?: number): string {
  if (!chars) return "";
  const bytes = (chars * 3) / 4;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} kB`;
  return `${Math.round(bytes)} B`;
}

export function MediaStrip({ media }: { media: MediaEntry[] }) {
  const [expanded, setExpanded] = useState<string | null>(null); // src of the open image

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  return (
    <div
      data-testid="godmode-media-strip"
      className="flex flex-wrap items-center gap-2 px-3 py-1.5"
    >
      {media.map((m, i) => (
        <MediaItem key={m.media_id ?? m.url ?? i} entry={m} onExpand={setExpanded} />
      ))}
      {expanded && (
        <div
          data-testid="godmode-lightbox"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
          onClick={() => setExpanded(null)}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={expanded}
            alt="request image (full size)"
            className="max-h-full max-w-full rounded object-contain"
          />
        </div>
      )}
    </div>
  );
}

function MediaItem({
  entry,
  onExpand,
}: {
  entry: MediaEntry;
  onExpand: (src: string) => void;
}) {
  if (entry.dropped === "count") {
    return (
      <span
        data-testid="godmode-media-overflow"
        className="rounded bg-slate-800 px-2 py-1 text-[10px] text-slate-400"
      >
        +{entry.count ?? 0} more images
      </span>
    );
  }
  if (entry.dropped === "too_large") {
    return (
      <span
        data-testid="godmode-media-dropped"
        className="rounded bg-slate-800 px-2 py-1 text-[10px] text-slate-400"
      >
        {entry.mime ?? "image"} · {mediaSizeLabel(entry.chars)} · too large
      </span>
    );
  }
  if (entry.url) {
    return <Thumb src={entry.url} onExpand={onExpand} />;
  }
  if (entry.media_id) {
    return <StoredThumb mediaId={entry.media_id} onExpand={onExpand} />;
  }
  return null;
}

function Thumb({ src, onExpand }: { src: string; onExpand: (src: string) => void }) {
  return (
    <button
      type="button"
      data-testid="godmode-media-thumb"
      className="h-[72px] overflow-hidden rounded border border-slate-700 hover:border-slate-500"
      onClick={() => onExpand(src)}
      title="click to expand"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="request image" loading="lazy" className="h-full w-auto" />
    </button>
  );
}

/** Stored item: authed fetch -> blob object URL, revoked on unmount.
 *  404 -> "evicted" chip (the store aged it out). */
function StoredThumb({
  mediaId,
  onExpand,
}: {
  mediaId: string;
  onExpand: (src: string) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let alive = true;
    authFetch(`/api/admin/godmode/media/${mediaId}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(String(r.status));
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        // Unmounted (or mediaId changed) while the fetch/blob() promise was
        // still pending: cleanup already ran with objectUrl still null, so
        // nothing was revoked there. The URL is only created here, after
        // cleanup, so it must be revoked immediately rather than adopted —
        // otherwise it leaks for the life of the tab.
        if (alive) {
          objectUrl = url;
          setSrc(url);
        } else {
          URL.revokeObjectURL(url);
        }
      })
      .catch(() => {
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [mediaId]);

  if (failed) {
    return (
      <span
        data-testid="godmode-media-evicted"
        className="rounded bg-slate-800 px-2 py-1 text-[10px] text-slate-500"
      >
        image evicted
      </span>
    );
  }
  if (!src) {
    return (
      <span className="h-[72px] w-[72px] animate-pulse rounded bg-slate-800/60" />
    );
  }
  return <Thumb src={src} onExpand={onExpand} />;
}
