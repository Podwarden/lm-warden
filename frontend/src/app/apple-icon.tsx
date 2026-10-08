// Apple touch icon for LM Warden (#152).
//
// Next.js App Router only accepts `.jpg`, `.jpeg`, or `.png` for the
// `apple-icon` file convention — SVG is rejected, so we cannot reuse
// the sibling `icon.svg` directly. Generating the PNG at build time via
// `ImageResponse` keeps a single source of truth (no separate raster
// asset checked into the repo) and lets the icon stay statically
// optimized — Next will cache the 180x180 PNG output and serve it from
// /apple-icon. The visual matches `icon.svg`: filled emerald shield
// with a white "W" wordmark, sized for iOS home-screen.
import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

// Lucide `Shield` path (the same silhouette the navbar renders), closed
// with `Z` so the flood-fill renders as a solid shape rather than a
// stroke.
const SHIELD_PATH =
  "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z";

// Same four-stroke "W" used in icon.svg — keeps the apple-icon visually
// identical to the favicon when both are rendered next to each other.
const W_PATH = "M8.4 9.5 L9.9 14.5 L12 11 L14.1 14.5 L15.6 9.5";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          // iOS masks apple-touch-icons into a rounded square. A pure
          // white background gives clean edges after the mask and keeps
          // the emerald shield visually consistent with the tab favicon.
          background: "#ffffff",
        }}
      >
        <svg
          width="160"
          height="160"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path fill="#10b981" d={SHIELD_PATH} />
          <path
            fill="none"
            stroke="#ffffff"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            d={W_PATH}
          />
        </svg>
      </div>
    ),
    { ...size },
  );
}
