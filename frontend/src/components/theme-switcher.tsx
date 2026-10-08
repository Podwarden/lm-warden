"use client";

import { useTheme, type Theme, THEME_ORDER, THEME_META } from "@/lib/theme";

/**
 * macOS-style theme cycling button.
 * Each click advances: Solar Light → Solar Dark → Retro Light → Retro Dark → repeat.
 * Three horizontal slider lines with a shifting dot indicate the current theme.
 */
export function ThemeSwitcher({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();

  const idx = THEME_ORDER.indexOf(theme);
  const meta = THEME_META[theme];
  const nextIdx = (idx + 1) % THEME_ORDER.length;
  const nextTheme = THEME_ORDER[nextIdx];

  function cycle() {
    setTheme(nextTheme);
  }

  // Dot positions shift per theme to give visual feedback
  const dotPositions: Record<Theme, [number, number, number]> = {
    retro:      [8, 14, 5],
    "retro-dark": [14, 8, 11],
  };
  const dots = dotPositions[theme];

  return (
    <button
      type="button"
      onClick={cycle}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          cycle();
        }
      }}
      className={`
        relative flex items-center justify-center
        w-8 h-8 rounded-lg
        transition-all duration-200 ease-out
        hover:bg-white/[0.06] active:scale-95
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/40
        ${className ?? ""}
      `}
      aria-label={`Switch theme — current: ${meta.label}`}
      title={meta.label}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 20 20"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-transform duration-150"
      >
        {/* Three slider lines */}
        {[5, 10, 15].map((y, i) => (
          <g key={y}>
            {/* Track */}
            <line
              x1="3"
              y1={y}
              x2="17"
              y2={y}
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              className="opacity-30"
            />
            {/* Knob dot — position shifts per theme */}
            <circle
              cx={dots[i]}
              cy={y}
              r="2.2"
              fill={meta.dot}
              className="transition-all duration-300 ease-out"
            />
          </g>
        ))}
      </svg>

      {/* Tiny accent indicator dot (bottom-right corner) */}
      <span
        className="absolute bottom-0.5 right-0.5 w-1.5 h-1.5 rounded-full transition-colors duration-300"
        style={{ backgroundColor: meta.dot }}
      />
    </button>
  );
}
