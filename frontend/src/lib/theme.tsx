"use client";

import { createContext, useContext, useState, useEffect, type ReactNode } from "react";

export type Theme = "retro" | "retro-dark";

/** Fixed cycle order */
export const THEME_ORDER: Theme[] = ["retro", "retro-dark"];

/** Display metadata per theme */
export const THEME_META: Record<Theme, { label: string; dot: string }> = {
  retro:        { label: "Retro Light",  dot: "#b45309" },
  "retro-dark": { label: "Retro Dark",   dot: "#f59e0b" },
};

interface ThemeContextValue {
  theme: Theme;
  setTheme: (t: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "retro-dark",
  setTheme: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

export function ThemeProvider({
  children,
  initialTheme = "retro-dark",
}: {
  children: ReactNode;
  initialTheme?: Theme;
}) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("vw-theme") as Theme | null;
      if (stored && THEME_ORDER.includes(stored)) return stored;
    }
    return initialTheme;
  });

  function setTheme(t: Theme) {
    setThemeState(t);
    localStorage.setItem("vw-theme", t);
    applyTheme(t);
  }

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function applyTheme(theme: Theme) {
  const html = document.documentElement;
  html.setAttribute("data-theme", theme);

  if (theme === "retro") {
    html.classList.remove("dark");
  } else {
    html.classList.add("dark");
  }
  // DM Sans (both retro themes) is self-hosted: app/layout.tsx imports
  // fonts/dm-sans.css, so no font is ever fetched from a third-party host.
}
