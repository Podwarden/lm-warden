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

  // Dynamically load DM Sans when switching to retro themes
  const needsFont = true; // Both retro themes use DM Sans
  const fontLinkId = "dm-sans-font";
  const existing = document.getElementById(fontLinkId);
  if (needsFont && !existing) {
    const link = document.createElement("link");
    link.id = fontLinkId;
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap";
    document.head.appendChild(link);
  }
}
