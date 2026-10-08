import type { Config } from "tailwindcss";

// eslint-disable-next-line @typescript-eslint/no-require-imports
const chatUiPreset = require('@podwarden/chat-ui/tailwind-preset') as Partial<Config>;

const config: Config = {
  presets: [chatUiPreset],
  darkMode: ["class"],
  content: [
    // EVERY directory that contains className strings must be listed: a class
    // used only in an unscanned file silently generates no CSS (the chat2
    // "-m-6"/"-translate-x-1/2" purge bug, 2026-08-24).
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
    // the shared chat UI ships source-level Tailwind classes; scan its dist
    "./node_modules/@podwarden/chat-ui/dist/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        // Token details page (spec 2026-09-18 §4.6). Values live in globals.css
        // for every theme; keep this list and VW_TOKENS in
        // components/tokens/detail/styles.ts in sync (contract-tested).
        vw: {
          'rule-soft': 'rgb(var(--vw-rule-soft) / <alpha-value>)',
          'btn': 'rgb(var(--vw-btn) / <alpha-value>)',
          'btn-hover': 'rgb(var(--vw-btn-hover) / <alpha-value>)',
          'on-btn': 'rgb(var(--vw-on-btn) / <alpha-value>)',
          'danger': 'rgb(var(--vw-danger) / <alpha-value>)',
          'danger-fg': 'rgb(var(--vw-danger-fg) / <alpha-value>)',
          'danger-bg': 'rgb(var(--vw-danger-bg) / <alpha-value>)',
          'amber-bg': 'rgb(var(--vw-amber-bg) / <alpha-value>)',
          'amber-fg': 'rgb(var(--vw-amber-fg) / <alpha-value>)',
          'band-fg': 'rgb(var(--vw-band-fg) / <alpha-value>)',
          'ok-bg': 'rgb(var(--vw-ok-bg) / <alpha-value>)',
          'ok-fg': 'rgb(var(--vw-ok-fg) / <alpha-value>)',
          'live': 'rgb(var(--vw-live) / <alpha-value>)',
          'info-bg': 'rgb(var(--vw-info-bg) / <alpha-value>)',
          'info-fg': 'rgb(var(--vw-info-fg) / <alpha-value>)',
          'low-bg': 'rgb(var(--vw-low-bg) / <alpha-value>)',
          'low-fg': 'rgb(var(--vw-low-fg) / <alpha-value>)',
          'dock': 'rgb(var(--vw-dock) / <alpha-value>)',
          'prompt': 'rgb(var(--vw-prompt) / <alpha-value>)',
          'completion': 'rgb(var(--vw-completion) / <alpha-value>)',
          'requests': 'rgb(var(--vw-requests) / <alpha-value>)',
          'queue': 'rgb(var(--vw-queue) / <alpha-value>)',
          'ttft': 'rgb(var(--vw-ttft) / <alpha-value>)',
          'dur': 'rgb(var(--vw-dur) / <alpha-value>)',
          'strip-chain': 'rgb(var(--vw-strip-chain) / <alpha-value>)',
          'strip-own': 'rgb(var(--vw-strip-own) / <alpha-value>)',
          'model-1': 'rgb(var(--vw-model-1) / <alpha-value>)',
          'model-2': 'rgb(var(--vw-model-2) / <alpha-value>)',
          'model-3': 'rgb(var(--vw-model-3) / <alpha-value>)',
          'model-4': 'rgb(var(--vw-model-4) / <alpha-value>)',
          'model-5': 'rgb(var(--vw-model-5) / <alpha-value>)',
          'model-6': 'rgb(var(--vw-model-6) / <alpha-value>)',
        },
      },
    },
  },
  plugins: [],
};

export default config;
