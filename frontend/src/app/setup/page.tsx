import { redirect } from "next/navigation";

/**
 * Root /setup page — redirects to the first step of the wizard.
 *
 * The /setup segment has a layout.tsx and five child pages
 * (welcome, gpus, hf-token, admin, done) but until v2026.05.15.5 was
 * missing this root page.tsx, so https://vllm.example.com/setup
 * returned 404. Standard Next.js App Router fix: a server component
 * that delegates to next/navigation's redirect() so the browser lands
 * on /setup/welcome without ever rendering an interstitial.
 *
 * Closes vllm-warden#41.
 */
export default function SetupRootPage() {
  redirect("/setup/welcome");
}
