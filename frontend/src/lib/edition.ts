"use client";

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";

export type ConsoleLink = { label: string; href: string };
export type Edition = { edition: string; features: string[]; console_links: ConsoleLink[] };

/**
 * GET /api/edition (admin). Returns undefined while loading and on any error
 * (401/403 for non-admins or logged-out visitors included); gated UI treats
 * undefined as "no extra features".
 */
export function useEdition(enabled = true): Edition | undefined {
  const { data } = useSWR<Partial<Edition> & { edition: string }>(enabled ? "/api/edition" : null, authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  return data
    ? { edition: data.edition, features: data.features ?? [], console_links: data.console_links ?? [] }
    : undefined;
}

export function hasFeature(e: Edition | undefined, feature: string): boolean {
  return !!e && e.features.includes(feature);
}
