"use client";

import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";

export type Edition = { edition: string; features: string[] };

/**
 * GET /api/edition (admin). Returns undefined while loading and on any error
 * (401/403 for non-admins or logged-out visitors included); gated UI treats
 * undefined as "no extra features".
 */
export function useEdition(): Edition | undefined {
  const { data } = useSWR<Edition>("/api/edition", authFetchJSON, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });
  return data ? { edition: data.edition, features: data.features ?? [] } : undefined;
}

export function hasFeature(e: Edition | undefined, feature: string): boolean {
  return !!e && e.features.includes(feature);
}
