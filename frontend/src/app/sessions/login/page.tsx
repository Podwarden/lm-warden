"use client";

// /sessions/login — the old key holder sign-in. Key holders now sign in on /login, the API key tab; this address
// keeps working for links already handed out.

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { SESSIONS_LOGIN } from "@/lib/sessions/key-token";

export default function SessionsLoginRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(SESSIONS_LOGIN);
  }, [router]);
  return null;
}
