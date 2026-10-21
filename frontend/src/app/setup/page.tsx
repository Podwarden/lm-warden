'use client';

// /setup root — forwards to the first wizard step. The /setup segment has a
// layout.tsx and five child pages (welcome, gpus, hf-token, admin, done);
// without this page /setup 404s (closes vllm-warden#41).
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function SetupRootPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace('/setup/welcome');
  }, [router]);
  return null;
}
