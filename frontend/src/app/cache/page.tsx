'use client';

// /cache → /models?tab=files. The downloaded model weights page became the
// "Model files" tab of /models (2026-11-23) and left the main menu; this
// keeps old bookmarks and /ui/cache links working. A client redirect, like
// app/chat/page.tsx: the static export cannot run a server redirect().

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { MODEL_FILES_HREF } from '@/lib/routes';

export default function CacheRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(MODEL_FILES_HREF);
  }, [router]);
  return null;
}
