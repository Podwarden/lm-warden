// /chat — retired 2026-08-23 in favour of /chat2 (persisted chats, markdown,
// attachments). Old bookmarks and muscle memory land here, so the route
// permanently forwards instead of 404ing. The playground backend endpoints
// (/api/chat/playground/ensure, /api/chat/completions) are still served; only
// this UI is gone.
import { redirect } from 'next/navigation';

export default function ChatRetired() {
  redirect('/chat2');
}
