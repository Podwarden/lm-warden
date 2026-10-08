// `page.tsx` is a client component ('use client'), and Next forbids exporting
// `metadata` from one. The route's title therefore lives here, in a server
// component that does nothing but pass its children through.
import type { ReactNode } from 'react';

export const metadata = { title: 'Chat 2 · LM Warden' };

export default function Chat2Layout({ children }: { children: ReactNode }) {
  return children;
}
