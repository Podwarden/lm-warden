import './globals.css';
import { ThemeProvider } from '@/lib/theme';
import { NavBar } from '@/components/nav-bar';
import { SessionGate } from '@/components/session-gate';
import { BreadcrumbHeader } from '@/components/breadcrumb-header';
import { NavStackProvider } from '@/lib/nav-stack';

export const metadata = { title: 'LLM Warden', description: 'LLM operator UI' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          {/* NavStackProvider holds the session's back-stack and the page
              titles pages register with useBreadcrumb; it wraps <main> so
              the pages can reach it. BreadcrumbHeader is the one-row trail
              + back button under the nav (hidden with it on /login, /setup). */}
          <NavStackProvider>
            <NavBar />
            <BreadcrumbHeader />
            {/* id="app-root" is the target of @podwarden/chat-ui's Modal
                rootInertId — chat2/page.tsx passes rootInertId="app-root" so
                an open modal can set `inert` on everything outside itself. */}
            {/* SessionGate holds the page back until the session is known, so a
                signed-out visitor never sees a flash of the Models page before
                being sent to /login. It wraps the children rather than <main>
                so id="app-root" is always in the DOM for chat-ui's inert
                handling (tests/contract/app-root.test.ts). */}
            <main id="app-root" className="container mx-auto p-6">
              <SessionGate>{children}</SessionGate>
            </main>
          </NavStackProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
