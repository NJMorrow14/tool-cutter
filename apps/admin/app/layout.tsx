import type { Metadata, Viewport } from 'next';
import './globals.css';
import AdminNav from '../components/AdminNav';
import { adminEnabled } from '../lib/session';

export const metadata: Metadata = {
  title: 'ToolFoam Pro — HQ',
  description: 'Foam workflow, walkthrough requests, brochure and analytics.',
  robots: { index: false, follow: false },
};

export const viewport: Viewport = { width: 'device-width', initialScale: 1 };

// Chrome only — access is enforced by middleware.ts once ADMIN_PASSWORD is set; in production the whole
// site additionally sits behind Cloudflare Access.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AdminNav walled={adminEnabled()} webUrl={process.env.NEXT_PUBLIC_WEB_URL || 'http://localhost:3001'} />
        {children}
      </body>
    </html>
  );
}
