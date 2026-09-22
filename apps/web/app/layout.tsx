import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import { Suspense } from 'react';
import './globals.css';
import styles from './layout.module.css';
import Track from '../components/Track';
import CtaLink from '../components/CtaLink';
import Logo from '../components/Logo';
import { REGION_SHORT, CITIES } from '../lib/region';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter', weight: ['300', '400', '500', '600', '700'] });

export const metadata: Metadata = {
  title: 'ToolFoam Pro — Custom foam tool drawers, Carolina Piedmont',
  description: `Custom-cut foam drawer systems, designed and installed in your facility. On-site across the Carolina Piedmont — ${CITIES}. We measure, we design, we install — 24-hour turnaround.`,
  metadataBase: new URL(process.env.NEXT_PUBLIC_WEB_URL || 'http://localhost:3001'),
  openGraph: { title: 'ToolFoam Pro — A place for every tool', description: 'On-site industrial tool organization. Custom-cut foam drawer systems with 24-hour turnaround.', images: ['/img/hero.jpg'] },
};

export const viewport: Viewport = { width: 'device-width', initialScale: 1, themeColor: '#1f2326' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <Suspense fallback={null}><Track /></Suspense>
        <header className={styles.header}>
          <div className={`wrap ${styles.headerInner}`}>
            <Link href="/" className={styles.logoLink} aria-label="ToolFoam Pro home"><Logo /></Link>
            <span className={styles.tagline}>On-site tool organization · {REGION_SHORT}</span>
            <nav className={styles.nav} aria-label="Main">
              <Link href="/#process">Process</Link>
              <Link href="/#industries">Industries</Link>
              <a href="/brochure" target="_blank" rel="noopener">Brochure</a>
              <CtaLink href="/walkthrough" where="nav" className={`btn btnOrange ${styles.navCta}`}>Book a walkthrough</CtaLink>
            </nav>
          </div>
        </header>
        {children}
        <footer className={styles.footer}>
          <div className={`wrap ${styles.footerInner}`}>
            <span>TOOLFOAM PRO · On-site tool organization</span>
            <span>Serving the {REGION_SHORT} — {CITIES}</span>
            <span>A place for every tool.</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
