'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useState } from 'react';
import ad from '../app/admin.module.css';
import Mark from './Mark';

const LINKS: [string, string][] = [
  ['/', 'Studio'],
  ['/analytics', 'Analytics'],
  ['/leads', 'Walkthrough requests'],
  ['/brochure', 'Brochure & QR'],
];

export default function AdminNav({ walled, webUrl }: { walled: boolean; webUrl: string }) {
  const path = usePathname();
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  if (path === '/login') return null;
  return (
    <nav className={ad.nav} aria-label="Admin">
      <div className={ad.navInner}>
        <span className={ad.navBrand}>
          <Mark size={22} />
          <span>TOOLFOAM<em>PRO</em> <span style={{ color: '#8b939c', fontWeight: 400 }}>· HQ</span></span>
        </span>
        {path === '/' && <button className={ad.workspaceMenu} aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>Workspace ▾</button>}
        {(path !== '/' || menuOpen) && LINKS.map(([href, label]) => (
          <Link key={href} href={href} className={path === href ? 'active' : ''}>{label}</Link>
        ))}
        <span className={ad.navSpacer} />
        {path !== '/' && <a href={webUrl} target="_blank" rel="noopener" className={ad.navMuted}>Landing site ↗</a>}
        {walled ? (
          <a href="#" onClick={async (e) => { e.preventDefault(); await fetch('/api/login', { method: 'DELETE' }); router.push('/login'); }}>Log out</a>
        ) : (
          <span className={ad.navMuted} title="Set ADMIN_PASSWORD to enable the login wall">Local workspace</span>
        )}
      </div>
    </nav>
  );
}
