'use client';

import Link from 'next/link';
import { sendEvent } from './Track';

/** A link that records which call-to-action was clicked before navigating. */
export default function CtaLink({ href, where, className, children }: { href: string; where: string; className?: string; children: React.ReactNode }) {
  return (
    <Link href={href} className={className} onClick={() => sendEvent('cta_click', { where })}>
      {children}
    </Link>
  );
}
