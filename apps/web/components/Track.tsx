'use client';

import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';

const SID_KEY = 'tf_sid';
const SRC_KEY = 'tf_src';

export function sessionId(): string {
  try {
    let sid = localStorage.getItem(SID_KEY);
    if (!sid) {
      sid = Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem(SID_KEY, sid);
    }
    return sid;
  } catch {
    return 'anon';
  }
}

/** Attribution: the first ?src= of a visit (e.g. the brochure's QR code) sticks for the whole session. */
export function visitSource(): string | null {
  try {
    return sessionStorage.getItem(SRC_KEY);
  } catch {
    return null;
  }
}

export function sendEvent(name: string, props: Record<string, unknown> = {}) {
  const body = JSON.stringify({
    name, props,
    path: location.pathname,
    src: visitSource(),
    referrer: document.referrer || null,
    session_id: sessionId(),
  });
  try {
    if (navigator.sendBeacon) navigator.sendBeacon('/api/events', new Blob([body], { type: 'application/json' }));
    else void fetch('/api/events', { method: 'POST', headers: { 'content-type': 'application/json' }, body, keepalive: true });
  } catch {
    /* analytics never breaks the page */
  }
}

/** Mount once in the root layout: records a page_view per route change and pins ?src= attribution. */
export default function Track() {
  const pathname = usePathname();
  const params = useSearchParams();
  useEffect(() => {
    const src = params.get('src');
    try {
      if (src && !sessionStorage.getItem(SRC_KEY)) sessionStorage.setItem(SRC_KEY, src.slice(0, 64));
    } catch { /* ignore */ }
    sendEvent(src ? 'qr_scan' : 'page_view', src ? { landing: pathname } : {});
    if (src) sendEvent('page_view');
  }, [pathname, params]);
  return null;
}
