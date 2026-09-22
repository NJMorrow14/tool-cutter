import { NextResponse } from 'next/server';
import { trackEvent } from '../../../lib/track';

export const dynamic = 'force-dynamic';

const ALLOWED = new Set(['page_view', 'qr_scan', 'cta_click', 'brochure_download', 'form_started']);

export async function POST(req: Request) {
  let body: Record<string, unknown> = {};
  try {
    body = (await req.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ ok: false }, { status: 400 });
  }
  const name = typeof body.name === 'string' ? body.name : '';
  if (!ALLOWED.has(name)) return NextResponse.json({ ok: false }, { status: 400 });
  const str = (v: unknown) => (typeof v === 'string' ? v : null);
  await trackEvent(
    {
      name,
      path: str(body.path),
      src: str(body.src),
      referrer: str(body.referrer),
      session_id: str(body.session_id),
      props: typeof body.props === 'object' && body.props ? (body.props as Record<string, unknown>) : {},
    },
    req.headers,
  );
  return NextResponse.json({ ok: true });
}
