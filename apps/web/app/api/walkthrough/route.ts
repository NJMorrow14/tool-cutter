import { NextResponse } from 'next/server';
import { hasDb, query } from '../../../lib/db';
import { trackEvent } from '../../../lib/track';

export const dynamic = 'force-dynamic';

const field = (v: unknown, max: number) => (typeof v === 'string' ? v.trim().slice(0, max) : '');

export async function POST(req: Request) {
  let body: Record<string, unknown> = {};
  try {
    body = (await req.json()) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ error: 'Bad request' }, { status: 400 });
  }
  const name = field(body.name, 120);
  const company = field(body.company, 160);
  const email = field(body.email, 200);
  const phone = field(body.phone, 60);
  const plant_location = field(body.plant_location, 200);
  const scope = field(body.scope, 200);
  const preferred_dates = field(body.preferred_dates, 200);
  const message = field(body.message, 2000);
  const src = field(body.src, 64) || null;
  const session_id = field(body.session_id, 64) || null;
  if (field(body.website, 200)) return NextResponse.json({ ok: true }); // honeypot
  if (!name || !company || !email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    return NextResponse.json({ error: 'Name, company and a valid email are required.' }, { status: 422 });
  }
  if (!hasDb()) return NextResponse.json({ error: 'The request desk is offline right now — please call or email us.' }, { status: 503 });
  try {
    const rows = await query<{ id: number }>(
      `insert into walkthrough_requests (name, company, email, phone, plant_location, scope, preferred_dates, message, src, session_id)
       values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) returning id`,
      [name, company, email, phone || null, plant_location || null, scope || null, preferred_dates || null, message || null, src, session_id],
    );
    await trackEvent({ name: 'walkthrough_submitted', path: '/walkthrough', src, session_id, props: { request_id: rows[0]?.id } }, req.headers);
    return NextResponse.json({ ok: true, id: rows[0]?.id });
  } catch (err) {
    console.error('walkthrough insert failed', err);
    return NextResponse.json({ error: 'Something went wrong on our side. Please try again or call us.' }, { status: 500 });
  }
}
