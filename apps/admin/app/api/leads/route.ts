import { NextResponse } from 'next/server';
import { hasDb, query } from '../../../lib/db';

export const dynamic = 'force-dynamic';

const STATUSES = new Set(['new', 'contacted', 'scheduled', 'done', 'lost']);

/** GET ?format=csv exports every request; PATCH {id, status?, notes?} updates one. */
export async function GET(req: Request) {
  if (!hasDb()) return new NextResponse('DATABASE_URL not set', { status: 503 });
  const rows = await query<Record<string, unknown>>(
    `select id, created_at, status, name, company, email, phone, plant_location, scope, preferred_dates, message, src, notes
     from walkthrough_requests order by created_at desc`,
  );
  if (new URL(req.url).searchParams.get('format') !== 'csv') return NextResponse.json({ rows });
  const cols = ['id', 'created_at', 'status', 'name', 'company', 'email', 'phone', 'plant_location', 'scope', 'preferred_dates', 'message', 'src', 'notes'];
  const esc = (v: unknown) => { const s = v == null ? '' : v instanceof Date ? v.toISOString() : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n');
  return new NextResponse(csv, { headers: { 'content-type': 'text/csv', 'content-disposition': 'attachment; filename="walkthrough-requests.csv"' } });
}

export async function PATCH(req: Request) {
  if (!hasDb()) return NextResponse.json({ error: 'DATABASE_URL not set' }, { status: 503 });
  const body = (await req.json().catch(() => ({}))) as { id?: number; status?: string; notes?: string };
  const id = Number(body.id);
  if (!Number.isFinite(id)) return NextResponse.json({ error: 'id required' }, { status: 400 });
  if (body.status !== undefined) {
    if (!STATUSES.has(body.status)) return NextResponse.json({ error: 'bad status' }, { status: 400 });
    await query(`update walkthrough_requests set status = $2 where id = $1`, [id, body.status]);
  }
  if (body.notes !== undefined) await query(`update walkthrough_requests set notes = $2 where id = $1`, [id, String(body.notes).slice(0, 4000)]);
  return NextResponse.json({ ok: true });
}
