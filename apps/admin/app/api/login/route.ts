import { NextResponse } from 'next/server';
import { ADMIN_COOKIE, adminEnabled, makeToken } from '../../../lib/session';

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  if (!adminEnabled()) return NextResponse.json({ ok: true, note: 'admin password not configured; wall is off' });
  const body = (await req.json().catch(() => ({}))) as { password?: string };
  const given = body.password ?? '';
  const want = process.env.ADMIN_PASSWORD ?? '';
  let diff = given.length === want.length ? 0 : 1;
  for (let i = 0; i < Math.min(given.length, want.length); i++) diff |= given.charCodeAt(i) ^ want.charCodeAt(i);
  if (diff !== 0) return NextResponse.json({ error: 'Wrong password' }, { status: 401 });
  const res = NextResponse.json({ ok: true });
  res.cookies.set(ADMIN_COOKIE, await makeToken(), {
    httpOnly: true, sameSite: 'lax', secure: process.env.NODE_ENV === 'production', path: '/', maxAge: 60 * 60 * 12,
  });
  return res;
}

export async function DELETE() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set(ADMIN_COOKIE, '', { httpOnly: true, path: '/', maxAge: 0 });
  return res;
}
