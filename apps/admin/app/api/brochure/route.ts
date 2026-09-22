import { NextResponse } from 'next/server';
import { hasDb, query, queryOne } from '../../../lib/db';

export const dynamic = 'force-dynamic';

const MAX_BYTES = 25 * 1024 * 1024;

/** GET: the current brochure PDF. POST (multipart, field "file"): replace it. */
export async function GET() {
  if (!hasDb()) return new NextResponse('DATABASE_URL not set', { status: 503 });
  const row = await queryOne<{ bytes: Buffer; filename: string; content_type: string }>(`select bytes, filename, content_type from assets where key = 'brochure'`);
  if (!row) return new NextResponse('No brochure uploaded yet', { status: 404 });
  return new NextResponse(new Uint8Array(row.bytes), {
    headers: { 'content-type': row.content_type, 'content-disposition': `inline; filename="${row.filename}"`, 'cache-control': 'no-store' },
  });
}

export async function POST(req: Request) {
  if (!hasDb()) return NextResponse.json({ error: 'DATABASE_URL not set' }, { status: 503 });
  const form = await req.formData();
  const file = form.get('file');
  if (!(file instanceof File)) return NextResponse.json({ error: 'No file' }, { status: 400 });
  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) return NextResponse.json({ error: 'Upload a PDF' }, { status: 415 });
  if (file.size > MAX_BYTES) return NextResponse.json({ error: 'PDF larger than 25 MB' }, { status: 413 });
  const bytes = Buffer.from(await file.arrayBuffer());
  const filename = file.name.replace(/[^A-Za-z0-9._-]+/g, '_') || 'brochure.pdf';
  await query(
    `insert into assets (key, filename, content_type, bytes, size) values ('brochure', $1, 'application/pdf', $2, $3)
     on conflict (key) do update set filename = excluded.filename, bytes = excluded.bytes, size = excluded.size, updated_at = now()`,
    [filename, bytes, bytes.length],
  );
  return NextResponse.json({ ok: true, filename, size: bytes.length });
}
