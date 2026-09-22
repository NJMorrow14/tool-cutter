import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { NextResponse } from 'next/server';
import { hasDb, queryOne } from '../../lib/db';
import { trackEvent } from '../../lib/track';

export const dynamic = 'force-dynamic';

/** The current brochure PDF (admin-uploaded, stored in the database; falls back to the file in brand/). */
export async function GET(req: Request) {
  let bytes: Buffer | null = null;
  let filename = 'toolfoam-brochure.pdf';
  if (hasDb()) {
    try {
      const row = await queryOne<{ bytes: Buffer; filename: string }>(`select bytes, filename from assets where key = 'brochure'`);
      if (row) { bytes = row.bytes; filename = row.filename; }
    } catch (err) {
      console.error('brochure lookup failed', err);
    }
  }
  if (!bytes) {
    try {
      bytes = await readFile(path.join(process.cwd(), '..', '..', 'brand', 'toolfoam-brochure.pdf'));
    } catch {
      return new NextResponse('No brochure uploaded yet', { status: 404 });
    }
  }
  const url = new URL(req.url);
  await trackEvent({ name: 'brochure_download', path: '/brochure', src: url.searchParams.get('src') }, req.headers);
  return new NextResponse(new Uint8Array(bytes), {
    headers: {
      'content-type': 'application/pdf',
      'content-disposition': `${url.searchParams.get('download') ? 'attachment' : 'inline'}; filename="${filename}"`,
      'cache-control': 'no-store',
    },
  });
}
