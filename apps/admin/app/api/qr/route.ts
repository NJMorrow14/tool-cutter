import { NextResponse } from 'next/server';
import QRCode from 'qrcode';

export const dynamic = 'force-dynamic';

/** QR code for a URL: /api/qr?url=…&fmt=svg|png&size=1024 */
export async function GET(req: Request) {
  const u = new URL(req.url);
  const target = u.searchParams.get('url') || `${process.env.NEXT_PUBLIC_WEB_URL || 'http://localhost:3001'}/?src=brochure-qr`;
  const fmt = u.searchParams.get('fmt') === 'png' ? 'png' : 'svg';
  const size = Math.min(4096, Math.max(128, Number(u.searchParams.get('size') || 1024)));
  const download = u.searchParams.get('download') === '1';
  const opts = { errorCorrectionLevel: 'M' as const, margin: 2, color: { dark: '#1f2326', light: '#ffffff' } };
  if (fmt === 'png') {
    const buf = await QRCode.toBuffer(target, { ...opts, type: 'png', width: size });
    return new NextResponse(new Uint8Array(buf), { headers: { 'content-type': 'image/png', 'content-disposition': `${download ? 'attachment' : 'inline'}; filename="toolfoam-qr.png"` } });
  }
  const svg = await QRCode.toString(target, { ...opts, type: 'svg', width: size });
  return new NextResponse(svg, { headers: { 'content-type': 'image/svg+xml', 'content-disposition': `${download ? 'attachment' : 'inline'}; filename="toolfoam-qr.svg"` } });
}
