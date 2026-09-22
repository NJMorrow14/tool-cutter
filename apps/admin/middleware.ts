import { NextResponse, type NextRequest } from 'next/server';
import { ADMIN_COOKIE, adminEnabled, verifyToken } from './lib/session';

// Every page and API of the admin site requires the admin cookie once ADMIN_PASSWORD is set.
// /login and /api/login stay open; static assets are excluded by the matcher.
export async function middleware(req: NextRequest) {
  if (!adminEnabled()) return NextResponse.next();
  const { pathname } = req.nextUrl;
  if (pathname === '/login' || pathname === '/api/login') return NextResponse.next();
  if (await verifyToken(req.cookies.get(ADMIN_COOKIE)?.value)) return NextResponse.next();
  if (pathname.startsWith('/api/')) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  const url = req.nextUrl.clone();
  url.pathname = '/login';
  url.search = `?next=${encodeURIComponent(pathname + req.nextUrl.search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|icon.svg|favicon.ico).*)'],
};
