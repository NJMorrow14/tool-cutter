// HMAC-signed admin session cookie (Web Crypto so the same code runs in middleware and route handlers).
// In production the admin site also sits behind Cloudflare Access; this is the app-level lock.
// With no ADMIN_PASSWORD configured (local dev) the wall is off entirely.

export const ADMIN_COOKIE = 'tc_admin';
const TTL_MS = 1000 * 60 * 60 * 12;

export function adminEnabled(): boolean {
  return !!process.env.ADMIN_PASSWORD;
}

function secret(): string {
  return process.env.SESSION_SECRET || 'dev-secret-not-for-prod';
}

const b64url = (buf: ArrayBuffer) => Buffer.from(buf).toString('base64url');

async function sign(payload: string): Promise<string> {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret()), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return b64url(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(payload)));
}

export async function makeToken(): Promise<string> {
  const payload = `admin.${Date.now() + TTL_MS}`;
  return `${payload}.${await sign(payload)}`;
}

export async function verifyToken(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  const idx = token.lastIndexOf('.');
  if (idx < 0) return false;
  const payload = token.slice(0, idx);
  const sig = token.slice(idx + 1);
  const expected = await sign(payload);
  if (sig.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < sig.length; i++) diff |= sig.charCodeAt(i) ^ expected.charCodeAt(i);
  if (diff !== 0) return false;
  const expiry = Number(payload.split('.')[1]);
  return Number.isFinite(expiry) && expiry > Date.now();
}

/** Server components / route handlers: is this request an authenticated admin (or is the wall off)? */
export async function isAdmin(): Promise<boolean> {
  if (!adminEnabled()) return true;
  const { cookies } = await import('next/headers');
  return verifyToken(cookies().get(ADMIN_COOKIE)?.value);
}
