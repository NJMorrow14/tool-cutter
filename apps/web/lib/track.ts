import { createHash } from 'node:crypto';
import { hasDb, query } from './db';

export interface EventInput {
  name: string;
  path?: string | null;
  src?: string | null;
  referrer?: string | null;
  session_id?: string | null;
  props?: Record<string, unknown>;
}

/** Anonymous per-day visitor fingerprint (address is never stored). */
export function visitorHash(ip: string | null, ua: string | null): string {
  const day = new Date().toISOString().slice(0, 10);
  return createHash('sha256').update(`${ip ?? ''}|${ua ?? ''}|${day}|${process.env.SESSION_SECRET ?? 'tf'}`).digest('hex').slice(0, 32);
}

export function clientIp(headers: Headers): string | null {
  const fwd = headers.get('x-forwarded-for');
  if (fwd) return fwd.split(',')[0].trim();
  return headers.get('cf-connecting-ip') ?? headers.get('x-real-ip');
}

/** Analytics must never break the page: failures are swallowed. */
export async function trackEvent(e: EventInput, headers?: Headers): Promise<void> {
  if (!hasDb()) return;
  try {
    const ua = headers?.get('user-agent') ?? null;
    await query(
      `insert into events (name, path, src, referrer, session_id, visitor_hash, user_agent, props)
       values ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [
        e.name.slice(0, 64),
        e.path?.slice(0, 512) ?? null,
        e.src?.slice(0, 64) ?? null,
        e.referrer?.slice(0, 512) ?? null,
        e.session_id?.slice(0, 64) ?? null,
        headers ? visitorHash(clientIp(headers), ua) : null,
        ua?.slice(0, 256) ?? null,
        JSON.stringify(e.props ?? {}),
      ],
    );
  } catch (err) {
    console.error('track failed', err instanceof Error ? err.message : err);
  }
}
