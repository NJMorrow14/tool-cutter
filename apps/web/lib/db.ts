import { Pool, types } from 'pg';

// Postgres bigint arrives as a string; our ids fit in JS numbers.
types.setTypeParser(types.builtins.INT8, (v) => parseInt(v, 10));

let pool: Pool | null = null;

export function hasDb(): boolean {
  return !!process.env.DATABASE_URL;
}

export function db(): Pool {
  if (!pool) {
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error('DATABASE_URL is not set');
    pool = new Pool({ connectionString: url, max: 5 });
  }
  return pool;
}

export async function query<T = Record<string, unknown>>(text: string, params: unknown[] = []): Promise<T[]> {
  const res = await db().query(text, params);
  return res.rows as T[];
}

export async function queryOne<T = Record<string, unknown>>(text: string, params: unknown[] = []): Promise<T | null> {
  const rows = await query<T>(text, params);
  return rows[0] ?? null;
}
