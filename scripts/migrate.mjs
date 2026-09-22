#!/usr/bin/env node
// Applies db/migrations/*.sql in filename order, tracking in schema_migrations.
// Usage: DATABASE_URL=postgres://... node scripts/migrate.mjs
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "db", "migrations");
const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
await client.connect();

await client.query(
  "create table if not exists schema_migrations (name text primary key, applied_at timestamptz default now())",
);
const { rows } = await client.query("select name from schema_migrations");
const applied = new Set(rows.map((r) => r.name));

for (const file of readdirSync(dir).filter((f) => f.endsWith(".sql")).sort()) {
  if (applied.has(file)) continue;
  process.stdout.write(`applying ${file}... `);
  const sql = readFileSync(path.join(dir, file), "utf8");
  try {
    await client.query("begin");
    await client.query(sql);
    await client.query("insert into schema_migrations (name) values ($1)", [file]);
    await client.query("commit");
    console.log("ok");
  } catch (err) {
    await client.query("rollback");
    console.error(`FAILED: ${err.message}`);
    process.exit(1);
  }
}
console.log("migrations up to date");
await client.end();
