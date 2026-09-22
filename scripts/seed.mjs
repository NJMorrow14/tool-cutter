#!/usr/bin/env node
// Idempotent seed: loads brand/toolfoam-brochure.pdf into assets('brochure') if no brochure exists yet.
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
await client.connect();
const { rows } = await client.query("select key from assets where key = 'brochure'");
if (!rows.length) {
  const file = path.join(root, "brand", "toolfoam-brochure.pdf");
  if (existsSync(file)) {
    const bytes = readFileSync(file);
    await client.query(
      "insert into assets (key, filename, content_type, bytes, size) values ('brochure', $1, 'application/pdf', $2, $3)",
      ["toolfoam-brochure.pdf", bytes, bytes.length],
    );
    console.log(`seeded brochure (${bytes.length} bytes)`);
  } else console.log("no brand/toolfoam-brochure.pdf to seed");
} else console.log("brochure already present");
await client.end();
