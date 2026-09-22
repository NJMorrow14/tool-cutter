-- Landing-page analytics, walkthrough requests and admin-managed assets (brochure PDF).

create table events (
  id bigint generated always as identity primary key,
  name text not null,                 -- page_view | walkthrough_submitted | brochure_download | qr_scan …
  path text,
  src text,                           -- attribution: ?src= on the first hit of the visit (e.g. brochure-qr)
  referrer text,
  session_id text,                    -- per-browser id (localStorage), for uniques + journeys
  visitor_hash text,                  -- sha256(ip + ua + day salt): uniques without storing the address
  user_agent text,
  props jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create index events_name_idx on events (name, created_at);
create index events_created_idx on events (created_at);
create index events_session_idx on events (session_id, created_at);

create table walkthrough_requests (
  id bigint generated always as identity primary key,
  name text not null,
  company text not null,
  email text not null,
  phone text,
  plant_location text,
  scope text,                         -- how many drawers / benches / toolboxes
  preferred_dates text,
  message text,
  src text,
  session_id text,
  status text not null default 'new', -- new | contacted | scheduled | done | lost
  notes text,
  created_at timestamptz not null default now()
);
create index walkthrough_requests_created_idx on walkthrough_requests (created_at);

-- Small admin-managed binary assets (the brochure PDF, a few MB). One row per key.
create table assets (
  key text primary key,
  filename text not null,
  content_type text not null,
  bytes bytea not null,
  size integer not null,
  updated_at timestamptz not null default now()
);
