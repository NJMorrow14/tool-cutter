import Link from 'next/link';
import ad from '../admin.module.css';
import { hasDb, query } from '../../lib/db';

export const dynamic = 'force-dynamic';

type Day = { day: string; views: number; visitors: number; scans: number; requests: number };

function Bars({ days, field, color }: { days: Day[]; field: keyof Day; color?: string }) {
  const max = Math.max(1, ...days.map((d) => Number(d[field])));
  return (
    <div>
      <div className={ad.bars}>
        {days.map((d) => (
          <div key={d.day} className={ad.bar} style={{ height: `${(Number(d[field]) / max) * 100}%`, background: color, minHeight: Number(d[field]) ? 3 : 1, opacity: Number(d[field]) ? 1 : 0.25 }}>
            <span>{d.day.slice(5)} · {d[field]}</span>
          </div>
        ))}
      </div>
      <div className={ad.barsAxis}><span>{days[0]?.day}</span><span>{days[days.length - 1]?.day}</span></div>
    </div>
  );
}

function Ranked({ rows, label }: { rows: { key: string; count: number }[]; label: string }) {
  const max = Math.max(1, ...rows.map((r) => r.count));
  if (!rows.length) return <p style={{ color: 'var(--muted)', margin: 0 }}>Nothing yet.</p>;
  return (
    <div className={ad.rows}>
      {rows.map((r) => (
        <div key={r.key} className={ad.row}>
          <span className={label === 'path' ? ad.mono : ''}>{r.key || '—'}</span><strong>{r.count}</strong>
          <span className="track"><i style={{ width: `${(r.count / max) * 100}%` }} /></span>
        </div>
      ))}
    </div>
  );
}

export default async function AnalyticsPage({ searchParams }: { searchParams: { days?: string } }) {
  const days = [7, 30, 90].includes(Number(searchParams.days)) ? Number(searchParams.days) : 30;
  if (!hasDb()) {
    return (
      <main className={ad.page}>
        <div className={ad.pageHead}><h1>Analytics</h1></div>
        <div className={ad.notice}>Set <code>DATABASE_URL</code> (see .env.example) and run <code>npm run migrate</code> to collect landing-page analytics.</div>
      </main>
    );
  }
  const since = `now() - interval '${days} days'`;
  const [totals, perDay, paths, sources, referrers, ctas, funnel] = await Promise.all([
    query<{ views: number; visitors: number; sessions: number; scans: number; downloads: number; requests: number; cta: number; forms: number }>(
      `select
         count(*) filter (where name = 'page_view')::int as views,
         count(distinct visitor_hash) filter (where name = 'page_view')::int as visitors,
         count(distinct session_id) filter (where name = 'page_view')::int as sessions,
         count(*) filter (where name = 'qr_scan')::int as scans,
         count(*) filter (where name = 'brochure_download')::int as downloads,
         count(*) filter (where name = 'walkthrough_submitted')::int as requests,
         count(*) filter (where name = 'cta_click')::int as cta,
         count(*) filter (where name = 'form_started')::int as forms
       from events where created_at > ${since}`,
    ),
    query<Day>(
      `select to_char(d::date, 'YYYY-MM-DD') as day,
         coalesce(e.views, 0)::int as views, coalesce(e.visitors, 0)::int as visitors, coalesce(e.scans, 0)::int as scans, coalesce(e.requests, 0)::int as requests
       from generate_series((now() - interval '${days - 1} days')::date, now()::date, '1 day') d
       left join (
         select created_at::date as day,
           count(*) filter (where name = 'page_view') as views,
           count(distinct visitor_hash) filter (where name = 'page_view') as visitors,
           count(*) filter (where name = 'qr_scan') as scans,
           count(*) filter (where name = 'walkthrough_submitted') as requests
         from events where created_at > ${since} group by 1
       ) e on e.day = d::date order by d`,
    ),
    query<{ key: string; count: number }>(`select coalesce(path, '—') as key, count(*)::int from events where name = 'page_view' and created_at > ${since} group by 1 order by 2 desc limit 10`),
    query<{ key: string; count: number }>(`select coalesce(src, 'direct / organic') as key, count(distinct session_id)::int from events where name = 'page_view' and created_at > ${since} group by 1 order by 2 desc limit 10`),
    query<{ key: string; count: number }>(
      `select coalesce(nullif(substring(referrer from '^https?://([^/]+)'), ''), 'none') as key, count(distinct session_id)::int
       from events where name = 'page_view' and created_at > ${since} group by 1 order by 2 desc limit 10`,
    ),
    query<{ key: string; count: number }>(`select coalesce(props->>'where', '—') as key, count(*)::int from events where name = 'cta_click' and created_at > ${since} group by 1 order by 2 desc`),
    query<{ src: string; sessions: number; requests: number }>(
      `select coalesce(src, 'direct / organic') as src, count(distinct session_id)::int as sessions,
         count(distinct session_id) filter (where name = 'walkthrough_submitted')::int as requests
       from events where created_at > ${since} group by 1 order by 2 desc limit 8`,
    ),
  ]);
  const t = totals[0];
  const conv = t.sessions ? ((t.requests / t.sessions) * 100).toFixed(1) : '0.0';

  return (
    <main className={ad.page}>
      <div className={ad.pageHead}>
        <div><h1>Analytics</h1><p>Landing-site visits, QR scans and walkthrough requests. Visitors are counted from a daily anonymous hash; no personal data is stored for page views.</p></div>
        <span className={ad.range}>
          {[7, 30, 90].map((d) => <Link key={d} href={`/analytics?days=${d}`} className={d === days ? 'active' : ''}>{d} days</Link>)}
        </span>
      </div>

      <div className={ad.tiles}>
        <div className={ad.tile}><span>Page views</span><strong>{t.views}</strong></div>
        <div className={ad.tile}><span>Visitors</span><strong>{t.visitors}</strong><small>{t.sessions} browser sessions</small></div>
        <div className={ad.tile}><span>QR scans</span><strong>{t.scans}</strong><small>brochure code</small></div>
        <div className={ad.tile}><span>Brochure opens</span><strong>{t.downloads}</strong></div>
        <div className={ad.tile}><span>CTA clicks</span><strong>{t.cta}</strong><small>{t.forms} started the form</small></div>
        <div className={ad.tile}><span>Walkthrough requests</span><strong>{t.requests}</strong><small>{conv}% of sessions</small></div>
      </div>

      <div className={ad.grid2}>
        <section className={ad.card}><h2>Page views per day</h2><Bars days={perDay} field="views" /></section>
        <section className={ad.card}><h2>Visitors per day</h2><Bars days={perDay} field="visitors" color="#0891b2" /></section>
        <section className={ad.card}><h2>QR scans per day</h2><Bars days={perDay} field="scans" color="#f26a1b" /></section>
        <section className={ad.card}><h2>Walkthrough requests per day</h2><Bars days={perDay} field="requests" color="#16a34a" /></section>
      </div>

      <div className={ad.grid2}>
        <section className={ad.card}><h2>Top pages</h2><Ranked rows={paths} label="path" /></section>
        <section className={ad.card}><h2>Sources (sessions)</h2><Ranked rows={sources} label="src" /><p style={{ margin: 0, fontSize: 12, color: 'var(--muted)' }}>Anything scanned from the brochure arrives as <code>brochure-qr</code>. Add <code>?src=…</code> to any link you hand out to track it here.</p></section>
        <section className={ad.card}><h2>Referrers (sessions)</h2><Ranked rows={referrers} label="ref" /></section>
        <section className={ad.card}><h2>Which button converts</h2><Ranked rows={ctas} label="cta" /></section>
        <section className={ad.card}>
          <h2>Source → request funnel</h2>
          <table className={ad.table}>
            <thead><tr><th>Source</th><th>Sessions</th><th>Requests</th><th>Rate</th></tr></thead>
            <tbody>
              {funnel.map((f) => (
                <tr key={f.src}><td>{f.src}</td><td>{f.sessions}</td><td>{f.requests}</td><td>{f.sessions ? ((f.requests / f.sessions) * 100).toFixed(1) : '0.0'}%</td></tr>
              ))}
              {!funnel.length && <tr><td colSpan={4} style={{ color: 'var(--muted)' }}>No traffic yet.</td></tr>}
            </tbody>
          </table>
        </section>
      </div>
    </main>
  );
}
