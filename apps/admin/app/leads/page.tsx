import ad from '../admin.module.css';
import ui from '../../components/ui.module.css';
import { hasDb, query } from '../../lib/db';
import LeadRow, { type Lead } from '../../components/LeadRow';

export const dynamic = 'force-dynamic';

export default async function LeadsPage() {
  if (!hasDb()) {
    return (
      <main className={ad.page}>
        <div className={ad.pageHead}><h1>Walkthrough requests</h1></div>
        <div className={ad.notice}>Set <code>DATABASE_URL</code> and run <code>npm run migrate</code> to receive walkthrough requests from the landing site.</div>
      </main>
    );
  }
  const leads = await query<Lead>(
    `select id, created_at::text, status, name, company, email, phone, plant_location, scope, preferred_dates, message, src, notes
     from walkthrough_requests order by created_at desc limit 500`,
  );
  const open = leads.filter((l) => l.status === 'new').length;
  return (
    <main className={ad.page}>
      <div className={ad.pageHead}>
        <div><h1>Walkthrough requests</h1><p>{leads.length} total · {open} new. Set a status as you work them; notes save on blur.</p></div>
        <a className={`${ui.btn} ${ui.btnSm}`} href="/api/leads?format=csv">Export CSV</a>
      </div>
      <section className={ad.card} style={{ overflowX: 'auto' }}>
        <table className={ad.table}>
          <thead>
            <tr><th>When</th><th>Status</th><th>Who</th><th>Plant</th><th>Scope · dates</th><th>Message</th><th>Source</th><th>Notes</th></tr>
          </thead>
          <tbody>
            {leads.map((l) => <LeadRow key={l.id} lead={l} />)}
            {!leads.length && <tr><td colSpan={8} style={{ color: 'var(--muted)' }}>No requests yet. The form lives at the landing site&apos;s /walkthrough page.</td></tr>}
          </tbody>
        </table>
      </section>
    </main>
  );
}
