import ad from '../admin.module.css';
import ui from '../../components/ui.module.css';
import { hasDb, queryOne } from '../../lib/db';
import BrochureUpload from '../../components/BrochureUpload';

export const dynamic = 'force-dynamic';

export default async function BrochurePage() {
  const webUrl = process.env.NEXT_PUBLIC_WEB_URL || 'http://localhost:3001';
  const target = `${webUrl}/?src=brochure-qr`;
  const meta = hasDb()
    ? await queryOne<{ filename: string; size: number; updated_at: string }>(`select filename, size, updated_at::text from assets where key = 'brochure'`)
    : null;
  return (
    <main className={ad.page}>
      <div className={ad.pageHead}>
        <div><h1>Brochure &amp; QR code</h1><p>The PDF handed out in plants, and the code printed on it. Scans land on the landing site tagged <code>brochure-qr</code>, so they show up separately in Analytics and on each walkthrough request.</p></div>
      </div>

      <div className={ad.grid2}>
        <section className={ad.card}>
          <h2>QR code for the brochure</h2>
          <div className={ad.qrBox}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/api/qr?url=${encodeURIComponent(target)}&size=440`} alt="QR code linking to the landing site" />
            <div style={{ display: 'grid', gap: 10, fontSize: 13 }}>
              <div><span className={ui.label}>Points to</span><br /><code>{target}</code></div>
              <p style={{ margin: 0, color: 'var(--muted)' }}>Print at 20 mm or larger. Set <code>NEXT_PUBLIC_WEB_URL</code> to the public domain before printing — the code encodes the URL literally.</p>
              <div className={ui.row}>
                <a className={`${ui.btn} ${ui.btnSm} ${ui.btnPrimary}`} href={`/api/qr?url=${encodeURIComponent(target)}&fmt=svg&download=1`}>Download SVG</a>
                <a className={`${ui.btn} ${ui.btnSm}`} href={`/api/qr?url=${encodeURIComponent(target)}&fmt=png&size=2048&download=1`}>Download PNG (2048 px)</a>
              </div>
            </div>
          </div>
        </section>

        <section className={ad.card}>
          <h2>Current PDF</h2>
          {!hasDb() && <div className={ad.notice}>Set <code>DATABASE_URL</code> to store an uploaded brochure. The landing site falls back to <code>brand/toolfoam-brochure.pdf</code>.</div>}
          {meta ? (
            <p style={{ margin: 0, fontSize: 13 }}><strong>{meta.filename}</strong> · {(meta.size / 1024 / 1024).toFixed(2)} MB · updated {meta.updated_at.slice(0, 16).replace('T', ' ')}</p>
          ) : hasDb() ? <p style={{ margin: 0, color: 'var(--muted)' }}>No brochure uploaded yet.</p> : null}
          <div className={ui.row}>
            <a className={`${ui.btn} ${ui.btnSm}`} href="/api/brochure" target="_blank" rel="noopener">Open PDF</a>
            <a className={`${ui.btn} ${ui.btnSm}`} href={`${webUrl}/brochure`} target="_blank" rel="noopener">Public link ↗</a>
            <a className={`${ui.btn} ${ui.btnSm}`} href="/brochure/print" target="_blank" rel="noopener">Print template ↗</a>
          </div>
          <BrochureUpload />
          <p className={ui.hint} style={{ margin: 0 }}>The brochure is generated from the print template (photos + this QR code). To refresh it after changing the domain or copy:
            <code> npm run brochure</code> — renders it with Chrome, saves <code>brand/toolfoam-brochure.pdf</code> and replaces the PDF here. Or open the template and print to PDF (no margins, background graphics on), then upload it.</p>
        </section>
      </div>

      {meta && (
        <section className={ad.card}>
          <h2>Preview</h2>
          <iframe className={ad.pdfFrame} src="/api/brochure#view=FitH" title="Brochure preview" />
        </section>
      )}
    </main>
  );
}
