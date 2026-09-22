import type { Metadata } from 'next';
import './print.css';
import { REGION, REGION_SHORT, CITIES } from '../../../lib/region';
import Mark from '../../../components/Mark';

export const metadata: Metadata = { title: 'ToolFoam Pro — brochure (print)' };
export const dynamic = 'force-dynamic';

// Two US-Letter pages, rendered to PDF by scripts/render-brochure.mjs (headless Chrome) or with the browser's
// print dialog (margins: none, background graphics: on). The QR code encodes NEXT_PUBLIC_WEB_URL/?src=brochure-qr.
export default function BrochurePrint() {
  const webUrl = (process.env.NEXT_PUBLIC_WEB_URL || 'http://localhost:3001').replace(/\/$/, '');
  const target = `${webUrl}/?src=brochure-qr`;
  const host = webUrl.replace(/^https?:\/\//, '');
  const qr = `/api/qr?url=${encodeURIComponent(target)}&fmt=svg&size=600`;

  const Logo = ({ size = 22 }: { size?: number }) => (
    <span className="logo" style={{ fontSize: size }}>
      <Mark size={size * 1.7} />
      <span>TOOLFOAM<em>PRO</em></span>
    </span>
  );

  return (
    <div className="sheets">
      {/* ------------------------------------------------------------------ page 1 */}
      <section className="page p1">
        <header className="bar">
          <Logo />
          <span className="tagline">On-site tool organization · {REGION_SHORT}</span>
        </header>
        <div className="hero">
          <img src="/brochure/hero.jpg" alt="" />
          <div className="heroShade" />
          <div className="heroCopy">
            <p className="eyebrow">Organized plants outperform · Serving {REGION}</p>
            <h1>A place for<br /><span>every tool.</span></h1>
            <p className="lede">Custom-cut foam drawer systems, designed and installed in your facility — on site, anywhere in {REGION}.</p>
          </div>
          <div className="badge"><strong>24-hour turnaround</strong><span>Fast service built around your production schedule</span></div>
        </div>
        <div className="body">
          <p className="eyebrow">What we do</p>
          <h2>We come to your plant, map your tools, and create a precise foam organization system for your drawers and workstations.</h2>
          <div className="cards">
            {[['01', 'Faster shifts', 'Reduce time spent searching for tools.'], ['02', 'Safer workspaces', 'Give every tool a clear, secure home.'], ['03', 'Visible accountability', 'Spot a missing tool at a glance.']].map(([n, t, d]) => (
              <div key={n} className="card"><span>{n}</span><h3>{t}</h3><p>{d}</p></div>
            ))}
          </div>
        </div>
        <footer className="cta">
          <div>
            <p className="eyebrow">Ready to transform your tool storage?</p>
            <h2>Book a plant walkthrough</h2>
            <p>We measure. We design. We install. Locally owned — {CITIES}.</p>
          </div>
          <div className="qr">
            <img src={qr} alt="QR code" />
            <div><strong>Scan to book</strong><span>{host}</span></div>
          </div>
        </footer>
      </section>

      {/* ------------------------------------------------------------------ page 2 */}
      <section className="page p2">
        <header className="bar bar2">
          <Logo size={18} />
          <span className="tagline">On-site industrial tool organization</span>
        </header>
        <div className="body2">
          <p className="eyebrow">A simple on-site process</p>
          <h2 className="display">From cluttered to controlled.</h2>
          <p className="ledeDark">A practical system your team can understand at a glance — and maintain shift after shift.</p>
          <ol className="steps">
            {[['We measure.', 'We map tools, drawers, and the way your team works.'], ['We design.', 'We create a custom foam layout for your workflow.'], ['We install.', 'We deliver an organized system ready for the floor.']].map(([t, d], i) => (
              <li key={t}><span className="num">{i + 1}</span><div><h3>{t}</h3><p>{d}</p></div></li>
            ))}
          </ol>
          <div className="pairs">
            {[['machining', 'Machining cell'], ['metrology', 'Quality lab'], ['aerospace', 'Aerospace MRO']].map(([k, t], i) => (
              <figure key={k} className={i === 0 ? 'pair big' : 'pair'}>
                <div><span className="tag">Before</span><img src={`/brochure/ba-${k}-before.jpg`} alt="" /></div>
                <div><span className="tag orange">After</span><img src={`/brochure/ba-${k}-after.jpg`} alt="" /></div>
                <figcaption>{t}</figcaption>
              </figure>
            ))}
          </div>
          <div className="industries">
            <p className="eyebrow">Built for demanding environments</p>
            <ul>{['Automotive', 'Aerospace', 'Machining', 'Fabrication', 'Maintenance', 'Assembly'].map((n) => <li key={n}>{n}</li>)}</ul>
            <p className="regionLine">Serving {REGION} — {CITIES}. Within a couple of hours of Charlotte? We come to you.</p>
          </div>
          <div className="band">
            <div>
              <h2>24-hour turnaround</h2>
              <p>Less waiting. Less disruption. A more organized plant — fast.</p>
            </div>
            <div className="qr qrBand">
              <img src={qr} alt="QR code" />
              <div><strong>Book your walkthrough</strong><span>{host}</span></div>
            </div>
          </div>
        </div>
        <footer className="foot"><span>ToolFoam Pro · On-site tool organization · {REGION_SHORT}</span><span>A place for every tool.</span></footer>
      </section>
    </div>
  );
}
