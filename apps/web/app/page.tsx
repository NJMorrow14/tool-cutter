import Link from 'next/link';
import styles from './page.module.css';
import CtaLink from '../components/CtaLink';
import { REGION, CITIES, REGION_LINE } from '../lib/region';

const INDUSTRIES = ['Automotive', 'Aerospace', 'Machining', 'Fabrication', 'Maintenance', 'Assembly'];

export default function Home() {
  return (
    <main>
      {/* ------------------------------------------------------------- hero */}
      <section className={styles.hero}>
        <div className={styles.heroClip}><img className={styles.heroImg} src="/img/hero.jpg" alt="Foam-organized tool drawer being installed in a plant" /></div>
        <div className={`wrap ${styles.heroInner}`}>
          <div className={styles.heroCopy}>
            <p className="eyebrow">Organized plants outperform · Serving {REGION}</p>
            <h1 className={`display ${styles.h1}`}>
              A place for<br /><span>every tool.</span>
            </h1>
            <p className={styles.lede}>Custom-cut foam drawer systems, designed and installed in your facility — on site, anywhere in {REGION}.</p>
            <div className={styles.heroActions}>
              <CtaLink href="/walkthrough" className="btn btnOrange" where="hero">Book a plant walkthrough <span className="arrow">→</span></CtaLink>
              <a href="/brochure" target="_blank" rel="noopener" className="btn btnGhost">View the brochure</a>
            </div>
          </div>
        </div>
        <div className={`wrap ${styles.badgeRow}`}>
          <div className={styles.badge}>
            <strong>24-hour turnaround</strong>
            <span>Fast service built around your production schedule</span>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------- what we do */}
      <section className={`wrap ${styles.section}`}>
        <p className="eyebrow">What we do</p>
        <h2 className={styles.h2}>We come to your plant, map your tools, and create a precise foam organization system for your drawers and workstations.</h2>
        <div className={styles.cards}>
          {[
            ['01', 'Faster shifts', 'Reduce time spent searching for tools.'],
            ['02', 'Safer workspaces', 'Give every tool a clear, secure home.'],
            ['03', 'Visible accountability', 'Spot a missing tool at a glance.'],
          ].map(([n, t, d]) => (
            <article key={n} className={styles.card}>
              <span className={styles.cardNum}>{n}</span>
              <h3>{t}</h3>
              <p>{d}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------- process */}
      <section id="process" className={styles.dark}>
        <div className={`wrap ${styles.section}`}>
          <p className="eyebrow">A simple on-site process</p>
          <h2 className={`display ${styles.h2Display}`}>From cluttered to controlled.</h2>
          <p className={styles.ledeDark}>A practical system your team can understand at a glance — and maintain shift after shift.</p>
          <ol className={styles.steps}>
            {[
              ['We measure.', 'We map tools, drawers, and the way your team works.'],
              ['We design.', 'We create a custom foam layout for your workflow.'],
              ['We install.', 'We deliver an organized system ready for the floor.'],
            ].map(([t, d], i) => (
              <li key={t}>
                <span className={styles.stepNum}>{i + 1}</span>
                <div><h3>{t}</h3><p>{d}</p></div>
              </li>
            ))}
          </ol>
          <div className={styles.gallery}>
            {[
              ['machining', 'Machining cell', 'Wrenches, sockets and drivers — 60+ pieces, one drawer, every one accounted for at shift change.'],
              ['metrology', 'Quality lab', 'Micrometers, indicators and gauge blocks each in a padded pocket. Nothing knocks, nothing drifts.'],
              ['aerospace', 'Aerospace MRO', 'Power tools, batteries and a torque wrench with a home — the missing-tool check takes one look.'],
            ].map(([key, title, blurb]) => (
              <figure key={key} className={styles.pair}>
                <div className={styles.pairImgs}>
                  <div><span className={styles.tag}>Before</span><img src={`/img/ba-${key}-before.jpg`} alt={`${title} drawer before: loose tools`} /></div>
                  <div><span className={`${styles.tag} ${styles.tagOrange}`}>After</span><img src={`/img/ba-${key}-after.jpg`} alt={`${title} drawer after: every tool in a foam pocket`} /></div>
                </div>
                <figcaption><strong>{title}</strong><span>{blurb}</span></figcaption>
              </figure>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------- industries */}
      <section id="industries" className={`wrap ${styles.section}`}>
        <div className={styles.industries}>
          <p className="eyebrow">Built for demanding environments</p>
          <ul>
            {INDUSTRIES.map((n) => <li key={n}>{n}</li>)}
          </ul>
          <p className={styles.regionLine}>{REGION_LINE} If your plant is within a couple of hours of Charlotte, we come to you.</p>
        </div>
      </section>

      {/* ------------------------------------------------------------- CTA */}
      <section className={styles.ctaBand}>
        <div className={`wrap ${styles.ctaInner}`}>
          <div>
            <p className="eyebrow">Ready to transform your tool storage?</p>
            <h2 className={`display ${styles.ctaTitle}`}>Book a plant walkthrough</h2>
            <p>We measure. We design. We install. Less waiting, less disruption — a more organized plant, fast. Serving {CITIES}.</p>
          </div>
          <CtaLink href="/walkthrough" className="btn btnOrange" where="band">Get started <span className="arrow">→</span></CtaLink>
        </div>
      </section>

      <p className={styles.brochureLine}>
        Prefer paper? <Link href="/brochure" target="_blank">Download the two-page brochure (PDF)</Link>.
      </p>
    </main>
  );
}
