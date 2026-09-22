import type { Metadata } from 'next';
import styles from './walkthrough.module.css';
import WalkthroughForm from '../../components/WalkthroughForm';
import { REGION, CITIES } from '../../lib/region';

export const metadata: Metadata = { title: 'Book a plant walkthrough — ToolFoam Pro' };

export default function WalkthroughPage() {
  return (
    <main className={styles.page}>
      <div className={`wrap ${styles.grid}`}>
        <div className={styles.copy}>
          <p className="eyebrow">Book a plant walkthrough</p>
          <h1 className={`display ${styles.h1}`}>Tell us about<br />your floor.</h1>
          <p className={styles.lede}>We come out to plants across {REGION} — {CITIES} — map your drawers and workstations, and come back with a foam system cut to your tools. No obligation, no disruption to the shift.</p>
          <ol className={styles.steps}>
            <li><strong>Walkthrough</strong> — 30 to 60 minutes on your floor. We photograph and measure every drawer you want done.</li>
            <li><strong>Design</strong> — a to-scale layout for each drawer, sent for your sign-off.</li>
            <li><strong>Install</strong> — inserts cut and dropped in. 24-hour turnaround from approval.</li>
          </ol>
          <p className={styles.small}>Questions first? Email <a href="mailto:hello@toolfoampro.com">hello@toolfoampro.com</a>.</p>
        </div>
        <WalkthroughForm />
      </div>
    </main>
  );
}
