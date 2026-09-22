import type { Metadata } from 'next';
import Link from 'next/link';
import styles from '../walkthrough/walkthrough.module.css';

export const metadata: Metadata = { title: 'Request received — ToolFoam Pro' };

export default function Thanks() {
  return (
    <main className={styles.page}>
      <div className={`wrap ${styles.thanks}`}>
        <p className="eyebrow">Request received</p>
        <h1 className="display">We&apos;ll be in touch.</h1>
        <p>Thanks — a walkthrough request just landed on our desk. Expect a reply within one business day to lock in a time that fits your shift schedule.</p>
        <Link href="/" className="btn btnDark">Back to the site</Link>
      </div>
    </main>
  );
}
