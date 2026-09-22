'use client';

import { useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import styles from '../app/walkthrough/walkthrough.module.css';
import { sendEvent, sessionId, visitSource } from './Track';

export default function WalkthroughForm() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  const onFocus = () => {
    if (!started.current) { started.current = true; sendEvent('form_started'); }
  };

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const fd = new FormData(e.currentTarget);
    const body: Record<string, string> = {};
    fd.forEach((v, k) => { body[k] = String(v); });
    body.src = visitSource() ?? '';
    body.session_id = sessionId();
    try {
      const res = await fetch('/api/walkthrough', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
      const data = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) { setError(data.error ?? 'Something went wrong. Please try again.'); setBusy(false); return; }
      router.push('/thanks');
    } catch {
      setError('Could not reach the server. Please try again in a moment.');
      setBusy(false);
    }
  };

  return (
    <form className={styles.form} onSubmit={onSubmit} onFocus={onFocus}>
      <h2>Schedule your walkthrough</h2>
      <div className={styles.row2}>
        <label className={styles.field}><span>Your name *</span><input name="name" required autoComplete="name" /></label>
        <label className={styles.field}><span>Company *</span><input name="company" required autoComplete="organization" /></label>
      </div>
      <div className={styles.row2}>
        <label className={styles.field}><span>Work email *</span><input name="email" type="email" required autoComplete="email" /></label>
        <label className={styles.field}><span>Phone</span><input name="phone" type="tel" autoComplete="tel" /></label>
      </div>
      <label className={styles.field}><span>Plant location</span><input name="plant_location" placeholder="e.g. Concord, NC — or the full address" /></label>
      <div className={styles.row2}>
        <label className={styles.field}>
          <span>Scope</span>
          <select name="scope" defaultValue="">
            <option value="">How many drawers?</option>
            <option>1–5 drawers</option>
            <option>6–20 drawers</option>
            <option>21–50 drawers</option>
            <option>50+ drawers / multiple cells</option>
            <option>Whole plant — let&apos;s talk</option>
          </select>
        </label>
        <label className={styles.field}><span>Preferred dates</span><input name="preferred_dates" placeholder="e.g. any Tuesday morning" /></label>
      </div>
      <label className={styles.field}><span>Anything we should know?</span><textarea name="message" placeholder="Shift schedule, tool types, existing toolboxes, safety requirements…" /></label>
      <input className={styles.hp} name="website" tabIndex={-1} autoComplete="off" aria-hidden="true" />
      {error && <p className={styles.error} role="alert">{error}</p>}
      <button type="submit" className="btn btnOrange" disabled={busy}>{busy ? 'Sending…' : 'Request my walkthrough'} <span className="arrow">→</span></button>
      <p className={styles.fine}>We reply within one business day. No spam, no sharing your details.</p>
    </form>
  );
}
