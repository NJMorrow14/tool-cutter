'use client';

import { useState } from 'react';
import ad from '../app/admin.module.css';

export interface Lead {
  id: number; created_at: string; status: string; name: string; company: string; email: string; phone: string | null;
  plant_location: string | null; scope: string | null; preferred_dates: string | null; message: string | null; src: string | null; notes: string | null;
}

const STATUSES = ['new', 'contacted', 'scheduled', 'done', 'lost'];

export default function LeadRow({ lead }: { lead: Lead }) {
  const [status, setStatus] = useState(lead.status);
  const [notes, setNotes] = useState(lead.notes ?? '');
  const [saving, setSaving] = useState(false);
  const save = async (patch: { status?: string; notes?: string }) => {
    setSaving(true);
    await fetch('/api/leads', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: lead.id, ...patch }) }).catch(() => null);
    setSaving(false);
  };
  return (
    <tr style={{ opacity: saving ? 0.6 : 1 }}>
      <td style={{ whiteSpace: 'nowrap' }}>{lead.created_at.slice(0, 16).replace('T', ' ')}</td>
      <td>
        <select className={ad.status} value={status} onChange={(e) => { setStatus(e.target.value); void save({ status: e.target.value }); }}>
          {STATUSES.map((s) => <option key={s}>{s}</option>)}
        </select>
      </td>
      <td>
        <strong>{lead.name}</strong> · {lead.company}<br />
        <a href={`mailto:${lead.email}`}>{lead.email}</a>{lead.phone ? <> · <a href={`tel:${lead.phone}`}>{lead.phone}</a></> : null}
      </td>
      <td>{lead.plant_location ?? '—'}</td>
      <td>{lead.scope ?? '—'}{lead.preferred_dates ? <><br /><span style={{ color: 'var(--muted)' }}>{lead.preferred_dates}</span></> : null}</td>
      <td style={{ maxWidth: 320, whiteSpace: 'pre-wrap' }}>{lead.message ?? '—'}</td>
      <td><span className={lead.src === 'brochure-qr' ? ad.badgeNew : ''}>{lead.src ?? 'direct'}</span></td>
      <td><textarea value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => void save({ notes })} rows={2} style={{ width: 200, border: '1px solid var(--border-strong)', borderRadius: 6, padding: 6, fontSize: 12 }} /></td>
    </tr>
  );
}
