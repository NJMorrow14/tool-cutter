'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import ui from './ui.module.css';

export default function BrochureUpload() {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const router = useRouter();
  return (
    <form
      className={ui.row}
      onSubmit={async (e) => {
        e.preventDefault();
        const input = e.currentTarget.querySelector('input[type=file]') as HTMLInputElement;
        const file = input.files?.[0];
        if (!file) return;
        setBusy(true); setMsg(null);
        const fd = new FormData(); fd.append('file', file);
        const res = await fetch('/api/brochure', { method: 'POST', body: fd });
        const data = (await res.json().catch(() => ({}))) as { error?: string; size?: number };
        setBusy(false);
        if (!res.ok) { setMsg(data.error ?? 'Upload failed'); return; }
        setMsg(`Replaced (${((data.size ?? 0) / 1024 / 1024).toFixed(2)} MB). The public link serves it immediately.`);
        input.value = '';
        router.refresh();
      }}
    >
      <input type="file" accept="application/pdf" className={ui.input} style={{ flex: 1 }} />
      <button type="submit" className={`${ui.btn} ${ui.btnSm} ${ui.btnPrimary}`} disabled={busy}>{busy ? 'Uploading…' : 'Replace PDF'}</button>
      {msg && <span className={ui.hint}>{msg}</span>}
    </form>
  );
}
