'use client';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import ui from '../../components/ui.module.css';
import ad from '../admin.module.css';

function LoginForm() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const router = useRouter();
  const params = useSearchParams();
  return (
    <form
      className={`${ui.panel} ${ad.loginCard}`}
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true); setError('');
        const res = await fetch('/api/login', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ password }) });
        setBusy(false);
        if (res.ok) router.push(params.get('next') || '/');
        else setError(((await res.json()) as { error?: string }).error ?? 'Nope');
      }}
    >
      <h1 className={ui.panelTitle}>ToolFoam Pro · HQ</h1>
      <label className={ui.field}>
        <span className={ui.label}>Password</span>
        <input className={ui.input} type="password" autoFocus value={password} onChange={(e) => setPassword(e.target.value)} />
      </label>
      {error && <p className={ui.error} role="alert">{error}</p>}
      <button className={`${ui.btn} ${ui.btnPrimary} ${ui.btnBlock}`} disabled={busy}>{busy ? '…' : 'Open up'}</button>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className={ad.loginWrap}>
      <Suspense fallback={null}><LoginForm /></Suspense>
    </div>
  );
}
