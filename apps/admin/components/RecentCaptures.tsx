'use client';

import { useCallback, useEffect, useState } from 'react';
import ui from './ui.module.css';
import { deleteSession, getSession, listSessions, type SessionSummary } from '../lib/api';
import type { SessionInfo } from '../lib/types';

/** Drawers scanned with the iPhone app (and other recent uploads) — open one here without knowing its id.
 *  Polls while visible, so a scan shows up a few seconds after the phone finishes uploading. */
export default function RecentCaptures({ onOpen }: { onOpen: (info: SessionInfo) => void }) {
  const [items, setItems] = useState<SessionSummary[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => { listSessions().then(setItems).catch(() => setItems([])); }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  if (!items || !items.length) return null;
  const open = async (id: string) => {
    setBusy(id); setError(null);
    try {
      const info = await getSession(id);           // rebuilds a saved capture if the server restarted (~10–30 s)
      window.history.replaceState(null, '', `?session=${id}`);
      onOpen(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not open that capture');
    } finally {
      setBusy(null);
    }
  };
  const remove = async (id: string) => {
    setBusy(id); setError(null);
    try {
      await deleteSession(id);
      setConfirming(null);
      setItems((prev) => (prev ? prev.filter((s) => s.id !== id) : prev));
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete that scan');
    } finally {
      setBusy(null);
    }
  };
  const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '');
  return (
    <div className={ui.panel}>
      <div className={ui.rowBetween}>
        <h2 className={ui.panelTitle}>Recent captures</h2>
        <span className={ui.hint}>scans from the iPhone app appear here</span>
      </div>
      <div className={ui.list}>
        {items.map((s) => (
          <div key={s.id} className={ui.toolRow} style={{ gridTemplateColumns: '1fr auto auto auto', cursor: confirming === s.id ? 'default' : 'pointer' }}
               onClick={() => { if (confirming !== s.id) void open(s.id); }}>
            <span>
              <strong>{s.source_kind === 'capture' ? 'Drawer scan' : s.filename ?? s.source_kind}</strong>
              {s.mat_mm ? ` · ${Math.round(s.mat_mm.width)} × ${Math.round(s.mat_mm.height)} mm` : ''}
              {s.frames ? ` · ${s.frames} frames` : ''}
            </span>
            {confirming === s.id ? (
              <>
                <span className={ui.toolMeta} style={{ color: 'var(--danger)' }}>
                  Delete for good{s.saved ? ` — the ${s.frames ?? ''} raw frames go too` : ''}?
                </span>
                <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} disabled={busy !== null}
                        onClick={(e) => { e.stopPropagation(); void remove(s.id); }}>
                  {busy === s.id ? <span className={ui.spinner} /> : 'Delete'}
                </button>
                <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} disabled={busy !== null}
                        onClick={(e) => { e.stopPropagation(); setConfirming(null); }}>Cancel</button>
              </>
            ) : (
              <>
                <span className={ui.toolMeta}>{when(s.created)}{!s.in_memory ? ' · saved' : ''}</span>
                <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={busy !== null}>{busy === s.id ? <span className={ui.spinner} /> : 'Open'}</button>
                <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} title="Delete this scan" aria-label="Delete this scan"
                        disabled={busy !== null} onClick={(e) => { e.stopPropagation(); setError(null); setConfirming(s.id); }}>✕</button>
              </>
            )}
          </div>
        ))}
      </div>
      {busy && <p className={ui.hint}>Opening… a saved scan is re-processed from its raw frames, which takes a few seconds per frame.</p>}
      {error && <div className={ui.error}>{error}</div>}
    </div>
  );
}
