'use client';

import { useCallback, useEffect, useState } from 'react';
import ui from './ui.module.css';
import { API_BASE_URL, createLibraryJob, deleteLibraryTool, listLibraryJobs, listLibraryTools } from '../lib/api';
import { libraryTool } from '../lib/geom';
import type { LibraryJob, LibraryTool, Tool } from '../lib/types';

const JOB_KEY = 'tf.libraryJob';
const NEW_JOB = '__new__';

function readStoredJob(): string | null {
  try { return window.localStorage.getItem(JOB_KEY); } catch { return null; }
}
function storeJob(id: string) {
  try { window.localStorage.setItem(JOB_KEY, id); } catch { /* private window: just not remembered */ }
}

/** Tools photographed one by one with the iPhone app, grouped by job (docs/tool-library.md). Polls while mounted,
 *  so a tool appears a few seconds after the phone saves it. Adding puts a flat-pocket copy on the board. */
export default function LibraryPanel({ tools, onAdd, initialJobId }: { tools: Tool[]; onAdd: (tool: Tool) => void; initialJobId?: string | null }) {
  const [jobs, setJobs] = useState<LibraryJob[] | null>(null);
  const [jobId, setJobId] = useState<string | null>(initialJobId ?? null);
  const [items, setItems] = useState<LibraryTool[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ?job=<id> wins; otherwise the last job picked on this browser
  useEffect(() => {
    if (initialJobId) { setJobId(initialJobId); storeJob(initialJobId); }
    else setJobId((cur) => cur ?? readStoredJob());
  }, [initialJobId]);

  const refreshJobs = useCallback(() => { listLibraryJobs().then(setJobs).catch(() => setJobs([])); }, []);
  const refreshTools = useCallback(() => {
    if (!jobId) { setItems(null); return; }
    listLibraryTools(jobId).then((t) => { setItems(t); setError(null); })
      .catch((err) => { setItems([]); setError(err instanceof Error ? err.message : 'Could not list the tools'); });
  }, [jobId]);

  useEffect(() => {
    refreshJobs(); refreshTools();
    const t = setInterval(() => { refreshJobs(); refreshTools(); }, 5000);
    return () => clearInterval(t);
  }, [refreshJobs, refreshTools]);

  // no remembered job (or it was deleted): fall back to the newest one
  useEffect(() => {
    if (!jobs || !jobs.length) return;
    if (!jobId || (!jobs.some((j) => j.id === jobId) && jobId !== initialJobId)) setJobId(jobs[0].id);
  }, [jobs, jobId, initialJobId]);

  const pick = async (value: string) => {
    setError(null); setConfirming(null);
    if (value !== NEW_JOB) { setJobId(value); storeJob(value); setItems(null); return; }
    const name = window.prompt('Name for the new job (customer, van, workshop…)')?.trim();
    if (!name) return;
    setBusy(NEW_JOB);
    try {
      const job = await createLibraryJob(name);
      setJobs((prev) => [job, ...(prev ?? []).filter((j) => j.id !== job.id)]);
      setJobId(job.id); storeJob(job.id); setItems([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the job');
    } finally {
      setBusy(null);
    }
  };

  const onBoard = new Set(tools.map((t) => t.library_tool_id).filter((id): id is string => !!id));
  const add = (rec: LibraryTool, existing: Tool[]) => { const t = libraryTool(rec, existing); onAdd(t); return t; };
  const addAll = () => {
    const placed: Tool[] = [...tools];
    for (const rec of items ?? []) if (!onBoard.has(rec.id)) placed.push(add(rec, placed));
  };
  const remove = async (id: string) => {
    setBusy(id); setError(null);
    try {
      await deleteLibraryTool(id);
      setConfirming(null);
      setItems((prev) => (prev ? prev.filter((t) => t.id !== id) : prev));
      refreshTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete that tool');
    } finally {
      setBusy(null);
    }
  };

  const size = (t: LibraryTool) => {
    let [w, h] = t.bbox_mm ?? [0, 0];
    if (!w || !h) {
      const xs = t.polygon_mm.map((p) => p[0]), ys = t.polygon_mm.map((p) => p[1]);
      w = Math.max(...xs) - Math.min(...xs); h = Math.max(...ys) - Math.min(...ys);
    }
    return `${Math.round(w)} × ${Math.round(h)} mm`;
  };
  const missing = (items ?? []).filter((t) => !onBoard.has(t.id)).length;
  const knownJob = !!jobId && (jobs ?? []).some((j) => j.id === jobId);

  return (
    <div className={ui.section}>
      <div className={ui.rowBetween}>
        <h2 className={ui.panelTitle}>Tool library</h2>
        {!!items?.length && (
          <button type="button" className={`${ui.btn} ${ui.btnSm}`} disabled={!missing} onClick={addAll}
                  title="Put every tool of this job that is not on the board yet onto it">Add all{missing ? ` (${missing})` : ''}</button>
        )}
      </div>
      <select className={ui.select} aria-label="Library job" value={jobId ?? ''} disabled={busy === NEW_JOB}
              onChange={(e) => void pick(e.target.value)}>
        {!jobId && <option value="" disabled>Choose a job…</option>}
        {jobId && !knownJob && <option value={jobId}>{jobs === null ? 'Loading…' : `Job ${jobId}`}</option>}
        {(jobs ?? []).map((j) => <option key={j.id} value={j.id}>{j.name}{typeof j.tool_count === 'number' ? ` (${j.tool_count})` : ''}</option>)}
        <option value={NEW_JOB}>New job…</option>
      </select>
      {jobId && items === null && <p className={ui.hint}><span className={ui.spinner} /> Loading tools…</p>}
      {jobId && items !== null && !items.length && !error && <p className={ui.hint}>No tools in this job yet — photograph them with the iPhone app; they appear here by themselves.</p>}
      {!!items?.length && (
        <div className={ui.list}>
          {items.map((t) => (
            <div key={t.id} className={ui.toolRow} style={{ gridTemplateColumns: '48px minmax(0,1fr) auto auto', alignItems: 'center' }}>
              {/* library thumbnails come from the local API; next/image adds nothing here */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`${API_BASE_URL}${t.thumb_url}`} alt={t.name} width={48} height={48}
                   style={{ width: 48, height: 48, objectFit: 'cover', borderRadius: 4, background: 'var(--panel-2)' }} />
              <span style={{ minWidth: 0 }}>
                <strong style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.name}</strong>
                <span className={ui.toolMeta}>{size(t)} · {t.thickness_mm ? `${Math.round(t.thickness_mm * 10) / 10} mm thick` : 'no thickness'}</span>
              </span>
              {confirming === t.id ? (
                <>
                  <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnDanger}`} disabled={busy !== null}
                          title="Delete this tool and its photo for good" onClick={() => void remove(t.id)}>
                    {busy === t.id ? <span className={ui.spinner} /> : 'Delete'}
                  </button>
                  <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} disabled={busy !== null}
                          onClick={() => setConfirming(null)}>Cancel</button>
                </>
              ) : (
                <>
                  {onBoard.has(t.id)
                    ? <span className={ui.toolMeta}>on board</span>
                    : <button type="button" className={`${ui.btn} ${ui.btnSm}`} onClick={() => add(t, tools)}>Add</button>}
                  <button type="button" className={`${ui.btn} ${ui.btnSm} ${ui.btnGhost}`} title="Delete this tool from the library" aria-label={`Delete ${t.name} from the library`}
                          disabled={busy !== null} onClick={() => { setError(null); setConfirming(t.id); }}>✕</button>
                </>
              )}
            </div>
          ))}
        </div>
      )}
      {error && <div className={ui.error}>{error}</div>}
    </div>
  );
}
