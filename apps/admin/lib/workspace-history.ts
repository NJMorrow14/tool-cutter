import type { LayoutSettings, Tool } from './types';

export interface WorkspaceDocument {
  tools: Tool[];
  settings: LayoutSettings;
  matSize: { width_mm: number; height_mm: number };
}
export type ToolChange = ((value: Tool[] | ((previous: Tool[]) => Tool[])) => void) & { isCurrent: () => boolean };
type Update<T> = T | ((previous: T) => T);
type Entry = { document: WorkspaceDocument; group: object };

// Pending network work is not part of a document. Restoring it would leave a tool
// waiting forever for a request that undo has invalidated.
const snapshot = (document: WorkspaceDocument): WorkspaceDocument => ({
  ...document, tools: document.tools.map(t => t.pending ? { ...t, pending: false } : t),
});
const equal = (a: WorkspaceDocument, b: WorkspaceDocument) => JSON.stringify(a, (key, value) =>
  key === 'pending' || key === 'error' ? undefined : value) === JSON.stringify(b, (key, value) =>
  key === 'pending' || key === 'error' ? undefined : value);

/** Shared, chronological document history. All updates run once, outside React updaters. */
export function createWorkspaceHistory(initial: WorkspaceDocument, limit = 100) {
  let document = initial;
  let past: Entry[] = [];
  let future: Entry[] = [];
  let epoch = 0;
  let pointerGroup: object | null = null;
  let fieldGroup: object | null = null;
  let keyGroup: object | null = null;
  let turnGroup: object | null = null;
  const listeners = new Set<() => void>();
  let state = { document, canUndo: false, canRedo: false, epoch };
  const emit = () => {
    state = { document, canUndo: past.length > 0, canRedo: future.length > 0, epoch };
    listeners.forEach(listener => listener());
  };
  const endInteraction = () => { pointerGroup = fieldGroup = keyGroup = turnGroup = null; };
  const group = () => {
    if (pointerGroup || fieldGroup || keyGroup) return (pointerGroup || fieldGroup || keyGroup)!;
    if (!turnGroup) {
      const token = turnGroup = {};
      queueMicrotask(() => { if (turnGroup === token) turnGroup = null; });
    }
    return turnGroup;
  };
  const update = <K extends keyof WorkspaceDocument>(key: K, value: Update<WorkspaceDocument[K]>, token = group()) => {
    const nextValue = typeof value === 'function' ? (value as (v: WorkspaceDocument[K]) => WorkspaceDocument[K])(document[key]) : value;
    if (nextValue === document[key]) return;
    const next = { ...document, [key]: nextValue };
    if (!equal(document, next)) {
      if (past[past.length - 1]?.group !== token) {
        past = [...past.slice(Math.max(0, past.length - limit + 1)), { document: snapshot(document), group: token }];
      }
      future = [];
    }
    document = next;
    emit();
  };
  return {
    getSnapshot: () => state,
    subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
    update,
    // Capture before starting async work. An undone request must never replay its result.
    captureTools: () => {
      const started = epoch, token = group();
      return Object.assign((value: Update<Tool[]>) => { if (epoch === started) update('tools', value, token); }, { isCurrent: () => epoch === started });
    },
    writer: <K extends keyof WorkspaceDocument>(key: K) => {
      const started = epoch;
      return (value: Update<WorkspaceDocument[K]>) => { if (epoch === started) update(key, value); };
    },
    beginPointer: () => { pointerGroup = {}; },
    endPointer: () => { pointerGroup = null; },
    beginField: () => { fieldGroup = {}; },
    endField: () => { fieldGroup = null; },
    beginKey: () => { if (!keyGroup) keyGroup = {}; },
    endKey: () => { keyGroup = null; },
    endInteraction,
    undo: () => {
      if (!past.length) return;
      const entry = past[past.length - 1];
      future = [...future, { document: snapshot(document), group: entry.group }];
      document = entry.document;
      past = past.slice(0, -1);
      epoch++; endInteraction(); emit();
    },
    redo: () => {
      if (!future.length) return;
      const entry = future[future.length - 1];
      past = [...past, { document: snapshot(document), group: entry.group }];
      document = entry.document;
      future = future.slice(0, -1);
      epoch++; endInteraction(); emit();
    },
    // A newly calibrated/imported scan establishes a new coordinate system.
    reset: (next: WorkspaceDocument) => {
      document = snapshot(next); past = []; future = []; epoch++;
      endInteraction(); emit();
    },
  };
}
