'use client';

import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import { createWorkspaceHistory, type WorkspaceDocument } from './workspace-history';

export function useWorkspaceHistory(initial: WorkspaceDocument, enabled: boolean) {
  const [history] = useState(() => createWorkspaceHistory(initial));
  const state = useSyncExternalStore(history.subscribe, history.getSnapshot, history.getSnapshot);
  const writers = useMemo(() => ({
    setTools: history.writer('tools'), setSettings: history.writer('settings'), setMatSize: history.writer('matSize'),
  // Refresh guarded writers after undo/redo/reset; old async closures remain invalid.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [history, state.epoch]);

  useEffect(() => {
    if (!enabled) return;
    let release: ReturnType<typeof setTimeout> | undefined;
    const pointerDown = (e: PointerEvent) => {
      clearTimeout(release);
      // Canvas controls can prevent native blur; don't join their drag to a prior field edit.
      if (!(e.target as HTMLElement).closest('[data-history-fields] input, [data-history-fields] textarea, [data-history-fields] select')) history.endField();
      history.beginPointer();
    };
    // Run after pointerup AND click handlers, including native Three.js handlers.
    const pointerUp = () => { release = setTimeout(history.endPointer, 0); };
    const focusIn = (e: FocusEvent) => {
      if ((e.target as HTMLElement).matches('input:not([type=checkbox]), textarea, select') && (e.target as HTMLElement).closest('[data-history-fields]')) history.beginField();
    };
    const focusOut = () => history.endField();
    const keyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const field = target.closest('input, textarea, select, [contenteditable]');
      const documentField = field?.closest('[data-history-fields]');
      if (field && !documentField) return; // Keep native undo in search and uncommitted forms.
      const key = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && !e.altKey && (key === 'z' || key === 'y')) {
        e.preventDefault(); e.stopImmediatePropagation();
        if (key === 'y' || e.shiftKey) history.redo(); else history.undo();
        return;
      }
      if (!e.ctrlKey && !e.metaKey && ['arrowleft', 'arrowright', 'arrowup', 'arrowdown', 'r', '[', ']'].includes(key)) history.beginKey();
    };
    const keyUp = () => history.endKey();
    const blur = () => history.endInteraction();
    window.addEventListener('pointerdown', pointerDown, true);
    window.addEventListener('pointerup', pointerUp);
    window.addEventListener('pointercancel', pointerUp);
    window.addEventListener('focusin', focusIn);
    window.addEventListener('focusout', focusOut);
    window.addEventListener('keydown', keyDown, true);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', blur);
    return () => {
      clearTimeout(release); history.endInteraction();
      window.removeEventListener('pointerdown', pointerDown, true);
      window.removeEventListener('pointerup', pointerUp);
      window.removeEventListener('pointercancel', pointerUp);
      window.removeEventListener('focusin', focusIn);
      window.removeEventListener('focusout', focusOut);
      window.removeEventListener('keydown', keyDown, true);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', blur);
    };
  }, [history, enabled]);
  return { ...state, ...writers, history };
}
