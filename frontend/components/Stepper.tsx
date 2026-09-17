'use client';

import styles from '../app/page.module.css';

export type StepId = 'upload' | 'calibrate' | 'detect' | 'layout';

export interface StepDef {
  id: StepId;
  title: string;
  hint: string;
  enabled: boolean;
  done: boolean;
}

export default function Stepper({ steps, current, onSelect }: { steps: StepDef[]; current: StepId; onSelect: (s: StepId) => void }) {
  return (
    <nav className={styles.stepper} aria-label="Steps">
      {steps.map((s, i) => (
        <button
          key={s.id}
          type="button"
          className={`${styles.step} ${s.id === current ? styles.stepActive : ''} ${s.done && s.id !== current ? styles.stepDone : ''}`}
          disabled={!s.enabled}
          onClick={() => onSelect(s.id)}
        >
          <span className={styles.stepNum}>{s.done && s.id !== current ? '✓' : i + 1}</span>
          <span className={styles.stepText}>
            <span className={styles.stepTitle}>{s.title}</span>
            <span className={styles.stepHint}>{s.hint}</span>
          </span>
        </button>
      ))}
    </nav>
  );
}
