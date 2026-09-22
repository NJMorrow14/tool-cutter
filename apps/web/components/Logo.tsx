import styles from '../app/layout.module.css';
import Mark from './Mark';

export default function Logo({ light = true, size = 34 }: { light?: boolean; size?: number }) {
  return (
    <span className={styles.logo} style={{ color: light ? '#fff' : 'var(--ink)', fontSize: size * 0.65 }}>
      <Mark size={size} />
      <span>TOOLFOAM<em>PRO</em></span>
    </span>
  );
}
