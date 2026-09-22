/** ToolFoam Pro mark: a foam tile with an orange wrench pocket and two socket pockets cut into it. */
export default function Mark({ size = 34, tile = '#2b3035', edge = '#3d444a', pocket = '#f26a1b', className }: { size?: number; tile?: string; edge?: string; pocket?: string; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true" className={className}>
      <rect x="1" y="1" width="38" height="38" rx="8" fill={tile} />
      <rect x="1.75" y="1.75" width="36.5" height="36.5" rx="7.4" fill="none" stroke={edge} strokeWidth="1.5" />
      <g transform="translate(18 21) rotate(-42)">
        <rect x="-2.6" y="-7" width="5.2" height="20" rx="2.6" fill={pocket} />
        <circle cx="0" cy="-9.5" r="6.6" fill={pocket} />
        <rect x="-1.9" y="-17.5" width="3.8" height="8.4" rx="0.5" fill={tile} />
        <circle cx="0" cy="11.2" r="3.6" fill={pocket} />
        <circle cx="0" cy="11.2" r="1.6" fill={tile} />
      </g>
      <circle cx="31" cy="9.5" r="3.2" fill={pocket} />
      <circle cx="31" cy="9.5" r="1.4" fill={tile} />
      <circle cx="31" cy="18" r="2.4" fill={pocket} />
      <circle cx="31" cy="18" r="1" fill={tile} />
    </svg>
  );
}
