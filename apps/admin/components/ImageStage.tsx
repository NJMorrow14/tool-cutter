'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './stage.module.css';

export interface StagePoint {
  x: number;
  y: number;
}

interface Props {
  src: string;
  width: number;
  height: number;
  overlaySrc?: string | null;
  overlayOpacity?: number;
  cursor?: string;
  onPointerDown?: (p: StagePoint, e: React.PointerEvent<SVGSVGElement>) => void;
  onPointerMove?: (p: StagePoint, e: React.PointerEvent<SVGSVGElement>) => void;
  onPointerUp?: (p: StagePoint, e: React.PointerEvent<SVGSVGElement>) => void;
  /** Children draw in image pixel coordinates. `upp` = image units per CSS pixel (for constant-size handles). */
  children?: (upp: number) => React.ReactNode;
}

/** Full-width image with an SVG overlay whose coordinate system equals the image's pixel grid. */
export default function ImageStage({
  src, width, height, overlaySrc, overlayOpacity = 0.6, cursor, onPointerDown, onPointerMove, onPointerUp, children,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [upp, setUpp] = useState(1);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      if (r.width > 0) setUpp(width / r.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [width]);

  const toImage = useCallback(
    (e: React.PointerEvent<SVGSVGElement>): StagePoint => {
      const el = svgRef.current!;
      const r = el.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * width, y: ((e.clientY - r.top) / r.height) * height };
    },
    [width, height],
  );

  return (
    <div className={styles.stage} style={{ aspectRatio: `${width} / ${height}` }}>
      <img src={src} alt="" width={width} height={height} draggable={false} />
      {overlaySrc && <img className={styles.overlayImg} src={overlaySrc} alt="" style={{ opacity: overlayOpacity }} draggable={false} />}
      <svg
        ref={svgRef}
        className={styles.svg}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ cursor: cursor ?? 'default' }}
        onPointerDown={(e) => onPointerDown?.(toImage(e), e)}
        onPointerMove={(e) => onPointerMove?.(toImage(e), e)}
        onPointerUp={(e) => onPointerUp?.(toImage(e), e)}
        onPointerCancel={(e) => onPointerUp?.(toImage(e), e)}
      >
        {children?.(upp)}
      </svg>
    </div>
  );
}
