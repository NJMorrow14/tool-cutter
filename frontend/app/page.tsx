'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './page.module.css';
import {
  API_BASE_URL,
  PreviewPayload,
  PreviewResponse,
  PreviewStats,
  QuarterManual,
  Rect,
  SamPoint,
  exportSvg,
  requestPreview,
  uploadImage,
} from '../lib/api';

const QUARTER_DIAMETER_MM = 24.26;

type InteractionMode = 'point' | 'crop' | 'quarter-roi' | 'quarter-manual';

type DragState =
  | {
      type: 'crop' | 'quarter-roi';
      start: { imgX: number; imgY: number };
      current: { imgX: number; imgY: number } | null;
    }
  | {
      type: 'quarter-manual';
      start: { imgX: number; imgY: number };
      current: { imgX: number; imgY: number } | null;
      shiftKey: boolean;
    };

interface UploadStatus {
  filename: string;
  converted?: string;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function formatNumber(num: number, fractionDigits = 0): string {
  return num.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

export default function HomePage() {
  const [imageShape, setImageShape] = useState<[number, number] | null>(null);
  const [baseImageShape, setBaseImageShape] = useState<[number, number] | null>(null);
  const [imageRotation, setImageRotation] = useState<number>(0);
  const [overlaySrc, setOverlaySrc] = useState<string | null>(null);
  const [maskSrc, setMaskSrc] = useState<string | null>(null);
  const [svgDataUri, setSvgDataUri] = useState<string | null>(null);
  const [stats, setStats] = useState<PreviewStats | null>(null);
  const [scaleDown, setScaleDown] = useState<number>(1);
  const [samPoints, setSamPoints] = useState<SamPoint[]>([]);
  const [cropRect, setCropRect] = useState<Rect | null>(null);
  const [quarterRoi, setQuarterRoi] = useState<Rect | null>(null);
  const [quarterManual, setQuarterManual] = useState<QuarterManual | null>(null);
  const [useQuarter, setUseQuarter] = useState<boolean>(false);
  const [quarterDiameter, setQuarterDiameter] = useState<number>(QUARTER_DIAMETER_MM);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('point');
  const [pointLabel, setPointLabel] = useState<SamPoint['label']>('pos');
  const [samAuto, setSamAuto] = useState<boolean>(false);
  const [autoPreview, setAutoPreview] = useState<boolean>(true);
  const [isComputing, setIsComputing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
  const [overlayDropActive, setOverlayDropActive] = useState<boolean>(false);
  const [autoRotationDeg, setAutoRotationDeg] = useState<number>(0);

  const overlayImgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const previewTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastPayloadRef = useRef<PreviewPayload | null>(null);
  const requestCounterRef = useRef<number>(0);
  const imageShapeRef = useRef<[number, number] | null>(null);
  const skipAutoPreviewRef = useRef<boolean>(false);

  const [overlayNatural, setOverlayNatural] = useState<{ width: number; height: number } | null>(null);

  const hasImageLoaded = Boolean(imageShape && overlaySrc);

  const buildPayload = useCallback(
    (overrides: Partial<PreviewPayload> = {}): PreviewPayload => {
      const has = (key: keyof PreviewPayload) => Object.prototype.hasOwnProperty.call(overrides, key);

      const effectiveCrop = has('crop_rect') ? overrides.crop_rect ?? null : cropRect;
      const effectivePoints = has('sam_points') ? overrides.sam_points ?? [] : samPoints;

      const cropOffsetX = effectiveCrop ? effectiveCrop.x0 : 0;
      const cropOffsetY = effectiveCrop ? effectiveCrop.y0 : 0;
      const cropWidth = effectiveCrop ? effectiveCrop.x1 - effectiveCrop.x0 : null;
      const cropHeight = effectiveCrop ? effectiveCrop.y1 - effectiveCrop.y0 : null;

      const payloadPoints = effectivePoints.map((pt) => ({
        x: Math.round(pt.x - cropOffsetX),
        y: Math.round(pt.y - cropOffsetY),
        label: pt.label,
      }));

      for (const pt of payloadPoints) {
        if (pt.x < 0) pt.x = 0;
        if (pt.y < 0) pt.y = 0;
        if (cropWidth !== null) pt.x = clamp(pt.x, 0, Math.max(0, cropWidth - 1));
        if (cropHeight !== null) pt.y = clamp(pt.y, 0, Math.max(0, cropHeight - 1));
      }

      return {
        use_quarter: has('use_quarter') ? Boolean(overrides.use_quarter) : useQuarter,
        quarter_diameter_mm: has('quarter_diameter_mm')
          ? overrides.quarter_diameter_mm ?? quarterDiameter
          : quarterDiameter,
        sam_points: payloadPoints,
        crop_rect: effectiveCrop ?? null,
        quarter_roi: has('quarter_roi') ? overrides.quarter_roi ?? null : quarterRoi,
        quarter_manual: has('quarter_manual') ? overrides.quarter_manual ?? null : quarterManual,
        sam_auto: has('sam_auto') ? Boolean(overrides.sam_auto) : samAuto,
        sam_multimask: has('sam_multimask') ? Boolean(overrides.sam_multimask) : true,
        sam_union: has('sam_union') ? Boolean(overrides.sam_union) : true,
        compute_mask: has('compute_mask') ? Boolean(overrides.compute_mask) : true,
        image_rotation: has('image_rotation') ? overrides.image_rotation ?? imageRotation : imageRotation,
      };
    },
    [useQuarter, quarterDiameter, samPoints, cropRect, quarterRoi, quarterManual, samAuto, imageRotation],
  );

  const runPreview = useCallback(
    async (payload: PreviewPayload, opts: { silent?: boolean } = {}) => {
      if (!imageShapeRef.current) {
        return;
      }
      lastPayloadRef.current = payload;
      const requestId = ++requestCounterRef.current;
      if (!opts.silent) {
        setIsComputing(true);
      }
      setError(null);
      try {
        const response: PreviewResponse = await requestPreview(payload);
        if (requestCounterRef.current !== requestId) {
          return;
        }
        setOverlaySrc(response.overlay_png);
        setMaskSrc(response.mask_png);
        setSvgDataUri(response.svg_data_uri ?? null);
        setStats(response.stats);
        setScaleDown(response.scale_down ?? 1);
        if (response.stats && typeof response.stats.image_rotation === 'number' && baseImageShape) {
          const normalized = ((Math.round(response.stats.image_rotation / 90) * 90) % 360 + 360) % 360;
          if (normalized !== imageRotation) {
            const nextShape: [number, number] = normalized % 180 === 0
              ? baseImageShape
              : [baseImageShape[1], baseImageShape[0]];
            setImageRotation(normalized);
            imageShapeRef.current = nextShape;
            setImageShape(nextShape);
          }
        }
      } catch (err) {
        if (requestCounterRef.current !== requestId) {
          return;
        }
        const message = err instanceof Error ? err.message : 'Preview failed';
        setError(message);
      } finally {
        if (requestCounterRef.current === requestId) {
          setIsComputing(false);
        }
      }
    },
    [baseImageShape, imageRotation],
  );

  const schedulePreview = useCallback(
    (opts: { immediate?: boolean; force?: boolean; overrides?: Partial<PreviewPayload> } = {}) => {
      if (!imageShape) {
        return;
      }
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
        previewTimerRef.current = null;
      }
      const payload = buildPayload(opts.overrides ?? {});
      lastPayloadRef.current = payload;
      const allow = autoPreview || opts.force;
      if (!allow) {
        return;
      }
      const invoke = () => {
        runPreview(payload, { silent: false });
      };
      if (opts.immediate) {
        invoke();
      } else {
        previewTimerRef.current = setTimeout(invoke, 220);
      }
    },
    [autoPreview, buildPayload, imageShape, runPreview],
  );

  useEffect(() => {
    imageShapeRef.current = imageShape;
  }, [imageShape]);

  useEffect(() => {
    if (autoPreview && imageShape) {
      if (skipAutoPreviewRef.current) {
        skipAutoPreviewRef.current = false;
        return;
      }
      schedulePreview({ immediate: true });
    }
  }, [autoPreview, imageShape, schedulePreview]);

  const projectImageToDisplay = useCallback(
    (imgX: number, imgY: number): { x: number; y: number } | null => {
      const imgEl = overlayImgRef.current;
      if (!imgEl || !overlayNatural) {
        return null;
      }
      const rect = imgEl.getBoundingClientRect();
      const displayWidth = rect.width;
      const displayHeight = rect.height;
      if (!displayWidth || !displayHeight) {
        return null;
      }
      const overlayX = imgX * (scaleDown || 1);
      const overlayY = imgY * (scaleDown || 1);
      if (overlayX < 0 || overlayY < 0) {
        return null;
      }
      const scaleX = displayWidth / overlayNatural.width;
      const scaleY = displayHeight / overlayNatural.height;
      return {
        x: overlayX * scaleX,
        y: overlayY * scaleY,
      };
    },
    [overlayNatural, scaleDown],
  );

  const mapClientToImage = useCallback(
    (clientX: number, clientY: number): { imgX: number; imgY: number } | null => {
      const imgEl = overlayImgRef.current;
      if (!imgEl || !overlayNatural) {
        return null;
      }
      const rect = imgEl.getBoundingClientRect();
      const withinX = clientX >= rect.left && clientX <= rect.right;
      const withinY = clientY >= rect.top && clientY <= rect.bottom;
      if (!withinX || !withinY) {
        return null;
      }
      const displayX = clientX - rect.left;
      const displayY = clientY - rect.top;
      const scaleX = overlayNatural.width / rect.width;
      const scaleY = overlayNatural.height / rect.height;
      const overlayX = displayX * scaleX;
      const overlayY = displayY * scaleY;
      const imgX = overlayX / (scaleDown || 1);
      const imgY = overlayY / (scaleDown || 1);
      return { imgX, imgY };
    },
    [overlayNatural, scaleDown],
  );

  const drawOverlay = useCallback(
    (activeDrag?: DragState | null) => {
      const canvas = canvasRef.current;
      const imgEl = overlayImgRef.current;
      if (!canvas || !imgEl || !overlayNatural) {
        return;
      }
      const rect = imgEl.getBoundingClientRect();
      const displayWidth = rect.width;
      const displayHeight = rect.height;
      if (!displayWidth || !displayHeight) {
        canvas.width = 0;
        canvas.height = 0;
        return;
      }
      const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
      canvas.width = Math.round(displayWidth * dpr);
      canvas.height = Math.round(displayHeight * dpr);
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(dpr, dpr);

      const drawPoint = (pt: SamPoint) => {
        const disp = projectImageToDisplay(pt.x, pt.y);
        if (!disp) {
          return;
        }
        ctx.beginPath();
        ctx.fillStyle = pt.label === 'neg' ? 'rgba(239, 68, 68, 0.9)' : 'rgba(34, 197, 94, 0.9)';
        ctx.strokeStyle = pt.label === 'neg' ? 'rgba(239, 68, 68, 0.9)' : 'rgba(34, 197, 94, 0.9)';
        ctx.lineWidth = 2.5;
        ctx.arc(disp.x, disp.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(disp.x, disp.y, 14, 0, Math.PI * 2);
        ctx.stroke();
      };

      const drawRect = (rectData: Rect, color: string) => {
        const topLeft = projectImageToDisplay(rectData.x0, rectData.y0);
        const bottomRight = projectImageToDisplay(rectData.x1, rectData.y1);
        if (!topLeft || !bottomRight) {
          return;
        }
        ctx.save();
        ctx.strokeStyle = color;
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        ctx.stroke();
        ctx.restore();
      };

      const drawManualQuarter = (qm: QuarterManual) => {
        const center = projectImageToDisplay(qm.cx, qm.cy);
        if (!center) {
          return;
        }
        const rectWidth = overlayNatural.width;
        const rectHeight = overlayNatural.height;
        const dispRect = imgEl.getBoundingClientRect();
        const scaleX = dispRect.width / rectWidth;
        const scaleY = dispRect.height / rectHeight;
        const rx = (qm.MA / 2) * (scaleDown || 1) * scaleX;
        const ry = (qm.ma / 2) * (scaleDown || 1) * scaleY;
        ctx.save();
        ctx.translate(center.x, center.y);
        ctx.rotate((qm.angle * Math.PI) / 180);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)';
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.scale(rx, ry);
        ctx.arc(0, 0, 1, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
        ctx.save();
        ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
        ctx.beginPath();
        ctx.translate(center.x, center.y);
        ctx.rotate((qm.angle * Math.PI) / 180);
        ctx.scale(rx, ry);
        ctx.arc(0, 0, 1, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      };

      samPoints.forEach(drawPoint);
      if (cropRect) {
        drawRect(cropRect, 'rgba(59, 130, 246, 0.85)');
      }
      if (quarterRoi) {
        drawRect(quarterRoi, 'rgba(245, 158, 11, 0.85)');
      }
      if (quarterManual) {
        drawManualQuarter(quarterManual);
      }

      if (activeDrag) {
        if (activeDrag.type === 'crop' || activeDrag.type === 'quarter-roi') {
          const { start, current } = activeDrag;
          if (current) {
            drawRect(
              {
                x0: start.imgX,
                y0: start.imgY,
                x1: current.imgX,
                y1: current.imgY,
              },
              activeDrag.type === 'crop'
                ? 'rgba(59, 130, 246, 0.85)'
                : 'rgba(245, 158, 11, 0.85)',
            );
          }
        } else if (activeDrag.type === 'quarter-manual') {
          const { start, current } = activeDrag;
          if (current) {
            const dx = current.imgX - start.imgX;
            const dy = current.imgY - start.imgY;
            const useEllipse = activeDrag.shiftKey;
            let preview: QuarterManual;
            if (useEllipse) {
              const rx = Math.abs(dx);
              const ry = Math.abs(dy);
              const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
              preview = {
                cx: start.imgX,
                cy: start.imgY,
                MA: Math.max(2, rx * 2),
                ma: Math.max(2, ry * 2),
                angle,
              };
            } else {
              const radius = Math.hypot(dx, dy);
              preview = {
                cx: start.imgX,
                cy: start.imgY,
                MA: Math.max(2, radius * 2),
                ma: Math.max(2, radius * 2),
                angle: 0,
              };
            }
            drawManualQuarter(preview);
          }
        }
      }

      ctx.restore();
    },
    [cropRect, overlayNatural, projectImageToDisplay, quarterManual, quarterRoi, samPoints, scaleDown],
  );

  const resetPreviewCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = 0;
      canvas.height = 0;
    }
  }, []);

  useEffect(() => {
    drawOverlay(dragStateRef.current);
  }, [drawOverlay, overlaySrc]);

  useEffect(() => {
    const handler = () => {
      drawOverlay(dragStateRef.current);
    };
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, [drawOverlay]);

  const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    const target = event.currentTarget;
    setOverlayNatural({ width: target.naturalWidth || 0, height: target.naturalHeight || 0 });
    drawOverlay(dragStateRef.current);
  }, [drawOverlay]);

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!hasImageLoaded) {
        return;
      }
      const pos = mapClientToImage(event.clientX, event.clientY);
      if (!pos) {
        return;
      }
      if (interactionMode === 'point') {
        if (event.button !== 0) {
          return;
        }
        const newPoint: SamPoint = {
          x: Math.round(pos.imgX),
          y: Math.round(pos.imgY),
          label: pointLabel,
        };
        setSamPoints((prev) => {
          const next = [...prev, newPoint];
          schedulePreview({ overrides: { sam_points: next } });
          return next;
        });
        drawOverlay(dragStateRef.current);
        return;
      }

      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      const drag: DragState =
        interactionMode === 'quarter-manual'
          ? {
              type: 'quarter-manual',
              start: { imgX: pos.imgX, imgY: pos.imgY },
              current: { imgX: pos.imgX, imgY: pos.imgY },
              shiftKey: event.shiftKey,
            }
          : {
              type: interactionMode,
              start: { imgX: pos.imgX, imgY: pos.imgY },
              current: { imgX: pos.imgX, imgY: pos.imgY },
            };
      dragStateRef.current = drag;
      drawOverlay(drag);
    },
    [hasImageLoaded, interactionMode, mapClientToImage, pointLabel, schedulePreview, drawOverlay],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragStateRef.current;
      if (!drag) {
        return;
      }
      const pos = mapClientToImage(event.clientX, event.clientY);
      if (!pos) {
        return;
      }
      if (drag.type === 'quarter-manual') {
        drag.current = { imgX: pos.imgX, imgY: pos.imgY };
        drag.shiftKey = event.shiftKey;
      } else {
        drag.current = { imgX: pos.imgX, imgY: pos.imgY };
      }
      drawOverlay(drag);
    },
    [mapClientToImage, drawOverlay],
  );

  const finalizeDragRect = useCallback(
    (drag: Extract<DragState, { type: 'crop' | 'quarter-roi' }>) => {
      if (!imageShape) {
        return;
      }
      const { start, current } = drag;
      if (!current) {
        return;
      }
      const width = imageShape[1];
      const height = imageShape[0];
      const x0 = clamp(Math.round(Math.min(start.imgX, current.imgX)), 0, width - 1);
      const y0 = clamp(Math.round(Math.min(start.imgY, current.imgY)), 0, height - 1);
      const x1 = clamp(Math.round(Math.max(start.imgX, current.imgX)), x0 + 1, width);
      const y1 = clamp(Math.round(Math.max(start.imgY, current.imgY)), y0 + 1, height);
      const minSpan = 8;
      if (x1 - x0 < minSpan || y1 - y0 < minSpan) {
        return;
      }
      const rect: Rect = { x0, y0, x1, y1 };
      if (drag.type === 'crop') {
        setCropRect(rect);
        setSamPoints([]);
        setQuarterManual(null);
        setQuarterRoi(null);
        schedulePreview({
          immediate: true,
          force: true,
          overrides: {
            crop_rect: rect,
            sam_points: [],
            quarter_manual: null,
            quarter_roi: null,
          },
        });
        return;
      } else {
        setQuarterRoi(rect);
        schedulePreview({
          immediate: true,
          force: true,
          overrides: {
            quarter_roi: rect,
          },
        });
        return;
      }
    },
    [imageShape, schedulePreview],
  );

  const finalizeManualQuarter = useCallback(
    (drag: Extract<DragState, { type: 'quarter-manual' }>) => {
      if (!imageShape) {
        return;
      }
      const { start, current } = drag;
      if (!current) {
        return;
      }
      const dx = current.imgX - start.imgX;
      const dy = current.imgY - start.imgY;
      const ellipse = drag.shiftKey;
      let updated: QuarterManual;
      if (ellipse) {
        const rx = Math.abs(dx);
        const ry = Math.abs(dy);
        if (rx < 2 || ry < 2) {
          return;
        }
        updated = {
          cx: clamp(Math.round(start.imgX), 0, imageShape[1]),
          cy: clamp(Math.round(start.imgY), 0, imageShape[0]),
          MA: Math.max(4, Math.round(rx * 2)),
          ma: Math.max(4, Math.round(ry * 2)),
          angle: (Math.atan2(dy, dx) * 180) / Math.PI,
        };
      } else {
        const radius = Math.hypot(dx, dy);
        if (radius < 2) {
          return;
        }
        const axis = Math.max(4, Math.round(radius * 2));
        updated = {
          cx: clamp(Math.round(start.imgX), 0, imageShape[1]),
          cy: clamp(Math.round(start.imgY), 0, imageShape[0]),
          MA: axis,
          ma: axis,
          angle: 0,
        };
      }
      setQuarterManual(updated);
      schedulePreview({ immediate: true, force: true, overrides: { quarter_manual: updated } });
    },
    [imageShape, schedulePreview],
  );

  const handlePointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragStateRef.current;
      if (!drag) {
        return;
      }
      dragStateRef.current = null;
      drawOverlay(null);
      try {
        event.currentTarget.releasePointerCapture(event.pointerId);
      } catch (err) {
        // ignore
      }
      if (drag.type === 'crop' || drag.type === 'quarter-roi') {
        finalizeDragRect(drag);
      } else {
        finalizeManualQuarter(drag);
      }
    },
    [drawOverlay, finalizeDragRect, finalizeManualQuarter],
  );

  const handlePointerCancel = useCallback(() => {
    dragStateRef.current = null;
    drawOverlay(null);
  }, [drawOverlay]);

  const handleUpload = useCallback(
    async (file: File | null) => {
      if (!file) {
        return;
      }
      resetPreviewCanvas();
      setOverlaySrc(null);
      setMaskSrc(null);
      setSvgDataUri(null);
      setStats(null);
      setError(null);
      setSamPoints([]);
      setCropRect(null);
      setQuarterManual(null);
      setQuarterRoi(null);
      setScaleDown(1);
      setUploadStatus(null);
      setIsComputing(true);
      try {
        const info = await uploadImage(file);
        setBaseImageShape(info.shape);
        const initialRotation = ((info.auto_rotation_deg ?? 0) + 360) % 360;
        setImageRotation(initialRotation);
        setAutoRotationDeg(info.auto_rotation_deg ?? 0);
        imageShapeRef.current = info.shape;
        skipAutoPreviewRef.current = true;
        setImageShape(info.shape);
        setUploadStatus({ filename: file.name, converted: info.converted_to });
        const payload = buildPayload({
          sam_points: [],
          crop_rect: null,
          quarter_manual: null,
          quarter_roi: null,
          image_rotation: initialRotation,
          compute_mask: false,
        });
        lastPayloadRef.current = payload;
        await runPreview(payload, { silent: true });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Upload failed';
        setError(message);
      } finally {
        setIsComputing(false);
      }
    },
    [buildPayload, resetPreviewCanvas, runPreview],
  );

  const handleFileInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files && event.target.files[0];
      void handleUpload(file ?? null);
      event.target.value = '';
    },
    [handleUpload],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setOverlayDropActive(false);
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      void handleUpload(file ?? null);
    },
    [handleUpload],
  );

  const handleDragEnter = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setOverlayDropActive(true);
  }, []);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setOverlayDropActive(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const related = event.relatedTarget as Node | null;
    if (!related || !event.currentTarget.contains(related)) {
      setOverlayDropActive(false);
    }
  }, []);

  const rotateImage = useCallback(
    (delta: number) => {
      if (!baseImageShape) {
        return;
      }
      const next = (imageRotation + delta + 360) % 360;
      const baseShape = baseImageShape;
      const nextShape: [number, number] = next % 180 === 0 ? baseShape : [baseShape[1], baseShape[0]];
      setImageRotation(next);
      imageShapeRef.current = nextShape;
      skipAutoPreviewRef.current = true;
      setImageShape(nextShape);
      setSamPoints([]);
      setCropRect(null);
      setQuarterManual(null);
      setQuarterRoi(null);
      schedulePreview({
        immediate: true,
        force: true,
        overrides: {
          sam_points: [],
          crop_rect: null,
          quarter_manual: null,
          quarter_roi: null,
          image_rotation: next,
          compute_mask: false,
        },
      });
    },
    [baseImageShape, imageRotation, schedulePreview],
  );

  const handleUndoPoint = useCallback(() => {
    setSamPoints((prev) => {
      const next = prev.slice(0, -1);
      schedulePreview({ overrides: { sam_points: next } });
      return next;
    });
  }, [schedulePreview]);

  const handleClearPoints = useCallback(() => {
    setSamPoints(() => {
      schedulePreview({ overrides: { sam_points: [] } });
      return [];
    });
  }, [schedulePreview]);

  const handleResetCrop = useCallback(() => {
    if (!cropRect) {
      return;
    }
    setCropRect(() => {
      schedulePreview({ immediate: true, force: true, overrides: { crop_rect: null } });
      return null;
    });
  }, [cropRect, schedulePreview]);

  const handleClearQuarter = useCallback(() => {
    setQuarterManual(() => null);
    setQuarterRoi(() => null);
    schedulePreview({ immediate: true, force: true, overrides: { quarter_manual: null, quarter_roi: null } });
  }, [schedulePreview]);

  const handleDownload = useCallback(async () => {
    const payload = lastPayloadRef.current ?? buildPayload();
    try {
      const blob = await exportSvg(payload);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'tool_cutouts.svg';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Export failed';
      setError(message);
    }
  }, [buildPayload]);

  const quarterStatus = useMemo(() => {
    if (!stats) {
      return '—';
    }
    if (!useQuarter) {
      return 'off';
    }
    return stats.quarter_found ? 'found' : 'not found';
  }, [stats, useQuarter]);

  useEffect(() => {
    return () => {
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }
    };
  }, []);

  const totalArea = stats ? formatNumber(stats.total_area_px2, 0) : '0';

  return (
    <main className={styles.page}>
      <section className={styles.card}>
        <div>
          <h2 className={styles.sectionTitle}>Workflow</h2>
          <p className={styles.helpText}>
            Upload a tool photo, mark positive/negative prompts, crop as needed, and export SVG cutouts. HQ-SAM
            parameters stay server-side so you can focus on layout.
          </p>
        </div>
        <div className={styles.inlineButtons}>
          <button
            type="button"
            className={`${styles.button} ${styles.secondaryButton}`}
            onClick={() => fileInputRef.current?.click()}
          >
            Choose image…
          </button>
        </div>
        <input
          ref={fileInputRef}
          className={styles.hiddenInput}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
        />
        <div className={styles.helpText}>
          Drag a photo directly onto the preview canvas or use the button above.
        </div>
        {uploadStatus && (
          <div className={styles.helpText}>
            Loaded <strong>{uploadStatus.filename}</strong>
            {uploadStatus.converted ? ` · converted to ${uploadStatus.converted}` : ''}
          </div>
        )}

        <div>
          <h3 className={styles.sectionTitle}>Interaction Mode</h3>
          <div className={styles.modeToggle}>
            <button
              type="button"
              className={`${styles.modeButton} ${interactionMode === 'point' ? styles.modeButtonActive : ''}`}
              onClick={() => setInteractionMode('point')}
            >
              Add SAM points
            </button>
            <button
              type="button"
              className={`${styles.modeButton} ${interactionMode === 'crop' ? styles.modeButtonActive : ''}`}
              onClick={() => setInteractionMode('crop')}
            >
              Crop region
            </button>
            <button
              type="button"
              className={`${styles.modeButton} ${interactionMode === 'quarter-roi' ? styles.modeButtonActive : ''}`}
              onClick={() => setInteractionMode('quarter-roi')}
            >
              Quarter ROI
            </button>
            <button
              type="button"
              className={`${styles.modeButton} ${interactionMode === 'quarter-manual' ? styles.modeButtonActive : ''}`}
              onClick={() => setInteractionMode('quarter-manual')}
            >
              Manual quarter fit
            </button>
          </div>
          <div className={styles.helpText}>
            Point mode: click to add prompts (select label below). Quarter manual: click center, drag to size (hold
            Shift for ellipse).
          </div>
        </div>

        <div className={styles.field}>
          <label htmlFor="point-label">SAM point label</label>
          <div className={styles.inlineButtons}>
            <button
              type="button"
              className={`${styles.modeButton} ${pointLabel === 'pos' ? styles.modeButtonActive : ''}`}
              onClick={() => setPointLabel('pos')}
            >
              Positive point
            </button>
            <button
              type="button"
              className={`${styles.modeButton} ${pointLabel === 'neg' ? styles.modeButtonActive : ''}`}
              onClick={() => setPointLabel('neg')}
            >
              Negative point
            </button>
            <button type="button" className={`${styles.button} ${styles.secondaryButton}`} onClick={handleUndoPoint}>
              Undo
            </button>
            <button type="button" className={`${styles.button} ${styles.secondaryButton}`} onClick={handleClearPoints}>
              Clear points
            </button>
          </div>
        </div>

        <div className={styles.field}>
          <label>Crop & quarter</label>
          <div className={styles.inlineButtons}>
            <button type="button" className={`${styles.button} ${styles.secondaryButton}`} onClick={handleResetCrop}>
              Reset crop
            </button>
            <button type="button" className={`${styles.button} ${styles.secondaryButton}`} onClick={handleClearQuarter}>
              Clear quarter marks
            </button>
          </div>
          <div className={styles.switchRow}>
            <label>
              <input
                type="checkbox"
                checked={useQuarter}
                onChange={(event) => {
                  setUseQuarter(event.target.checked);
                  schedulePreview({
                    immediate: true,
                    force: true,
                    overrides: { use_quarter: event.target.checked },
                  });
                }}
              />
              {' '}Use quarter for scale + rectification
            </label>
          </div>
          <div className={styles.rangeField}>
            <span>Quarter diameter (mm)</span>
            <input
              type="number"
              min={10}
              max={40}
              step={0.01}
              value={quarterDiameter}
              onChange={(event) => {
                const next = Number(event.target.value) || QUARTER_DIAMETER_MM;
                setQuarterDiameter(next);
                schedulePreview({ overrides: { quarter_diameter_mm: next } });
              }}
            />
          </div>
          <div className={styles.helpText}>
            Quarter ROI: drag a rectangle around the coin to guide detection. Manual quarter lets you set the ellipse
            explicitly (Shift for ellipse, otherwise circle).
          </div>
        </div>

        <div className={styles.field}>
          <label>Orientation</label>
          <div className={styles.inlineButtons}>
            <button
              type="button"
              className={`${styles.button} ${styles.secondaryButton}`}
              onClick={() => rotateImage(-90)}
              disabled={!baseImageShape}
            >
              Rotate −90°
            </button>
            <button
              type="button"
              className={`${styles.button} ${styles.secondaryButton}`}
              onClick={() => rotateImage(90)}
              disabled={!baseImageShape}
            >
              Rotate +90°
            </button>
          </div>
          <div className={styles.helpText}>
            {baseImageShape
              ? `Current rotation: ${imageRotation}°${autoRotationDeg ? ` (auto rotated ${autoRotationDeg}° on upload)` : ''}`
              : 'Upload an image to enable orientation controls.'}
          </div>
        </div>

        <div className={styles.field}>
          <label>Preview behaviour</label>
          <div className={styles.switchRow}>
            <label>
              <input
                type="checkbox"
                checked={autoPreview}
                onChange={(event) => setAutoPreview(event.target.checked)}
              />{' '}
              Auto-update preview when controls change
            </label>
          </div>
          <div className={styles.switchRow}>
            <label>
              <input
                type="checkbox"
                checked={samAuto}
                onChange={(event) => {
                  setSamAuto(event.target.checked);
                  schedulePreview({ immediate: true, overrides: { sam_auto: event.target.checked } });
                }}
              />{' '}
              SAM auto-box (fills entire image)
            </label>
          </div>
          <div className={styles.inlineButtons}>
            <button
              type="button"
              className={styles.button}
              onClick={() => runPreview(buildPayload(), { silent: false })}
              disabled={!hasImageLoaded}
            >
              Compute preview
            </button>
            <button
              type="button"
              className={styles.button}
              onClick={handleDownload}
              disabled={!svgDataUri}
            >
              Download SVG
            </button>
          </div>
        </div>

        <div className={styles.field}>
          <label>Session</label>
          <div className={styles.helpText}>API endpoint: {API_BASE_URL}</div>
          <div className={styles.helpText}>Scale-down: ×{scaleDown.toFixed(2)}</div>
        </div>
      </section>

      <section className={styles.previewShell}>
        <div className={styles.previewHeader}>
          <div>
            <h1>ToolCutter Studio</h1>
            <p>Segment, tweak, and export machine-ready SVG cutouts.</p>
          </div>
          <div className={styles.statusBar}>
            <span className={`${styles.statusDot} ${isComputing ? styles.statusDotBusy : ''}`} />
            {isComputing ? 'Processing...' : 'Ready'}
          </div>
        </div>

        {error && <div className={styles.errorBar}>{error}</div>}

        <div
          className={`${styles.overlayShell} ${overlayDropActive ? styles.overlayShellDropping : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div
            className={styles.overlayInner}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
          >
            {overlaySrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                ref={overlayImgRef}
                className={styles.overlayImage}
                src={overlaySrc}
                alt="overlay"
                onLoad={handleImageLoad}
              />
            ) : (
              <div style={{ width: 480, height: 320, display: 'grid', placeItems: 'center', color: '#94a3b8' }}>
                Drop an image to begin
              </div>
            )}
            <canvas ref={canvasRef} className={styles.overlayCanvas} />
          </div>
        </div>

        <div className={styles.maskRow}>
          <div className={styles.maskPanel}>
            <div className={styles.fieldRow}>
              <strong>Mask preview</strong>
            </div>
            {maskSrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={maskSrc} alt="mask" />
            ) : (
              <div style={{ height: 220, display: 'grid', placeItems: 'center', color: '#94a3b8' }}>
                Mask will appear after running preview
              </div>
            )}
          </div>
          <div className={styles.maskPanel}>
            <div className={styles.fieldRow}>
              <strong>SVG trace</strong>
              {svgDataUri ? <span className={styles.tag}>live</span> : null}
            </div>
            {svgDataUri ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={svgDataUri} alt="svg preview" />
            ) : (
              <div style={{ height: 220, display: 'grid', placeItems: 'center', color: '#94a3b8' }}>
                Enable auto preview or run compute to generate SVG preview
              </div>
            )}
          </div>
        </div>

        <div className={styles.statsBar}>
          <div className={styles.tag}>Contours {stats ? stats.contour_count : 0}</div>
          <div className={styles.tag}>Area {totalArea} px²</div>
          <div className={styles.tag}>Quarter {quarterStatus}</div>
          <div className={styles.tag}>Rotation {imageRotation}°</div>
          {stats?.mm_per_px ? (
            <div className={styles.tag}>Scale {stats.mm_per_px.toFixed(4)} mm/px</div>
          ) : null}
          <div className={styles.helpText}>Crop: {cropRect ? `${cropRect.x0},${cropRect.y0} → ${cropRect.x1},${cropRect.y1}` : 'none'}</div>
        </div>
      </section>
    </main>
  );
}
