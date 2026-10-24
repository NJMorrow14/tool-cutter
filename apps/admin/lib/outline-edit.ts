import type { Tool } from './types';
import { resizeShapeFromDrag, shapeName, shapePolygon } from './geom';

type Point = number[];
const cross = (a: Point, b: Point) => a[0] * b[1] - a[1] * b[0];
const sub = (a: Point, b: Point) => [a[0] - b[0], a[1] - b[1]];
const area = (p: Point[]) => p.reduce((s, a, i) => s + cross(a, p[(i + 1) % p.length]), 0) / 2;
export function ringCentre(p: Point[]): Point {
  const a = area(p);
  if (Math.abs(a) < 1e-8) return [p.reduce((s,v)=>s+v[0],0)/p.length,p.reduce((s,v)=>s+v[1],0)/p.length];
  const c = [0,0];
  p.forEach((v,i)=>{ const w=p[(i+1)%p.length], k=cross(v,w); c[0]+=(v[0]+w[0])*k; c[1]+=(v[1]+w[1])*k; });
  return c.map(v=>v/(6*a));
}
export function placedOutline(t: Tool): Point[] {
  const c=ringCentre(t.polygon_mm), angle=t.rotation_deg*Math.PI/180, cs=Math.cos(angle), sn=Math.sin(angle);
  return t.polygon_mm.map(([x,y])=>[c[0]+(x-c[0])*cs-(y-c[1])*sn+t.offset_mm.x,c[1]+(x-c[0])*sn+(y-c[1])*cs+t.offset_mm.y]);
}
/** Split a simple closed ring with a finite line crossing exactly twice. */
export function splitRing(ring: Point[], line: [Point, Point]): Point[][] {
  const [a,b]=line, d=sub(b,a), hits: { edge:number; at:number; p:Point }[]=[];
  ring.forEach((p,i)=>{
    const q=ring[(i+1)%ring.length], e=sub(q,p), den=cross(d,e);
    if(Math.abs(den)<1e-8)return;
    const ap=sub(p,a), t=cross(ap,e)/den, u=cross(ap,d)/den;
    if(t < -1e-8 || t > 1+1e-8 || u < -1e-8 || u > 1+1e-8)return;
    const point=[a[0]+t*d[0],a[1]+t*d[1]];
    if(!hits.some(h=>Math.hypot(h.p[0]-point[0],h.p[1]-point[1])<1e-7))hits.push({edge:i,at:u,p:point});
  });
  if(hits.length!==2)throw new Error('Draw the split line across the outline, crossing its edge exactly twice.');
  const expanded:Point[]=[];
  ring.forEach((p,i)=>{ expanded.push(p); for(const h of hits.filter(h=>h.edge===i)) if(h.at>1e-7 && h.at<1-1e-7)expanded.push(h.p); });
  const indices=hits.map(h=>expanded.findIndex(p=>Math.hypot(p[0]-h.p[0],p[1]-h.p[1])<1e-7)).sort((x,y)=>x-y);
  const [i,j]=indices;
  const parts=[expanded.slice(i,j+1),[...expanded.slice(j),...expanded.slice(0,i+1)]];
  if(parts.some(p=>p.length<3||Math.abs(area(p))<1e-5) || Math.abs(parts.reduce((s,p)=>s+Math.abs(area(p)),0)-Math.abs(area(ring)))>1e-4)throw new Error('The line must pass through the inside of the outline.');
  return parts;
}
/** Cut in displayed coordinates while preserving tool placement and calibration. */
export function splitOutlineTool(t: Tool, line: [Point, Point]): Tool[] {
  const c=ringCentre(t.polygon_mm), angle=t.rotation_deg*Math.PI/180, cs=Math.cos(angle), sn=Math.sin(angle);
  const local=line.map(([x,y])=>{const dx=x-c[0]-t.offset_mm.x,dy=y-c[1]-t.offset_mm.y;return [c[0]+dx*cs+dy*sn,c[1]-dx*sn+dy*cs];}) as [Point,Point];
  const span=(p:Point[])=>Math.hypot(Math.max(...p.map(v=>v[0]))-Math.min(...p.map(v=>v[0])),Math.max(...p.map(v=>v[1]))-Math.min(...p.map(v=>v[1])));
  const scale=t.polygon_px.length>=3?span(t.polygon_px)/span(t.polygon_mm):0;
  return splitRing(t.polygon_mm,local).map((ring,i)=>{
    const nc=ringCentre(ring), dx=nc[0]-c[0],dy=nc[1]-c[1];
    const pixels=scale?ring.map(p=>p.map(v=>v*scale)):[];
    return {...t,id:i === 0 ? t.id : (globalThis.crypto?.randomUUID?.() ?? `split-${Date.now()}-${Math.random().toString(36).slice(2)}`),name:`${t.name} ${i+1}`,polygon_mm:ring,polygon_px:pixels,auto_polygon_px:pixels,
      area_mm2:Math.abs(area(ring)),offset_mm:{x:t.offset_mm.x+dx*cs-dy*sn-dx,y:t.offset_mm.y+dx*sn+dy*cs-dy},
      shape:undefined,points:[],box:null,notch:null,pending:false,error:null,edited:true};
  });
}
export function verticesInBox(points: Point[], a: Point, b: Point): number[] {
  return points.flatMap(([x,y],i)=>x>=Math.min(a[0],b[0]) && x<=Math.max(a[0],b[0]) && y>=Math.min(a[1],b[1]) && y<=Math.max(a[1],b[1])?[i]:[]);
}

export function unplacedOutline(t: Tool, points: Point[]): Point[] {
  const c=ringCentre(t.polygon_mm), angle=t.rotation_deg*Math.PI/180, cs=Math.cos(angle), sn=Math.sin(angle);
  return points.map(([x,y])=>{const dx=x-c[0]-t.offset_mm.x,dy=y-c[1]-t.offset_mm.y;return [c[0]+dx*cs+dy*sn,c[1]-dx*sn+dy*cs];});
}
/** Keep unaffected vertices stationary when the ring's rotation centre changes. */
export function editToolOutline(t: Tool, ring: Point[]): Tool {
  const c=ringCentre(t.polygon_mm), nc=ringCentre(ring), angle=t.rotation_deg*Math.PI/180, cs=Math.cos(angle), sn=Math.sin(angle), dx=nc[0]-c[0],dy=nc[1]-c[1];
  const span=(p:Point[])=>Math.hypot(Math.max(...p.map(v=>v[0]))-Math.min(...p.map(v=>v[0])),Math.max(...p.map(v=>v[1]))-Math.min(...p.map(v=>v[1])));
  const scale=t.polygon_px.length>=3?span(t.polygon_px)/span(t.polygon_mm):0;
  return {...t,polygon_mm:ring,polygon_px:scale?ring.map(p=>p.map(v=>v*scale)):[],area_mm2:Math.abs(area(ring)),shape:undefined,edited:true,pending:false,
    auto_polygon_px:t.auto_polygon_px ?? t.polygon_px,
    offset_mm:{x:t.offset_mm.x+dx*cs-dy*sn-dx,y:t.offset_mm.y+dx*sn+dy*cs-dy}};
}

/** A handle drag on a parametric shape RESIZES it (circle diameter, rect width/height ...) instead of turning it into a
 *  free polygon with hundreds of nodes. `ring` is the dragged outline in the tool's own frame. Keeps the shape's centre
 *  where it was on the mat. Returns null when the tool is not a resizable shape (then edit it as a free outline). */
export function resizeShapeTool(t: Tool, ring: Point[]): Tool | null {
  if (!t.shape || t.shape.kind === 'poly') return null;
  const res = resizeShapeFromDrag(t.shape, t.polygon_mm, ring);
  if (!res) return t;
  const poly = shapePolygon(res.spec);
  const angle = t.rotation_deg * Math.PI / 180, cs = Math.cos(angle), sn = Math.sin(angle);
  // the centre moved by centreShift in the shape frame; rotation is about the ring centre, so the placed centre moves by
  // the rotated shift — compensate in offset_mm
  const dx = res.centreShift.x, dy = res.centreShift.y;
  const autoNamed = t.name === shapeName(t.shape);
  return { ...t, shape: res.spec, polygon_mm: poly, area_mm2: Math.abs(area(poly)), name: autoNamed ? shapeName(res.spec) : t.name,
    offset_mm: { x: t.offset_mm.x - (dx * cs - dy * sn), y: t.offset_mm.y - (dx * sn + dy * cs) } };
}
