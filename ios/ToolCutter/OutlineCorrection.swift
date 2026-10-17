import Foundation
import CoreGraphics

// The finger retrace for the per-tool capture (docs/tool-library.md, "Phone"). Foundation / CoreGraphics only (plus
// the Prompt type from SAM2Engine.swift), so `ios/tests/run_outline_correction_test.sh` checks it on the Mac.
// Pixel convention as OutlineEngine.swift: pixel (i, j) has its centre at (i + 0.5, j + 0.5); rings are y-down.
//
// The rule that matters to the person: a stroke changes the outline NEAR THE STROKE ONLY. Everything outside the
// stroke's corridor keeps its vertices, so an outline they already accepted elsewhere does not drift.

public struct OutlineState: Equatable, Sendable {
    public var ring: [CGPoint]
    public var mask: BitMask
    public init(ring: [CGPoint], mask: BitMask) { self.ring = ring; self.mask = mask }
}

public enum OutlineCorrection {
    /// How the last correction was made (for tests and the status line).
    public enum Method: String, Sendable { case sam, geometric, snapped }

    /// Corridor half-width around a stroke, px: 2 % of the image's long side, at least 12 px.
    public static func corridorRadius(imageWidth w: Int, imageHeight h: Int) -> Double {
        max(12, 0.02 * Double(max(w, h)))
    }

    /// Mask -> the outline the person first sees: largest component, simplified, smoothed, and its own raster.
    public static func initialOutline(mask: BitMask, minAreaPx: Int) -> OutlineState? {
        guard let traced = OutlineEngine.largestComponentPolygon(mask: mask, minArea: minAreaPx) else { return nil }
        var ring = OutlineEngine.simplify(traced, tolerance: 1.0)
        if ring.count >= 3 { ring = OutlineEngine.smooth(ring, sigma: 2, step: 2) }
        guard ring.count >= 3 else { return nil }
        return OutlineState(ring: ring, mask: OutlineEngine.rasterize(ring, width: mask.width, height: mask.height))
    }

    /// A closed finger loop drawn when there is no outline at all (no Vision, no SAM): the loop IS the outline.
    public static func outlineFromLoop(_ stroke: [CGPoint], imageWidth w: Int, imageHeight h: Int, minAreaPx: Int) -> OutlineState? {
        guard stroke.count >= 8 else { return nil }
        var ring = OutlineEngine.simplify(stroke, tolerance: 1.0)
        if ring.count < 3 { return nil }
        ring = OutlineEngine.smooth(ring, sigma: 1.5, step: 2)
        if OutlineEngine.isSelfIntersecting(ring) {
            let m = OutlineEngine.rasterize(ring, width: w, height: h)
            return initialOutline(mask: m, minAreaPx: minAreaPx)
        }
        guard OutlineEngine.area(ring) >= Double(minAreaPx) else { return nil }
        return OutlineState(ring: ring, mask: OutlineEngine.rasterize(ring, width: w, height: h))
    }

    /// THE RETRACE. See the file header; steps (a)-(d) as in docs/tool-library.md.
    public static func applyStroke(state: OutlineState, stroke: [CGPoint], imageWidth w: Int, imageHeight h: Int,
                                   gradient: [Float]?,
                                   sam: ((_ prompts: [SAM2Engine.Prompt]) throws -> BitMask)?) -> OutlineState {
        applyStrokeDetailed(state: state, stroke: stroke, imageWidth: w, imageHeight: h, gradient: gradient, sam: sam).state
    }

    public static func applyStrokeDetailed(state: OutlineState, stroke rawStroke: [CGPoint], imageWidth w: Int, imageHeight h: Int,
                                           gradient: [Float]?,
                                           sam: ((_ prompts: [SAM2Engine.Prompt]) throws -> BitMask)?) -> (state: OutlineState, method: Method) {
        let stroke = OutlineEngine.dedupe(rawStroke)
        guard stroke.count >= 2, state.ring.count >= 3, OutlineEngine.polylineLength(stroke) >= 3 else { return (state, .geometric) }
        let r = corridorRadius(imageWidth: w, imageHeight: h)

        // (a) geometric: the stroke replaces the nearest run of the outline. `splice` is OutlineEngine.replaceRun's rule
        // (cut at the vertices nearest the stroke's ends, replace the closer arc) with a LOCAL repair: replaceRun
        // re-traces the whole ring when a join makes a tiny loop, which moved every vertex 0.1-0.3 px per stroke.
        let geo = splice(state.ring, with: stroke)
        let corridor = OutlineEngine.corridorMask(stroke: stroke, radius: r, width: w, height: h)

        var ring = geo
        var method = Method.geometric
        // (b) SAM re-segments inside the corridor; adopted only when its edge there follows the stroke
        if let sam {
            let geoMask = OutlineEngine.rasterize(geo, width: w, height: h)
            let prompts = samPrompts(stroke: stroke, insideMask: geoMask, oldMask: state.mask, corridor: corridor, radius: r)
            if prompts.contains(where: { $0.label == 1 }), let samMask = try? sam(prompts), samMask.width == w, samMask.height == h {
                let merged = OutlineEngine.merge(old: state.mask, new: samMask, corridor: corridor)
                if let traced = OutlineEngine.largestComponentPolygon(mask: merged, minArea: max(25, state.mask.count / 4)),
                   let run = corridorRun(of: traced, stroke: stroke, radius: r),
                   acceptSAMRun(run, stroke: stroke, radius: r) {
                    let oriented = OutlineEngine.dist2(run[0], stroke[0]) <= OutlineEngine.dist2(run[run.count - 1], stroke[0]) ? run : run.reversed()
                    ring = splice(state.ring, with: oriented)
                    method = .sam
                }
            }
        }
        // (c) no SAM (or SAM refused): nudge the drawn line onto the photo's edge, inside the corridor only
        if method == .geometric, let gradient, gradient.count == w * h {
            let dense = OutlineEngine.densify(ring, maxStep: 2)
            let snapped = OutlineEngine.snapToEdges(dense, gradientMagnitude: gradient, width: w, height: h, maxShift: 0.3 * r)
            if snapped.count == dense.count {
                var out = dense
                var moved = false
                for i in 0..<dense.count where inCorridor(dense[i], corridor) {
                    if out[i] != snapped[i] { moved = true }
                    out[i] = snapped[i]
                }
                // drop the densified extras outside the corridor again so untouched vertices stay exactly as they were
                if moved {
                    let keep = Set(ring.map { HashablePoint($0) })
                    ring = zip(dense, out).compactMap { (d, o) in inCorridor(d, corridor) || keep.contains(HashablePoint(d)) ? o : nil }
                    method = .snapped
                }
            }
        }
        // (d) tidy ONLY the vertices inside the corridor
        ring = smoothInside(ring, corridor: corridor)
        if ring.count < 3 { return (state, .geometric) }
        ring = untangle(ring, orientationOf: state.ring)
        return (OutlineState(ring: ring, mask: OutlineEngine.rasterize(ring, width: w, height: h)), method)
    }

    // MARK: - SAM prompts

    /// Prompts for a correction: up to 5 FOREGROUND points sampled evenly along the stroke and pushed 0.5 r to the
    /// side that is INSIDE the geometrically corrected outline, up to 5 BACKGROUND points pushed 0.5 r outside, and up
    /// to 2 foreground anchors deep inside the old mask (outside the corridor, so they cannot contradict the stroke).
    public static func samPrompts(stroke: [CGPoint], insideMask: BitMask, oldMask: BitMask, corridor: BitMask, radius r: Double) -> [SAM2Engine.Prompt] {
        let w = insideMask.width, h = insideMask.height
        var fg: [SAM2Engine.Prompt] = [], bg: [SAM2Engine.Prompt] = []
        func inside(_ p: CGPoint) -> Bool? {
            let x = Int(floor(p.x)), y = Int(floor(p.y))
            guard x >= 0, y >= 0, x < w, y < h else { return nil }
            return insideMask[x, y] != 0
        }
        let total = OutlineEngine.polylineLength(stroke)
        if total > 0 {
            for k in 0..<5 {
                let sk = total * (Double(k) + 0.5) / 5
                let p = pointAndTangent(stroke, at: sk).0
                // direction over +-0.5 r of the stroke, not the local segment: finger jitter turns local segments
                // across the edge, which would put the fg/bg pair ALONG it
                let pa = pointAndTangent(stroke, at: max(0, sk - 0.5 * r)).0, pb = pointAndTangent(stroke, at: min(total, sk + 0.5 * r)).0
                let tl = max(1e-9, hypot(pb.x - pa.x, pb.y - pa.y))
                let nx = -(pb.y - pa.y) / tl, ny = (pb.x - pa.x) / tl
                let a = CGPoint(x: p.x + nx * 0.5 * r, y: p.y + ny * 0.5 * r)
                let b = CGPoint(x: p.x - nx * 0.5 * r, y: p.y - ny * 0.5 * r)
                switch (inside(a), inside(b)) {
                case (true?, false?): fg.append(.init(x: a.x, y: a.y, label: 1)); bg.append(.init(x: b.x, y: b.y, label: 0))
                case (false?, true?): fg.append(.init(x: b.x, y: b.y, label: 1)); bg.append(.init(x: a.x, y: a.y, label: 0))
                case (false?, nil): bg.append(.init(x: a.x, y: a.y, label: 0))
                case (nil, false?): bg.append(.init(x: b.x, y: b.y, label: 0))
                case (true?, nil): fg.append(.init(x: a.x, y: a.y, label: 1))
                case (nil, true?): fg.append(.init(x: b.x, y: b.y, label: 1))
                default: break       // both on one side: this sample says nothing about the edge
                }
            }
        }
        let anchors = deepAnchors(mask: oldMask, avoid: corridor, count: 2).map { SAM2Engine.Prompt(x: $0.x, y: $0.y, label: 1) }
        return Array((fg + bg + anchors).prefix(SAM2Engine.maxPrompts))
    }

    /// Up to `count` points well inside `mask` (and outside `avoid`): the deepest pixel by a chamfer distance, then the
    /// deepest one at least 30 % of the mask's long extent away from it. Computed on a grid of <= ~512 px.
    public static func deepAnchors(mask: BitMask, avoid: BitMask?, count: Int) -> [CGPoint] {
        let W = mask.width, H = mask.height
        guard W > 0, H > 0, count > 0 else { return [] }
        let s = max(1, max(W, H) / 512)
        let gw = (W + s - 1) / s, gh = (H + s - 1) / s
        let big = 1 << 28
        var d = [Int](repeating: 0, count: gw * gh)
        var minX = Int.max, maxX = Int.min, minY = Int.max, maxY = Int.min
        for gy in 0..<gh {
            for gx in 0..<gw {
                let x = min(W - 1, gx * s + s / 2), y = min(H - 1, gy * s + s / 2)
                let on = mask[x, y] != 0 && !(avoid.map { $0.width == W && $0.height == H && $0[x, y] != 0 } ?? false)
                d[gy * gw + gx] = on ? big : 0
                if mask[x, y] != 0 { minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y) }
            }
        }
        guard minX <= maxX else { return [] }
        // two-pass 3-4 chamfer; the grid border counts as outside
        for gy in 0..<gh { for gx in 0..<gw where d[gy * gw + gx] > 0 {
            var v = d[gy * gw + gx]
            v = min(v, gx > 0 ? d[gy * gw + gx - 1] + 3 : 3)
            v = min(v, gy > 0 ? d[(gy - 1) * gw + gx] + 3 : 3)
            v = min(v, gx > 0 && gy > 0 ? d[(gy - 1) * gw + gx - 1] + 4 : 4)
            v = min(v, gx < gw - 1 && gy > 0 ? d[(gy - 1) * gw + gx + 1] + 4 : 4)
            d[gy * gw + gx] = v
        } }
        for gy in stride(from: gh - 1, through: 0, by: -1) { for gx in stride(from: gw - 1, through: 0, by: -1) where d[gy * gw + gx] > 0 {
            var v = d[gy * gw + gx]
            v = min(v, gx < gw - 1 ? d[gy * gw + gx + 1] + 3 : 3)
            v = min(v, gy < gh - 1 ? d[(gy + 1) * gw + gx] + 3 : 3)
            v = min(v, gx < gw - 1 && gy < gh - 1 ? d[(gy + 1) * gw + gx + 1] + 4 : 4)
            v = min(v, gx > 0 && gy < gh - 1 ? d[(gy + 1) * gw + gx - 1] + 4 : 4)
            d[gy * gw + gx] = v
        } }
        func centre(_ i: Int) -> CGPoint {
            CGPoint(x: Double(min(W - 1, (i % gw) * s + s / 2)) + 0.5, y: Double(min(H - 1, (i / gw) * s + s / 2)) + 0.5)
        }
        var best = -1, bestV = 0
        for i in 0..<d.count where d[i] > bestV { bestV = d[i]; best = i }
        guard best >= 0 else { return [] }
        var out = [centre(best)]
        if count >= 2 {
            let minSep = 0.3 * Double(max(maxX - minX, maxY - minY))
            let a = out[0]
            var b2 = -1, bV = 0
            for i in 0..<d.count where d[i] > bV {
                let c = centre(i)
                if hypot(c.x - a.x, c.y - a.y) >= minSep { bV = d[i]; b2 = i }
            }
            if b2 >= 0 { out.append(centre(b2)) }
        }
        return out
    }

    // MARK: - Pieces (internal; tested)

    /// The longest contiguous run of `ring` (densified to 1 px) whose points lie within `radius` of the stroke.
    static func corridorRun(of ring: [CGPoint], stroke: [CGPoint], radius: Double) -> [CGPoint]? {
        let dense = OutlineEngine.densify(ring, maxStep: 1)
        let n = dense.count
        guard n >= 3 else { return nil }
        let r2 = radius * radius
        let flag = dense.map { OutlineEngine.polylineDist2($0, stroke) <= r2 }
        if flag.allSatisfy({ $0 }) { return nil }        // the whole outline is "the correction": not a local edit
        guard let start = flag.firstIndex(of: false) else { return nil }
        var best: (Int, Int) = (0, 0), cur = 0, curStart = 0
        for k in 1...n {
            let i = (start + k) % n
            if flag[i] { if cur == 0 { curStart = i }; cur += 1; if cur > best.1 { best = (curStart, cur) } } else { cur = 0 }
        }
        guard best.1 >= 2 else { return nil }
        return (0..<best.1).map { dense[(best.0 + $0) % n] }
    }

    /// SAM's edge inside the corridor must follow the stroke: mean distance <= 0.6 r, and lie within r of >= 60 % of it.
    static func acceptSAMRun(_ run: [CGPoint], stroke: [CGPoint], radius r: Double) -> Bool {
        guard run.count >= 2 else { return false }
        var s = 0.0
        for p in run { s += sqrt(OutlineEngine.polylineDist2(p, stroke)) }
        let mean = s / Double(run.count)
        guard mean <= 0.6 * r else { return false }
        // coverage, not length: most of the stroke must have SAM's edge near it (a jittery touch path's length is
        // inflated many times, so comparing lengths refused good answers)
        let total = OutlineEngine.polylineLength(stroke)
        let k = 40
        var near = 0
        for i in 0..<k {
            let p = pointAndTangent(stroke, at: total * (Double(i) + 0.5) / Double(k)).0
            if OutlineEngine.polylineDist2(p, run) <= r * r { near += 1 }
        }
        return Double(near) / Double(k) >= 0.6
    }

    /// Replace the run of `ring` nearest `run` by `run` (OutlineEngine.replaceRun's choice of arc), keeping every
    /// other vertex exactly; loops made at the joins are cut out locally.
    static func splice(_ ring: [CGPoint], with run: [CGPoint]) -> [CGPoint] {
        guard ring.count >= 3, let first = run.first, let last = run.last else { return ring }
        let originals = Set(ring.map { HashablePoint($0) })
        let dense = OutlineEngine.densify(ring, maxStep: 1.0)
        let n = dense.count
        func nearest(_ p: CGPoint) -> Int {
            var bi = 0, bd = Double.infinity
            for i in 0..<n { let d = OutlineEngine.dist2(dense[i], p); if d < bd { bd = d; bi = i } }
            return bi
        }
        let i0 = nearest(first), i1 = nearest(last)
        func arc(_ a: Int, _ b: Int) -> [Int] {
            var out = [Int](); var i = (a + 1) % n
            while i != b { out.append(i); i = (i + 1) % n }
            return out
        }
        var kept: [Int]
        var tail: [CGPoint]
        if i0 == i1 {
            if OutlineEngine.polylineLength(run) < 2 { return ring }
            kept = arc(i0, i0) + [i0]
            tail = run
        } else {
            let a = arc(i0, i1), b = arc(i1, i0)
            func meanDist(_ idx: [Int]) -> Double {
                idx.isEmpty ? 0 : idx.reduce(0.0) { $0 + sqrt(OutlineEngine.polylineDist2(dense[$1], run)) } / Double(idx.count)
            }
            if meanDist(a) <= meanDist(b) { kept = [i1] + b + [i0]; tail = run }
            else { kept = [i0] + a + [i1]; tail = run.reversed() }
        }
        // the densified extras lie ON the old edges: drop them (the joins stay) so the kept part is the old ring itself
        let joins = Set([kept.first!, kept.last!])
        var out = kept.filter { joins.contains($0) || originals.contains(HashablePoint(dense[$0])) }.map { dense[$0] }
        out += tail
        out = OutlineEngine.dedupe(out)
        if out.count < 3 { return ring }
        return untangle(out, orientationOf: ring)
    }

    /// Cut small loops out where two edges cross (the shorter side of each crossing goes, replaced by the crossing
    /// point); whole-ring re-tracing (OutlineEngine.repairRing) only if that does not settle it. Orientation as `reference`.
    static func untangle(_ ringIn: [CGPoint], orientationOf reference: [CGPoint]) -> [CGPoint] {
        var ring = ringIn
        var rounds = 0
        while rounds < 64, let (i, j, x) = firstCrossing(ring) {
            rounds += 1
            let n = ring.count
            let inner = j - i                       // vertices i+1 ... j form one side
            if inner <= n - inner {
                ring = Array(ring[...i]) + [x] + Array(ring[(j + 1)...])
            } else {
                ring = [x] + Array(ring[(i + 1)...j])
            }
            if ring.count < 3 { return OutlineEngine.repairRing(ringIn, orientationOf: reference) ?? ringIn }
        }
        if OutlineEngine.isSelfIntersecting(ring) {
            ring = OutlineEngine.repairRing(ring, orientationOf: reference) ?? ring
        }
        if OutlineEngine.signedArea(ring) * OutlineEngine.signedArea(reference) < 0 { ring.reverse() }
        return ring
    }

    /// First pair of non-adjacent crossing edges (i, i+1) x (j, j+1), i < j, and the crossing point.
    static func firstCrossing(_ p: [CGPoint]) -> (Int, Int, CGPoint)? {
        let n = p.count
        guard n >= 4 else { return nil }
        // edge boxes once, then a sweep-free O(n^2) box test (rings here are a few thousand vertices at most)
        var minx = [Double](repeating: 0, count: n), maxx = minx, miny = minx, maxy = minx
        for i in 0..<n {
            let a = p[i], b = p[(i + 1) % n]
            minx[i] = min(a.x, b.x); maxx[i] = max(a.x, b.x); miny[i] = min(a.y, b.y); maxy[i] = max(a.y, b.y)
        }
        for i in 0..<(n - 2) {
            for j in (i + 2)..<n {
                if i == 0 && j == n - 1 { continue }
                if maxx[i] < minx[j] || maxx[j] < minx[i] || maxy[i] < miny[j] || maxy[j] < miny[i] { continue }
                if let x = intersection(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]) { return (i, j, x) }
            }
        }
        return nil
    }

    static func intersection(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint, _ d: CGPoint) -> CGPoint? {
        let r = CGPoint(x: b.x - a.x, y: b.y - a.y), s = CGPoint(x: d.x - c.x, y: d.y - c.y)
        let den = Double(r.x * s.y - r.y * s.x)
        if abs(den) < 1e-12 { return nil }
        let qp = CGPoint(x: c.x - a.x, y: c.y - a.y)
        let t = Double(qp.x * s.y - qp.y * s.x) / den, u = Double(qp.x * r.y - qp.y * r.x) / den
        guard t >= 0, t <= 1, u >= 0, u <= 1 else { return nil }
        return CGPoint(x: a.x + r.x * t, y: a.y + r.y * t)
    }

    static func inCorridor(_ p: CGPoint, _ c: BitMask) -> Bool {
        let x = Int(floor(p.x)), y = Int(floor(p.y))
        return x >= 0 && y >= 0 && x < c.width && y < c.height && c[x, y] != 0
    }

    /// simplify(1.0) + smooth(sigma 1.5, step 2) applied to each run of vertices inside the corridor; the vertex just
    /// outside each end is a fixed anchor, so the rest of the ring is returned untouched.
    static func smoothInside(_ ring: [CGPoint], corridor: BitMask) -> [CGPoint] {
        let n = ring.count
        guard n >= 3 else { return ring }
        let flag = ring.map { inCorridor($0, corridor) }
        if flag.allSatisfy({ $0 }) {
            let s = OutlineEngine.simplify(ring, tolerance: 1.0)
            return s.count >= 3 ? OutlineEngine.smooth(s, sigma: 1.5, step: 2) : ring
        }
        guard let start = flag.firstIndex(of: false) else { return ring }
        var out: [CGPoint] = []
        var k = 0
        while k < n {
            let i = (start + k) % n
            if !flag[i] { out.append(ring[i]); k += 1; continue }
            // run of corridor vertices i ..< j, anchored by the outside vertices before and after it
            var len = 0
            while k + len < n && flag[(start + k + len) % n] { len += 1 }
            let prev = ring[(i - 1 + n) % n], next = ring[(start + k + len) % n]
            let run = (0..<len).map { ring[(i + $0) % n] }
            let tidy = smoothOpen([prev] + run + [next], simplifyTol: 1.0, sigma: 1.5, step: 2)
            out.append(contentsOf: tidy.dropFirst().dropLast())
            k += len
        }
        return out
    }

    /// Open polyline: Douglas-Peucker (endpoints kept), resample every `step`, Gaussian along it with the endpoints
    /// pinned and the smoothing faded in over 3 sigma from each end so the ring does not step where the run meets it.
    static func smoothOpen(_ line: [CGPoint], simplifyTol: Double, sigma: Double, step: Double) -> [CGPoint] {
        guard line.count >= 3 else { return line }
        let simp = simplifyOpen(line, tol: simplifyTol)
        let total = OutlineEngine.polylineLength(simp)
        guard total > 2 * step else { return simp }
        let m = max(2, Int((total / step).rounded()))
        var pts = (0...m).map { pointAndTangent(simp, at: total * Double($0) / Double(m)).0 }
        let sp = total / Double(m)
        let half = max(1, Int(ceil(3 * sigma / sp)))
        var sm = pts
        for i in 1..<m {
            var sx = 0.0, sy = 0.0, sw = 0.0
            for o in -half...half {
                let j = i + o
                guard j >= 0, j <= m else { continue }
                let wgt = exp(-0.5 * pow(Double(o) * sp / sigma, 2))
                sx += pts[j].x * wgt; sy += pts[j].y * wgt; sw += wgt
            }
            let fromEnd = Double(min(i, m - i)) * sp
            let a = min(1, fromEnd / (3 * sigma))
            sm[i] = CGPoint(x: pts[i].x + (sx / sw - pts[i].x) * a, y: pts[i].y + (sy / sw - pts[i].y) * a)
        }
        pts = sm
        pts[0] = line[0]; pts[m] = line[line.count - 1]
        return pts
    }

    static func simplifyOpen(_ p: [CGPoint], tol: Double) -> [CGPoint] {
        guard p.count > 2 else { return p }
        var keep = [Bool](repeating: false, count: p.count)
        keep[0] = true; keep[p.count - 1] = true
        var stack = [(0, p.count - 1)]
        let t2 = tol * tol
        while let (a, b) = stack.popLast() {
            guard b > a + 1 else { continue }
            var bi = -1, bd = -1.0
            for i in (a + 1)..<b { let d = OutlineEngine.segDist2(p[i], p[a], p[b]); if d > bd { bd = d; bi = i } }
            if bd > t2 { keep[bi] = true; stack.append((a, bi)); stack.append((bi, b)) }
        }
        return p.indices.filter { keep[$0] }.map { p[$0] }
    }

    /// Point at arc length `s` along an open polyline, and the unit tangent there.
    static func pointAndTangent(_ line: [CGPoint], at s: Double) -> (CGPoint, CGPoint) {
        var acc = 0.0
        for i in 0..<(line.count - 1) {
            let a = line[i], b = line[i + 1]
            let l = hypot(b.x - a.x, b.y - a.y)
            if l <= 0 { continue }
            if acc + l >= s || i == line.count - 2 {
                let t = max(0, min(1, (s - acc) / l))
                return (CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t), CGPoint(x: (b.x - a.x) / l, y: (b.y - a.y) / l))
            }
            acc += l
        }
        return (line.last ?? .zero, CGPoint(x: 1, y: 0))
    }

    private struct HashablePoint: Hashable {
        let x: Double, y: Double
        init(_ p: CGPoint) { x = Double(p.x); y = Double(p.y) }
    }
}
