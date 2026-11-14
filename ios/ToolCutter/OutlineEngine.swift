import Foundation
import CoreGraphics
import simd

// On-device outlining maths for the tool-by-tool phone capture (docs/tool-library.md, "Phone").
// Foundation / CoreGraphics / simd only, so `ios/tests/run_outline_engine_test.sh` compiles it on the Mac with swiftc.
//
// PIXEL CONVENTION (everything in this file): pixel (i, j) covers [i, i+1) x [j, j+1); its CENTRE is (i + 0.5, j + 0.5).
// Rings are in that continuous pixel space (y down). A ring returned by `largestComponentPolygon` runs through the
// midpoints of the mask's boundary pixel edges, so a 200 x 100 px block comes back 200 x 100 (corners chamfered by
// half a pixel), and `rasterize` (pixel-centre test) turns it back into the same mask.
// Camera intrinsics (cx, cy) follow the usual convention that INTEGER (u, v) is a pixel centre, so a ring point
// (x, y) is the ray through (x - 0.5, y - 0.5) in intrinsics terms.
// ORIENTATION: `largestComponentPolygon` returns rings whose signed shoelace sum is NEGATIVE in these y-down
// coordinates (counter-clockwise as drawn on screen). Functions that take a ring work with either orientation and
// return rings in the orientation they were given.

public struct BitMask: Equatable, Sendable {
    public var width: Int
    public var height: Int
    /// 0 / 1, row-major, `width * height` bytes.
    public var data: [UInt8]

    public init(width: Int, height: Int) {
        self.width = max(0, width)
        self.height = max(0, height)
        self.data = [UInt8](repeating: 0, count: self.width * self.height)
    }

    public subscript(x: Int, y: Int) -> UInt8 {
        get { data[y * width + x] }
        set { data[y * width + x] = newValue }
    }

    /// Number of set pixels.
    public var count: Int { data.reduce(0) { $0 + Int($1 != 0 ? 1 : 0) } }
}

public struct CameraIntrinsics: Equatable, Sendable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var width: Int
    public var height: Int

    public init(fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy; self.width = width; self.height = height
    }

    /// The same camera at another resolution (e.g. colour intrinsics -> the depth map's size).
    public func scaled(toWidth w: Int, height h: Int) -> CameraIntrinsics {
        let sx = Double(w) / Double(width), sy = Double(h) / Double(height)
        return CameraIntrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy, width: w, height: h)
    }
}

public struct Plane: Equatable, Sendable {
    /// Unit normal in camera coordinates (+x right, +y DOWN, +z forward), pointing toward the camera (normal.z < 0).
    public var normal: SIMD3<Double>
    /// Metres: dot(normal, P) == d for points on the plane.
    public var d: Double
    public var inlierFraction: Double
    /// Angle between the normal and the optical axis, degrees.
    public var tiltDeg: Double
    /// Distance to the plane along the optical axis (the ray through the image centre), metres.
    public var distanceM: Double

    public init(normal: SIMD3<Double>, d: Double, inlierFraction: Double) {
        var n = simd_normalize(normal), dd = d
        if n.z > 0 { n = -n; dd = -dd }
        self.normal = n
        self.d = dd
        self.inlierFraction = inlierFraction
        self.tiltDeg = acos(min(1, abs(n.z))) * 180 / .pi
        self.distanceM = abs(n.z) > 1e-9 ? dd / n.z : .infinity
    }
}

public enum OutlineEngine {

    // MARK: - Mask -> polygon

    /// Outer contour (holes filled) of the largest 8-connected component, in pixel coordinates (see the file header
    /// for the convention and orientation). nil when no component has at least `minArea` pixels.
    public static func largestComponentPolygon(mask: BitMask, minArea: Int) -> [CGPoint]? {
        let W = mask.width, H = mask.height
        guard W > 0, H > 0 else { return nil }
        var labels = [Int32](repeating: 0, count: W * H)
        var bestLabel: Int32 = 0, bestCount = 0, bestStart = -1
        var next: Int32 = 0
        var stack = [Int]()
        stack.reserveCapacity(4096)
        mask.data.withUnsafeBufferPointer { m in
            labels.withUnsafeMutableBufferPointer { lab in
                for start in 0..<(W * H) where m[start] != 0 && lab[start] == 0 {
                    next += 1
                    lab[start] = next
                    stack.append(start)
                    var count = 0
                    while let p = stack.popLast() {
                        count += 1
                        let x = p % W, y = p / W
                        for dy in -1...1 {
                            let yy = y + dy
                            if yy < 0 || yy >= H { continue }
                            for dx in -1...1 {
                                let xx = x + dx
                                if xx < 0 || xx >= W || (dx == 0 && dy == 0) { continue }
                                let q = yy * W + xx
                                if m[q] != 0 && lab[q] == 0 { lab[q] = next; stack.append(q) }
                            }
                        }
                    }
                    if count > bestCount { bestCount = count; bestLabel = next; bestStart = start }
                }
            }
        }
        guard bestCount >= max(1, minArea), bestStart >= 0 else { return nil }

        // Component bounding box, then fill its holes: background reachable (4-connected) from outside the box stays
        // background, everything else is the component.
        var x0 = W, y0 = H, x1 = -1, y1 = -1
        for i in 0..<(W * H) where labels[i] == bestLabel {
            let x = i % W, y = i / W
            if x < x0 { x0 = x }; if x > x1 { x1 = x }; if y < y0 { y0 = y }; if y > y1 { y1 = y }
        }
        let bw = x1 - x0 + 3, bh = y1 - y0 + 3          // 1 px padding all round
        var filled = [UInt8](repeating: 1, count: bw * bh)  // 1 = component (or hole) until shown to be outside
        for y in y0...y1 { for x in x0...x1 where labels[y * W + x] == bestLabel { filled[(y - y0 + 1) * bw + (x - x0 + 1)] = 2 } }
        // flood the outside (value 0) from the padding corner through non-component cells
        stack.removeAll(keepingCapacity: true)
        filled[0] = 0; stack.append(0)
        while let p = stack.popLast() {
            let x = p % bw, y = p / bw
            if x > 0, filled[p - 1] == 1 { filled[p - 1] = 0; stack.append(p - 1) }
            if x < bw - 1, filled[p + 1] == 1 { filled[p + 1] = 0; stack.append(p + 1) }
            if y > 0, filled[p - bw] == 1 { filled[p - bw] = 0; stack.append(p - bw) }
            if y < bh - 1, filled[p + bw] == 1 { filled[p + bw] = 0; stack.append(p + bw) }
        }
        // `filled` != 0 is the hole-filled component on the padded grid.
        let ring = traceOuterBoundary(filled, width: bw, height: bh)
        let ox = Double(x0 - 1), oy = Double(y0 - 1)
        return ring.map { CGPoint(x: $0.x + ox, y: $0.y + oy) }
    }

    /// Crack-following trace of the outer boundary of the (single, hole-free) set `grid != 0`. The grid must have a
    /// background border. Returns the midpoints of the boundary pixel edges with collinear runs dropped.
    private static func traceOuterBoundary(_ grid: [UInt8], width W: Int, height H: Int) -> [CGPoint] {
        @inline(__always) func fg(_ x: Int, _ y: Int) -> Bool { x >= 0 && y >= 0 && x < W && y < H && grid[y * W + x] != 0 }
        // first set pixel in raster order: its top edge runs (i+1, j) -> (i, j), direction -x
        var sx = -1, sy = -1
        outer: for y in 0..<H { for x in 0..<W where grid[y * W + x] != 0 { sx = x; sy = y; break outer } }
        guard sx >= 0 else { return [] }
        let startV = (sx + 1, sy), startD = (-1, 0)
        var v = startV, d = startD
        var mids = [CGPoint]()
        var guardCount = 0
        let limit = 4 * W * H + 8
        repeat {
            // edge from v along d (unit length); record its midpoint and step
            mids.append(CGPoint(x: Double(v.0) + 0.5 * Double(d.0), y: Double(v.1) + 0.5 * Double(d.1)))
            v = (v.0 + d.0, v.1 + d.1)
            // outgoing edges at vertex v; TL=(vx-1,vy-1) TR=(vx,vy-1) BL=(vx-1,vy) BR=(vx,vy)
            let tl = fg(v.0 - 1, v.1 - 1), tr = fg(v.0, v.1 - 1), bl = fg(v.0 - 1, v.1), br = fg(v.0, v.1)
            var outs = [(Int, Int)]()
            if tr && !br { outs.append((1, 0)) }
            if bl && !tl { outs.append((-1, 0)) }
            if br && !bl { outs.append((0, 1)) }
            if tl && !tr { outs.append((0, -1)) }
            if outs.count == 1 {
                d = outs[0]
            } else if outs.count == 2 {
                // saddle: 8-connectivity keeps the diagonal pixels together = the RIGHT turn (y-down screen)
                let right = (-d.1, d.0)
                d = outs.contains(where: { $0 == right }) ? right : outs[0]
            } else {
                break  // cannot happen on a valid boundary
            }
            guardCount += 1
        } while !(v == startV && d == startD) && guardCount < limit
        return dropCollinear(mids)
    }

    private static func dropCollinear(_ p: [CGPoint]) -> [CGPoint] {
        let n = p.count
        guard n > 3 else { return p }
        var out = [CGPoint]()
        out.reserveCapacity(n / 2)
        for i in 0..<n {
            let a = p[(i + n - 1) % n], b = p[i], c = p[(i + 1) % n]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if abs(cross) > 1e-9 { out.append(b) }
        }
        return out.count >= 3 ? out : p
    }

    // MARK: - Ring geometry

    /// Douglas-Peucker on a closed ring. Splits at the vertex farthest from vertex 0, so the ring keeps its start
    /// point and its orientation.
    public static func simplify(_ ring: [CGPoint], tolerance: Double) -> [CGPoint] {
        let n = ring.count
        guard n > 3, tolerance > 0 else { return ring }
        var far = 0, best = -1.0
        for i in 1..<n {
            let dd = dist2(ring[i], ring[0])
            if dd > best { best = dd; far = i }
        }
        var keep = [Bool](repeating: false, count: n)
        keep[0] = true; keep[far] = true
        let tol2 = tolerance * tolerance
        // closed: segments 0 -> far and far -> n (== 0)
        var stack: [(Int, Int)] = [(0, far), (far, n)]
        while let (a, b) = stack.popLast() {
            guard b - a > 1 else { continue }
            let pa = ring[a], pb = ring[b % n]
            var worst = -1.0, wi = -1
            for i in (a + 1)..<b {
                let dd = segDist2(ring[i], pa, pb)
                if dd > worst { worst = dd; wi = i }
            }
            if worst > tol2 { keep[wi] = true; stack.append((a, wi)); stack.append((wi, b)) }
        }
        var out = [CGPoint]()
        for i in 0..<n where keep[i] { out.append(ring[i]) }
        return out.count >= 3 ? out : ring  // a degenerate sliver is returned as it was
    }

    /// Resample the closed ring every `step` px along its length, then a circular Gaussian (sigma px) along it.
    public static func smooth(_ ring: [CGPoint], sigma: Double, step: Double) -> [CGPoint] {
        guard ring.count >= 3, step > 0 else { return ring }
        let pts = resample(ring, step: step)
        let n = pts.count
        guard sigma > 0, n >= 3 else { return pts }
        let s = sigma / step
        let r = min(n / 2, max(1, Int((3 * s).rounded(.up))))
        var w = [Double](repeating: 0, count: 2 * r + 1)
        for k in -r...r { w[k + r] = exp(-0.5 * Double(k * k) / (s * s)) }
        let wsum = w.reduce(0, +)
        var out = [CGPoint](repeating: .zero, count: n)
        for i in 0..<n {
            var x = 0.0, y = 0.0
            for k in -r...r {
                let p = pts[((i + k) % n + n) % n]
                x += w[k + r] * p.x; y += w[k + r] * p.y
            }
            out[i] = CGPoint(x: x / wsum, y: y / wsum)
        }
        return out
    }

    /// Uniform resampling of a closed ring, `step` px apart (the count is rounded so the spacing is even).
    static func resample(_ ring: [CGPoint], step: Double) -> [CGPoint] {
        let n = ring.count
        guard n >= 2 else { return ring }
        var cum = [Double](repeating: 0, count: n + 1)
        for i in 0..<n { cum[i + 1] = cum[i] + hypot(ring[(i + 1) % n].x - ring[i].x, ring[(i + 1) % n].y - ring[i].y) }
        let L = cum[n]
        guard L > 0 else { return ring }
        let m = max(3, Int((L / step).rounded()))
        let ds = L / Double(m)
        var out = [CGPoint]()
        out.reserveCapacity(m)
        var seg = 0
        for k in 0..<m {
            let s = Double(k) * ds
            while seg < n - 1 && cum[seg + 1] < s { seg += 1 }
            let a = ring[seg], b = ring[(seg + 1) % n]
            let len = cum[seg + 1] - cum[seg]
            let t = len > 0 ? (s - cum[seg]) / len : 0
            out.append(CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t))
        }
        return out
    }

    /// Scanline even-odd fill; a pixel is set when its centre (i + 0.5, j + 0.5) is inside.
    public static func rasterize(_ ring: [CGPoint], width: Int, height: Int) -> BitMask {
        var mask = BitMask(width: width, height: height)
        let n = ring.count
        guard n >= 3, width > 0, height > 0 else { return mask }
        var rows = [[Double]](repeating: [], count: height)
        for i in 0..<n {
            let a = ring[i], b = ring[(i + 1) % n]
            if a.y == b.y { continue }
            let lo = min(a.y, b.y), hi = max(a.y, b.y)
            // rows whose centre yc = j + 0.5 satisfies lo <= yc < hi
            let j0 = max(0, Int((lo - 0.5).rounded(.up)))
            let j1 = min(height - 1, Int((hi - 0.5).rounded(.up)) - 1)
            if j0 > j1 { continue }
            let inv = (b.x - a.x) / (b.y - a.y)
            for j in j0...j1 {
                let yc = Double(j) + 0.5
                rows[j].append(a.x + (yc - a.y) * inv)
            }
        }
        mask.data.withUnsafeMutableBufferPointer { m in
            for j in 0..<height where rows[j].count >= 2 {
                let xs = rows[j].sorted()
                var k = 0
                while k + 1 < xs.count {
                    // pixel centres i + 0.5 in [xa, xb)
                    let i0 = max(0, Int((xs[k] - 0.5).rounded(.up)))
                    let i1 = min(width - 1, Int((xs[k + 1] - 0.5).rounded(.up)) - 1)
                    if i0 <= i1 { for i in i0...i1 { m[j * width + i] = 1 } }
                    k += 2
                }
            }
        }
        return mask
    }

    /// THE RETRACE. The ring vertices nearest the stroke's first and last points cut the ring into two arcs; the arc
    /// closer to the stroke (smaller mean distance) is replaced by the stroke. The ring's orientation is kept. A
    /// self-intersecting result is repaired by rasterize + largestComponentPolygon.
    /// The ring is densified to <= 1 px spacing first so a coarse (simplified) ring is cut where the stroke starts,
    /// not at the nearest corner; the extra points are dropped again on output.
    public static func replaceRun(in ring: [CGPoint], with stroke: [CGPoint]) -> [CGPoint] {
        guard ring.count >= 3, let first = stroke.first, let last = stroke.last else { return ring }
        let dense = densify(ring, maxStep: 1.0)
        let n = dense.count
        func nearest(_ p: CGPoint) -> Int {
            var bi = 0, bd = Double.infinity
            for i in 0..<n { let dd = dist2(dense[i], p); if dd < bd { bd = dd; bi = i } }
            return bi
        }
        let i0 = nearest(first), i1 = nearest(last)
        var result: [CGPoint]
        if i0 == i1 {
            // closed-ish stroke starting and ending at one place: splice it in there
            if polylineLength(stroke) < 2 { return ring }
            result = Array(dense[(i0 + 1)...] + dense[...i0]) + stroke
        } else {
            // arc A: i0 -> i1 forward (interior points); arc B: i1 -> i0 forward
            func arc(_ a: Int, _ b: Int) -> [CGPoint] {
                var out = [CGPoint](); var i = (a + 1) % n
                while i != b { out.append(dense[i]); i = (i + 1) % n }
                return out
            }
            let arcA = arc(i0, i1), arcB = arc(i1, i0)
            func meanDist(_ pts: [CGPoint]) -> Double {
                if pts.isEmpty { return 0 }
                var s = 0.0
                for p in pts { s += sqrt(polylineDist2(p, stroke)) }
                return s / Double(pts.count)
            }
            if meanDist(arcA) <= meanDist(arcB) {
                // keep B (i1 -> i0 forward), then i0 -> i1 via the stroke as drawn
                result = [dense[i1]] + arcB + [dense[i0]] + stroke
            } else {
                // keep A (i0 -> i1 forward), then i1 -> i0 via the reversed stroke
                result = [dense[i0]] + arcA + [dense[i1]] + stroke.reversed()
            }
        }
        result = dedupe(result)
        result = simplify(result, tolerance: 0.05)
        if result.count < 3 { return ring }
        if isSelfIntersecting(result) {
            result = repairRing(result, orientationOf: ring) ?? result
        } else if signedArea(result) * signedArea(ring) < 0 {
            result.reverse()
        }
        return result
    }

    /// All pixels whose centre is within `radius` of the polyline.
    public static func corridorMask(stroke: [CGPoint], radius: Double, width: Int, height: Int) -> BitMask {
        var mask = BitMask(width: width, height: height)
        guard !stroke.isEmpty, width > 0, height > 0, radius >= 0 else { return mask }
        let r2 = radius * radius
        let segs: [(CGPoint, CGPoint)] = stroke.count == 1 ? [(stroke[0], stroke[0])] : (0..<(stroke.count - 1)).map { (stroke[$0], stroke[$0 + 1]) }
        mask.data.withUnsafeMutableBufferPointer { m in
            for (a, b) in segs {
                let x0 = max(0, Int(floor(min(a.x, b.x) - radius - 0.5))), x1 = min(width - 1, Int(ceil(max(a.x, b.x) + radius - 0.5)))
                let y0 = max(0, Int(floor(min(a.y, b.y) - radius - 0.5))), y1 = min(height - 1, Int(ceil(max(a.y, b.y) + radius - 0.5)))
                if x0 > x1 || y0 > y1 { continue }
                for y in y0...y1 {
                    for x in x0...x1 where m[y * width + x] == 0 {
                        if segDist2(CGPoint(x: Double(x) + 0.5, y: Double(y) + 0.5), a, b) <= r2 { m[y * width + x] = 1 }
                    }
                }
            }
        }
        return mask
    }

    /// `old` outside the corridor, `new` inside it. All three must be the same size.
    public static func merge(old: BitMask, new: BitMask, corridor: BitMask) -> BitMask {
        precondition(old.width == new.width && old.height == new.height && old.width == corridor.width && old.height == corridor.height,
                     "merge: masks differ in size")
        var out = old
        for i in 0..<out.data.count where corridor.data[i] != 0 { out.data[i] = new.data[i] != 0 ? 1 : 0 }
        return out
    }

    // MARK: - Edges

    /// Sobel gradient magnitude (borders replicated).
    public static func gradientMagnitude(gray: [UInt8], width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        guard width > 0, height > 0, gray.count >= width * height else { return out }
        gray.withUnsafeBufferPointer { g in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<height {
                    let ym = max(0, y - 1) * width, y0 = y * width, yp = min(height - 1, y + 1) * width
                    for x in 0..<width {
                        let xm = max(0, x - 1), xp = min(width - 1, x + 1)
                        let a = Float(g[ym + xm]), b = Float(g[ym + x]), c = Float(g[ym + xp])
                        let d = Float(g[y0 + xm]), f = Float(g[y0 + xp])
                        let h = Float(g[yp + xm]), i = Float(g[yp + x]), j = Float(g[yp + xp])
                        let gx = (c + 2 * f + j) - (a + 2 * d + h)
                        let gy = (h + 2 * i + j) - (a + 2 * b + c)
                        o[y0 + x] = (gx * gx + gy * gy).squareRoot()
                    }
                }
            }
        }
        return out
    }

    /// Move each vertex along its outward normal to the strongest gradient within +-maxShift px when that edge is
    /// clearly (>= 2x) stronger than the gradient at the vertex; shifts are then smoothed along the ring (vertices
    /// that found no edge count as 0, so the ring tapers instead of stepping). Vertices are not added: pass a
    /// resampled ring (`smooth`) when the edges between sparse vertices matter.
    public static func snapToEdges(_ ring: [CGPoint], gradientMagnitude: [Float], width: Int, height: Int, maxShift: Double) -> [CGPoint] {
        let n = ring.count
        guard n >= 3, maxShift > 0, gradientMagnitude.count >= width * height, width > 1, height > 1 else { return ring }
        let outwardSign: Double = signedArea(ring) < 0 ? 1 : -1
        func sample(_ x: Double, _ y: Double) -> Float {
            // pixel centres at +0.5
            let fx = x - 0.5, fy = y - 0.5
            if fx < 0 || fy < 0 || fx > Double(width - 1) || fy > Double(height - 1) { return 0 }
            let ix = min(width - 2, Int(fx)), iy = min(height - 2, Int(fy))
            let tx = Float(fx - Double(ix)), ty = Float(fy - Double(iy))
            let i = iy * width + ix
            let a = gradientMagnitude[i], b = gradientMagnitude[i + 1], c = gradientMagnitude[i + width], d = gradientMagnitude[i + width + 1]
            return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
        }
        let stepPx = 0.25
        let ks = Int((maxShift / stepPx).rounded(.down))
        var shifts = [Double](repeating: 0, count: n)
        var normals = [SIMD2<Double>](repeating: .zero, count: n)
        for i in 0..<n {
            let a = ring[(i + n - 1) % n], b = ring[(i + 1) % n]
            var t = SIMD2<Double>(b.x - a.x, b.y - a.y)
            let len = simd_length(t)
            if len < 1e-9 { continue }
            t /= len
            // for a negative-area (y-down) ring the outward normal is (-ty, tx)
            let nrm = SIMD2<Double>(-t.y, t.x) * outwardSign
            normals[i] = nrm
            let p = ring[i]
            var vals = [Float](repeating: 0, count: 2 * ks + 1)
            var best: Float = 0, bestK = 0
            for k in -ks...ks {
                let o = Double(k) * stepPx
                let v = sample(p.x + nrm.x * o, p.y + nrm.y * o)
                vals[k + ks] = v
                if v > best || (v == best && abs(k) < abs(bestK)) { best = v; bestK = k }
            }
            let here = vals[ks]
            guard best > 1e-3, best >= 2 * here else { continue }
            // centre of the plateau around the peak (a Sobel step reads equal on both pixels beside the edge)
            var lo = bestK, hi = bestK
            while lo - 1 >= -ks && vals[lo - 1 + ks] >= best * 0.99 { lo -= 1 }
            while hi + 1 <= ks && vals[hi + 1 + ks] >= best * 0.99 { hi += 1 }
            shifts[i] = Double(lo + hi) * 0.5 * stepPx
        }
        // smooth the shifts along the ring (sigma 2 vertices)
        let r = min(n / 2, 6)
        var w = [Double](repeating: 0, count: 2 * r + 1)
        for k in -r...r { w[k + r] = exp(-0.5 * Double(k * k) / 4.0) }
        let ws = w.reduce(0, +)
        var out = ring
        for i in 0..<n {
            var s = 0.0
            for k in -r...r { s += w[k + r] * shifts[((i + k) % n + n) % n] }
            s /= ws
            out[i] = CGPoint(x: ring[i].x + normals[i].x * s, y: ring[i].y + normals[i].y * s)
        }
        return out
    }

    // MARK: - Depth -> plane -> millimetres

    /// RANSAC plane through the valid depth pixels (minus `exclude`), 8 mm inlier band, least-squares refit on the
    /// inliers. nil unless >= 200 points were usable and >= 30 % of them are inliers. Deterministic (seeded).
    public static func fitPlane(depth: [Float], width: Int, height: Int, intrinsics K: CameraIntrinsics, exclude: BitMask?) -> Plane? {
        guard width > 0, height > 0, depth.count >= width * height else { return nil }
        let useEx = exclude.map { $0.width == width && $0.height == height } ?? false
        // subsample to ~8000 candidate pixels
        let stride = max(1, Int(sqrt(Double(width * height) / 8000.0)))
        var pts = [SIMD3<Double>]()
        pts.reserveCapacity((width / stride + 1) * (height / stride + 1))
        var y = 0
        while y < height {
            var x = 0
            while x < width {
                let z = Double(depth[y * width + x])
                if z.isFinite, z > 0, !(useEx && exclude![x, y] != 0) {
                    pts.append(SIMD3((Double(x) - K.cx) / K.fx * z, (Double(y) - K.cy) / K.fy * z, z))
                }
                x += stride
            }
            y += stride
        }
        let n = pts.count
        guard n >= 200 else { return nil }
        let tol = 0.008
        var rng = SplitMix(seed: 0x5EED_CAFE)
        var bestN = SIMD3<Double>(0, 0, -1), bestD = 0.0, bestCount = -1
        for _ in 0..<300 {
            let a = pts[rng.next(n)], b = pts[rng.next(n)], c = pts[rng.next(n)]
            var nn = simd_cross(b - a, c - a)
            let l = simd_length(nn)
            if l < 1e-9 { continue }
            nn /= l
            let d = simd_dot(nn, a)
            var count = 0
            for p in pts where abs(simd_dot(nn, p) - d) < tol { count += 1 }
            if count > bestCount { bestCount = count; bestN = nn; bestD = d }
        }
        guard bestCount > 0 else { return nil }
        // two rounds of least squares on the inliers
        var nrm = bestN, dd = bestD, inliers = bestCount
        for _ in 0..<2 {
            var sel = [SIMD3<Double>]()
            sel.reserveCapacity(inliers)
            for p in pts where abs(simd_dot(nrm, p) - dd) < tol { sel.append(p) }
            guard sel.count >= 3, let fit = leastSquaresPlane(sel) else { break }
            nrm = fit.0; dd = fit.1
            inliers = 0
            for p in pts where abs(simd_dot(nrm, p) - dd) < tol { inliers += 1 }
        }
        let frac = Double(inliers) / Double(n)
        guard frac >= 0.3, inliers >= 200 else { return nil }
        return Plane(normal: nrm, d: dd, inlierFraction: frac)
    }

    /// Ray through each PHOTO pixel intersected with the plane raised `raisedByMm` toward the camera (the tool's top
    /// face), expressed in the plane's own basis in millimetres: e1 = camera +x projected onto the plane, e2 = normal x e1
    /// signed so image-down is +y; translated so min x = min y = 0. mmPerPx = the size of one photo pixel on that
    /// raised plane at the ring's centroid (square root of the projected pixel's area).
    public static func projectToPlane(_ ring: [CGPoint], intrinsics K: CameraIntrinsics, plane: Plane, raisedByMm: Double) -> (ringMm: [CGPoint], mmPerPx: Double) {
        guard !ring.isEmpty else { return ([], 0) }
        let n = simd_normalize(plane.normal)
        // the plane as given points toward the camera with d = n.P; moving toward the camera by h adds h to d
        let sign: Double = n.z < 0 ? 1 : -1
        let nn = n * sign, d = plane.d * sign + raisedByMm / 1000.0
        var e1 = SIMD3<Double>(1, 0, 0) - simd_dot(SIMD3<Double>(1, 0, 0), nn) * nn
        if simd_length(e1) < 1e-9 { e1 = SIMD3<Double>(0, 0, 1) - nn.z * nn }
        e1 = simd_normalize(e1)
        var e2 = simd_normalize(simd_cross(nn, e1))
        if simd_dot(e2, SIMD3<Double>(0, 1, 0)) < 0 { e2 = -e2 }
        func hit(_ x: Double, _ y: Double) -> SIMD3<Double>? {
            let r = SIMD3<Double>((x - 0.5 - K.cx) / K.fx, (y - 0.5 - K.cy) / K.fy, 1)
            let den = simd_dot(nn, r)
            if abs(den) < 1e-12 { return nil }
            let t = d / den
            return t > 0 ? r * t : nil
        }
        var uv = [SIMD2<Double>]()
        uv.reserveCapacity(ring.count)
        for p in ring {
            guard let P = hit(p.x, p.y) else { continue }
            uv.append(SIMD2(simd_dot(P, e1), simd_dot(P, e2)) * 1000)
        }
        guard !uv.isEmpty else { return ([], 0) }
        let mx = uv.map(\.x).min()!, my = uv.map(\.y).min()!
        let out = uv.map { CGPoint(x: $0.x - mx, y: $0.y - my) }
        let c = centroid(ring)
        var mmPerPx = 0.0
        if let p0 = hit(c.x, c.y), let px = hit(c.x + 1, c.y), let py = hit(c.x, c.y + 1) {
            mmPerPx = sqrt(simd_length(simd_cross(px - p0, py - p0))) * 1000
        }
        return (out, mmPerPx)
    }

    // MARK: - Measures

    /// Shoelace area, always positive.
    public static func area(_ ring: [CGPoint]) -> Double { abs(signedArea(ring)) }

    public static func boundingBox(_ ring: [CGPoint]) -> CGRect {
        guard let f = ring.first else { return .null }
        var x0 = f.x, x1 = f.x, y0 = f.y, y1 = f.y
        for p in ring { x0 = min(x0, p.x); x1 = max(x1, p.x); y0 = min(y0, p.y); y1 = max(y1, p.y) }
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    /// Signed shoelace area in the ring's own coordinates (negative for `largestComponentPolygon` output).
    public static func signedArea(_ ring: [CGPoint]) -> Double {
        let n = ring.count
        guard n >= 3 else { return 0 }
        var s = 0.0
        for i in 0..<n { let a = ring[i], b = ring[(i + 1) % n]; s += a.x * b.y - b.x * a.y }
        return s / 2
    }

    /// True when two non-adjacent edges of the closed ring cross or touch.
    public static func isSelfIntersecting(_ ring: [CGPoint]) -> Bool {
        let n = ring.count
        guard n >= 4 else { return false }
        // sweep over edges sorted by min x
        struct E { var i: Int; var x0: Double; var x1: Double; var y0: Double; var y1: Double }
        var es = (0..<n).map { i -> E in
            let a = ring[i], b = ring[(i + 1) % n]
            return E(i: i, x0: min(a.x, b.x), x1: max(a.x, b.x), y0: min(a.y, b.y), y1: max(a.y, b.y))
        }
        es.sort { $0.x0 < $1.x0 }
        var active = [E]()
        for e in es {
            active.removeAll { $0.x1 < e.x0 }
            for f in active {
                if f.y1 < e.y0 || e.y1 < f.y0 { continue }
                let di = abs(e.i - f.i)
                if di <= 1 || di == n - 1 { continue }  // adjacent edges share a vertex
                if segmentsIntersect(ring[e.i], ring[(e.i + 1) % n], ring[f.i], ring[(f.i + 1) % n]) { return true }
            }
            active.append(e)
        }
        return false
    }

    // MARK: - Helpers

    @inline(__always) static func dist2(_ a: CGPoint, _ b: CGPoint) -> Double { let dx = a.x - b.x, dy = a.y - b.y; return dx * dx + dy * dy }

    @inline(__always) static func segDist2(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> Double {
        let vx = b.x - a.x, vy = b.y - a.y
        let l2 = vx * vx + vy * vy
        if l2 < 1e-18 { return dist2(p, a) }
        let t = max(0, min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / l2))
        let qx = a.x + t * vx - p.x, qy = a.y + t * vy - p.y
        return qx * qx + qy * qy
    }

    static func polylineDist2(_ p: CGPoint, _ line: [CGPoint]) -> Double {
        if line.count == 1 { return dist2(p, line[0]) }
        var best = Double.infinity
        for i in 0..<(line.count - 1) { best = min(best, segDist2(p, line[i], line[i + 1])) }
        return best
    }

    static func polylineLength(_ line: [CGPoint]) -> Double {
        guard line.count > 1 else { return 0 }
        var s = 0.0
        for i in 1..<line.count { s += hypot(line[i].x - line[i - 1].x, line[i].y - line[i - 1].y) }
        return s
    }

    /// Insert points so no edge of the closed ring is longer than maxStep; original vertices are kept.
    static func densify(_ ring: [CGPoint], maxStep: Double) -> [CGPoint] {
        var out = [CGPoint]()
        let n = ring.count
        for i in 0..<n {
            let a = ring[i], b = ring[(i + 1) % n]
            out.append(a)
            let k = Int((hypot(b.x - a.x, b.y - a.y) / maxStep).rounded(.up))
            if k > 1 { for j in 1..<k { let t = Double(j) / Double(k); out.append(CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)) } }
        }
        return out
    }

    static func dedupe(_ ring: [CGPoint]) -> [CGPoint] {
        var out = [CGPoint]()
        for p in ring where out.last.map({ dist2($0, p) > 1e-12 }) ?? true { out.append(p) }
        while out.count > 1, dist2(out[0], out[out.count - 1]) <= 1e-12 { out.removeLast() }
        return out
    }

    static func centroid(_ ring: [CGPoint]) -> CGPoint {
        let a = signedArea(ring)
        let n = ring.count
        if abs(a) < 1e-9 || n < 3 {
            let sx = ring.reduce(0) { $0 + $1.x }, sy = ring.reduce(0) { $0 + $1.y }
            return CGPoint(x: sx / Double(max(1, n)), y: sy / Double(max(1, n)))
        }
        var cx = 0.0, cy = 0.0
        for i in 0..<n {
            let p = ring[i], q = ring[(i + 1) % n]
            let c = p.x * q.y - q.x * p.y
            cx += (p.x + q.x) * c; cy += (p.y + q.y) * c
        }
        return CGPoint(x: cx / (6 * a), y: cy / (6 * a))
    }

    static func segmentsIntersect(_ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, _ p4: CGPoint) -> Bool {
        func orient(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double { (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) }
        func onSeg(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Bool {
            min(a.x, b.x) - 1e-12 <= c.x && c.x <= max(a.x, b.x) + 1e-12 && min(a.y, b.y) - 1e-12 <= c.y && c.y <= max(a.y, b.y) + 1e-12
        }
        let d1 = orient(p3, p4, p1), d2 = orient(p3, p4, p2), d3 = orient(p1, p2, p3), d4 = orient(p1, p2, p4)
        if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) { return true }
        let eps = 1e-12
        if abs(d1) < eps && onSeg(p3, p4, p1) { return true }
        if abs(d2) < eps && onSeg(p3, p4, p2) { return true }
        if abs(d3) < eps && onSeg(p1, p2, p3) { return true }
        if abs(d4) < eps && onSeg(p1, p2, p4) { return true }
        return false
    }

    /// Rasterise a self-intersecting ring around its own box (scaled up when small so the repair is sub-pixel) and
    /// trace the largest component back out, in the orientation of `reference`.
    static func repairRing(_ ring: [CGPoint], orientationOf reference: [CGPoint]) -> [CGPoint]? {
        let bb = boundingBox(ring)
        guard bb.width > 0, bb.height > 0 else { return nil }
        let longSide = max(bb.width, bb.height)
        let scale = max(1.0, min(8.0, 800.0 / longSide))
        let pad = 2.0
        let W = Int(ceil(bb.width * scale + 2 * pad)), H = Int(ceil(bb.height * scale + 2 * pad))
        let local = ring.map { CGPoint(x: ($0.x - bb.minX) * scale + pad, y: ($0.y - bb.minY) * scale + pad) }
        let m = rasterize(local, width: W, height: H)
        guard let traced = largestComponentPolygon(mask: m, minArea: 1) else { return nil }
        var out = traced.map { CGPoint(x: ($0.x - pad) / scale + bb.minX, y: ($0.y - pad) / scale + bb.minY) }
        out = simplify(out, tolerance: 0.3)
        if signedArea(out) * signedArea(reference) < 0 { out.reverse() }
        return out
    }

    /// Least-squares plane (smallest-eigenvalue eigenvector of the covariance) through the points.
    static func leastSquaresPlane(_ pts: [SIMD3<Double>]) -> (SIMD3<Double>, Double)? {
        guard pts.count >= 3 else { return nil }
        var c = SIMD3<Double>(0, 0, 0)
        for p in pts { c += p }
        c /= Double(pts.count)
        var A = simd_double3x3(0)
        for p in pts {
            let q = p - c
            A += simd_double3x3(rows: [q * q.x, q * q.y, q * q.z])
        }
        let n = smallestEigenvector(A)
        guard simd_length(n) > 0.5 else { return nil }
        return (n, simd_dot(n, c))
    }

    /// Jacobi eigen-decomposition of a symmetric 3x3; returns the unit eigenvector of the smallest eigenvalue.
    static func smallestEigenvector(_ M: simd_double3x3) -> SIMD3<Double> {
        var a = [[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]]
        var v = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        for _ in 0..<50 {
            var p = 0, q = 1
            var off = abs(a[0][1])
            if abs(a[0][2]) > off { off = abs(a[0][2]); p = 0; q = 2 }
            if abs(a[1][2]) > off { off = abs(a[1][2]); p = 1; q = 2 }
            if off < 1e-18 { break }
            let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1))
            let cs = 1 / sqrt(t * t + 1), sn = t * cs
            for k in 0..<3 {
                let akp = a[k][p], akq = a[k][q]
                a[k][p] = cs * akp - sn * akq; a[k][q] = sn * akp + cs * akq
            }
            for k in 0..<3 {
                let apk = a[p][k], aqk = a[q][k]
                a[p][k] = cs * apk - sn * aqk; a[q][k] = sn * apk + cs * aqk
            }
            for k in 0..<3 {
                let vkp = v[k][p], vkq = v[k][q]
                v[k][p] = cs * vkp - sn * vkq; v[k][q] = sn * vkp + cs * vkq
            }
        }
        var mi = 0
        for i in 1..<3 where a[i][i] < a[mi][mi] { mi = i }
        return simd_normalize(SIMD3(v[0][mi], v[1][mi], v[2][mi]))
    }
}

/// Small deterministic RNG so RANSAC gives the same plane for the same depth map.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next(_ n: Int) -> Int {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Int(z % UInt64(n))
    }
}
