import Foundation
import CoreGraphics
import simd

// OutlineEngine.swift checked on the Mac (no device, no simulator): masks -> rings, the retrace, corridor merge,
// edge snapping, and depth -> plane -> millimetres, each against exact synthetic truth.
var failures = 0
func check(_ ok: Bool, _ msg: String) { print((ok ? "PASS " : "FAIL ") + msg); if !ok { failures += 1 } }
func f(_ v: Double, _ d: Int = 2) -> String { String(format: "%.\(d)f", v) }

// MARK: (a) rectangle mask
do {
    var m = BitMask(width: 400, height: 300)
    for y in 80..<180 { for x in 50..<250 { m[x, y] = 1 } }
    m[5, 5] = 1  // a speck: must not win
    let ring = OutlineEngine.largestComponentPolygon(mask: m, minArea: 10)!
    let a = OutlineEngine.area(ring)
    check(abs(a - 20000) / 20000 < 0.01, "(a) 200x100 rectangle area \(f(a)) vs 20000 (within 1 %)")
    let bb = OutlineEngine.boundingBox(ring)
    check(abs(bb.minX - 50) < 1e-9 && abs(bb.maxX - 250) < 1e-9 && abs(bb.minY - 80) < 1e-9 && abs(bb.maxY - 180) < 1e-9,
          "(a) bounding box \(bb) == (50,80) 200x100")
    let s = OutlineEngine.simplify(ring, tolerance: 1.5)
    check((4...8).contains(s.count), "(a) simplify(1.5): \(ring.count) -> \(s.count) vertices (4-8)")
    check(OutlineEngine.signedArea(ring) < 0, "(a) traced ring orientation: signed area < 0 (CCW on screen)")
    let back = OutlineEngine.rasterize(ring, width: 400, height: 300)
    var diff = 0
    for i in 0..<back.data.count where back.data[i] != m.data[i] { diff += 1 }
    check(diff == 1, "(a) rasterize(ring) reproduces the mask (\(diff) px differ; the speck is the 1)")
    check(OutlineEngine.largestComponentPolygon(mask: m, minArea: 20001) == nil, "(a) nil when nothing reaches minArea")
}

// MARK: (b) disc, with a hole and a diagonal-only (8-connected) appendage
do {
    let r = 80.0, c = 150.0
    var m = BitMask(width: 300, height: 300)
    for y in 0..<300 { for x in 0..<300 where hypot(Double(x) + 0.5 - c, Double(y) + 0.5 - c) < r { m[x, y] = 1 } }
    let ring = OutlineEngine.largestComponentPolygon(mask: m, minArea: 10)!
    let a = OutlineEngine.area(ring)
    check(abs(a - .pi * r * r) / (.pi * r * r) < 0.02, "(b) disc r=80 area \(f(a)) vs pi r^2 \(f(.pi * r * r)) (within 2 %)")
    // punch a hole: the outer contour must ignore it
    var h = m
    for y in 140..<160 { for x in 140..<160 { h[x, y] = 0 } }
    let rh = OutlineEngine.largestComponentPolygon(mask: h, minArea: 10)!
    check(abs(OutlineEngine.area(rh) - a) < 1e-6, "(b) a hole is filled: area \(f(OutlineEngine.area(rh)))")
    // diagonal chain of pixels leaving the disc: 8-connected, so part of the same component and of the outline
    var g = m
    for k in 0..<10 { g[230 + k, 150 + k] = 1 }
    let rg = OutlineEngine.largestComponentPolygon(mask: g, minArea: 10)!
    let bg = OutlineEngine.boundingBox(rg)
    check(bg.maxX >= 240 - 1e-9, "(b) 8-connected diagonal tail is traced (maxX \(f(bg.maxX)))")
    check(!OutlineEngine.isSelfIntersecting(rg), "(b) traced ring with a diagonal tail is simple")
}

// MARK: (c) the retrace
do {
    // square 100..200, as 4 vertices in the traced orientation (signed area < 0) and as the traced mask contour
    let sq4: [CGPoint] = [CGPoint(x: 100, y: 100), CGPoint(x: 100, y: 200), CGPoint(x: 200, y: 200), CGPoint(x: 200, y: 100)]
    var sm = BitMask(width: 300, height: 300)
    for y in 100..<200 { for x in 100..<200 { sm[x, y] = 1 } }
    let traced = OutlineEngine.largestComponentPolygon(mask: sm, minArea: 10)!
    // stroke bulging the RIGHT side out by 20 px between y = 110 and 190 (trapezoid 80 / 40 wide, 20 deep = 1200 px^2)
    var stroke = [CGPoint]()
    func addLine(_ a: CGPoint, _ b: CGPoint) {
        let k = Int(ceil(hypot(b.x - a.x, b.y - a.y) / 2))
        for j in 0..<k { let t = Double(j) / Double(k); stroke.append(CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)) }
    }
    addLine(CGPoint(x: 200, y: 110), CGPoint(x: 220, y: 130)); addLine(CGPoint(x: 220, y: 130), CGPoint(x: 220, y: 170))
    addLine(CGPoint(x: 220, y: 170), CGPoint(x: 200, y: 190)); stroke.append(CGPoint(x: 200, y: 190))
    for (name, ring) in [("4-vertex square", sq4), ("traced square", traced), ("reversed square", Array(sq4.reversed()))] {
        let a0 = OutlineEngine.area(ring)
        let fwd = OutlineEngine.replaceRun(in: ring, with: stroke)
        let rev = OutlineEngine.replaceRun(in: ring, with: Array(stroke.reversed()))
        let da = OutlineEngine.area(fwd) - a0, dr = OutlineEngine.area(rev) - a0
        check(abs(da - 1200) < 30, "(c) \(name): bulge adds \(f(da)) px^2 (expected ~1200)")
        check(abs(dr - da) < 1, "(c) \(name): reversed stroke gives the same area (\(f(dr)))")
        check(!OutlineEngine.isSelfIntersecting(fwd) && !OutlineEngine.isSelfIntersecting(rev), "(c) \(name): results are simple rings")
        check(OutlineEngine.signedArea(fwd) * OutlineEngine.signedArea(ring) > 0 && OutlineEngine.signedArea(rev) * OutlineEngine.signedArea(ring) > 0,
              "(c) \(name): orientation kept")
        let bb = OutlineEngine.boundingBox(fwd)
        check(abs(bb.maxX - 220) < 0.5 && abs(bb.minX - 100) < 0.5, "(c) \(name): bulge reaches x=220, left side untouched")
    }
    // an inward stroke (dent) on the top side
    let dent: [CGPoint] = (0...40).map { k in let t = Double(k) / 40; return CGPoint(x: 120 + 60 * t, y: 100 + 15 * sin(.pi * t)) }
    let dented = OutlineEngine.replaceRun(in: sq4, with: dent)
    let expectedDent = 60 * 15 * 2 / Double.pi
    check(abs((10000 - OutlineEngine.area(dented)) - expectedDent) < 20, "(c) dent removes \(f(10000 - OutlineEngine.area(dented))) px^2 (expected \(f(expectedDent)))")
    // a stroke that crosses back over the ring: must be repaired into a simple ring
    let crossing: [CGPoint] = [CGPoint(x: 200, y: 120), CGPoint(x: 230, y: 150), CGPoint(x: 185, y: 165), CGPoint(x: 230, y: 175), CGPoint(x: 200, y: 185)]
    let fixed = OutlineEngine.replaceRun(in: sq4, with: crossing)
    check(!OutlineEngine.isSelfIntersecting(fixed) && OutlineEngine.signedArea(fixed) < 0, "(c) self-crossing stroke repaired into a simple ring (\(fixed.count) pts)")
}

// MARK: (d) corridor merge
do {
    let W = 300, H = 200
    var old = BitMask(width: W, height: H), new = BitMask(width: W, height: H)
    for y in 0..<H { for x in 0..<W {
        if hypot(Double(x) - 120, Double(y) - 100) < 60 { old[x, y] = 1 }
        if hypot(Double(x) - 140, Double(y) - 100) < 60 { new[x, y] = 1 }
    } }
    let stroke: [CGPoint] = [CGPoint(x: 180, y: 60), CGPoint(x: 195, y: 100), CGPoint(x: 180, y: 140)]
    let cor = OutlineEngine.corridorMask(stroke: stroke, radius: 12, width: W, height: H)
    let merged = OutlineEngine.merge(old: old, new: new, corridor: cor)
    var outsideDiff = 0, insideMismatch = 0, changed = 0, corridorPx = 0
    for i in 0..<merged.data.count {
        if cor.data[i] == 0 { if merged.data[i] != old.data[i] { outsideDiff += 1 } }
        else { corridorPx += 1; if merged.data[i] != new.data[i] { insideMismatch += 1 }; if merged.data[i] != old.data[i] { changed += 1 } }
    }
    check(outsideDiff == 0 && insideMismatch == 0, "(d) merged == old outside the corridor, == new inside (\(outsideDiff), \(insideMismatch) mismatches)")
    check(changed > 100, "(d) the corridor actually changed \(changed) of its \(corridorPx) px")
    // corridor geometry: a pixel 11 px from the stroke is in, 13 px is out
    check(cor[180 + 11, 60] == 1 || cor[191, 60] == 1, "(d) corridor covers radius")
    check(cor[195 + 13, 100] == 0 && cor[195 + 11, 100] == 1, "(d) corridor edge at the radius (in at 11 px, out at 13 px)")
}

// MARK: (e) plane fit
let dW = 256, dH = 192
let depthK = CameraIntrinsics(fx: 210, fy: 210, cx: 128, cy: 96, width: dW, height: dH)
let tilt = 10.0 * .pi / 180
let trueN = SIMD3<Double>(0, sin(tilt), -cos(tilt))        // toward the camera
let trueD = simd_dot(trueN, SIMD3<Double>(0, 0, 0.40))       // passes 0.40 m out along the optical axis
var fitted: Plane? = nil
do {
    var seed: UInt64 = 42
    func uniform() -> Double { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Double(seed >> 11) / Double(1 << 53) }
    func gauss() -> Double { sqrt(-2 * log(max(1e-12, uniform()))) * cos(2 * .pi * uniform()) }
    var depth = [Float](repeating: 0, count: dW * dH)
    var ex = BitMask(width: dW, height: dH)
    for v in 0..<dH { for u in 0..<dW {
        let r = SIMD3<Double>((Double(u) - 128) / 210, (Double(v) - 96) / 210, 1)
        let z = trueD / simd_dot(trueN, r)   // z-depth (r.z == 1)
        depth[v * dW + u] = Float(z + 0.001 * gauss())
    } }
    for v in 40..<70 { for u in 30..<70 { depth[v * dW + u] = .nan } }               // 40x30 invalid
    for v in 100..<130 { for u in 150..<190 { depth[v * dW + u] -= 0.030; ex[u, v] = 1 } }  // 30 mm tool, excluded
    let t0 = Date()
    fitted = OutlineEngine.fitPlane(depth: depth, width: dW, height: dH, intrinsics: depthK, exclude: ex)
    let ms = Date().timeIntervalSince(t0) * 1000
    if let p = fitted {
        check(abs(p.tiltDeg - 10) < 0.5, "(e) tilt \(f(p.tiltDeg, 3)) deg vs 10")
        check(abs(p.distanceM - 0.40) < 0.003, "(e) distance \(f(p.distanceM * 1000, 2)) mm vs 400")
        check(p.inlierFraction > 0.9, "(e) inliers \(f(p.inlierFraction * 100, 1)) %")
        check(p.normal.z < 0 && abs(simd_length(p.normal) - 1) < 1e-9, "(e) unit normal toward the camera")
        print("     fitPlane took \(f(ms, 1)) ms")
    } else { check(false, "(e) fitPlane returned nil") }
    // the same tool NOT excluded is still rejected by RANSAC
    let p2 = OutlineEngine.fitPlane(depth: depth, width: dW, height: dH, intrinsics: depthK, exclude: nil)
    check(p2.map { abs($0.distanceM - 0.40) < 0.003 && $0.inlierFraction > 0.85 } ?? false, "(e) without the exclusion mask the tool is an outlier (\(f((p2?.inlierFraction ?? 0) * 100, 1)) % inliers)")
    check(OutlineEngine.fitPlane(depth: [Float](repeating: .nan, count: dW * dH), width: dW, height: dH, intrinsics: depthK, exclude: nil) == nil,
          "(e) nil on an empty depth map")
}

// MARK: (f) photo ring -> plane millimetres
do {
    let K = CameraIntrinsics(fx: 3000, fy: 3000, cx: 2016, cy: 1512, width: 4032, height: 3024)
    let raise = 12.0
    let n = trueN, dRaised = trueD + raise / 1000
    var e1 = SIMD3<Double>(1, 0, 0) - n.x * n; e1 = simd_normalize(e1)
    var e2 = simd_normalize(simd_cross(n, e1)); if e2.y < 0 { e2 = -e2 }
    let O = SIMD3<Double>(0, 0, dRaised / n.z) + 0.020 * e1 - 0.015 * e2   // a little off-centre
    var corners3: [SIMD3<Double>] = []
    for (a, b) in [(-50.0, -30.0), (-50.0, 30.0), (50.0, 30.0), (50.0, -30.0)] { corners3.append(O + (a / 1000) * e1 + (b / 1000) * e2) }
    var ring = [CGPoint]()
    for i in 0..<4 {
        let A = corners3[i], B = corners3[(i + 1) % 4]
        for k in 0..<40 {
            let P = A + (B - A) * (Double(k) / 40)
            ring.append(CGPoint(x: K.fx * P.x / P.z + K.cx + 0.5, y: K.fy * P.y / P.z + K.cy + 0.5))
        }
    }
    let truePlane = Plane(normal: trueN, d: trueD, inlierFraction: 1)
    let (mm, mpp) = OutlineEngine.projectToPlane(ring, intrinsics: K, plane: truePlane, raisedByMm: raise)
    let bb = OutlineEngine.boundingBox(mm)
    check(abs(bb.width - 100) < 0.5 && abs(bb.height - 60) < 0.5, "(f) true plane: \(f(bb.width, 3)) x \(f(bb.height, 3)) mm vs 100 x 60")
    check(abs(bb.minX) < 1e-9 && abs(bb.minY) < 1e-9, "(f) normalised to min x = min y = 0")
    let ratio = sqrt(OutlineEngine.area(mm) / OutlineEngine.area(ring))
    check(abs(mpp / ratio - 1) < 0.02, "(f) mmPerPx \(f(mpp, 4)) vs sqrt(area mm / area px) \(f(ratio, 4))")
    // orientation of the y axis: the corner first in the ring (image top-left side) must be at small y
    check(mm[0].y < 1 && mm[0].x < 1, "(f) image-down maps to +y (first corner at \(f(mm[0].x)), \(f(mm[0].y)))")
    if let p = fitted {
        let (mm2, _) = OutlineEngine.projectToPlane(ring, intrinsics: K, plane: p, raisedByMm: raise)
        let b2 = OutlineEngine.boundingBox(mm2)
        check(abs(b2.width - 100) < 1 && abs(b2.height - 60) < 1, "(f) fitted plane from (e): \(f(b2.width, 3)) x \(f(b2.height, 3)) mm (within 1 mm)")
    }
}

// MARK: (g) snap to edges
do {
    let W = 300, H = 300, R = 80.0
    var gray = [UInt8](repeating: 50, count: W * H)
    for y in 0..<H { for x in 0..<W where hypot(Double(x) + 0.5 - 150, Double(y) + 0.5 - 150) < R { gray[y * W + x] = 200 } }
    let grad = OutlineEngine.gradientMagnitude(gray: gray, width: W, height: H)
    func circle(_ r: Double, ccwOnScreen: Bool) -> [CGPoint] {
        let n = Int(2 * .pi * r / 2)
        let pts = (0..<n).map { k -> CGPoint in let t = 2 * .pi * Double(k) / Double(n); return CGPoint(x: 150 + r * cos(t), y: 150 + r * sin(t)) }
        return ccwOnScreen ? Array(pts.reversed()) : pts
    }
    for ccw in [true, false] {
        let inner = circle(R - 4, ccwOnScreen: ccw)
        let snapped = OutlineEngine.snapToEdges(inner, gradientMagnitude: grad, width: W, height: H, maxShift: 6)
        let errs = snapped.map { abs(hypot($0.x - 150, $0.y - 150) - R) }
        check(errs.max()! <= 1.0, "(g) ring 4 px inside (\(ccw ? "CCW" : "CW")) snapped onto the edge: max |r - 80| = \(f(errs.max()!, 3)), mean \(f(errs.reduce(0, +) / Double(errs.count), 3))")
        let outer = circle(R + 4, ccwOnScreen: ccw)
        let s2 = OutlineEngine.snapToEdges(outer, gradientMagnitude: grad, width: W, height: H, maxShift: 6)
        let e2 = s2.map { abs(hypot($0.x - 150, $0.y - 150) - R) }
        check(e2.max()! <= 1.0, "(g) ring 4 px outside (\(ccw ? "CCW" : "CW")) snapped inward: max |r - 80| = \(f(e2.max()!, 3))")
    }
    let far = circle(30, ccwOnScreen: true)
    let s3 = OutlineEngine.snapToEdges(far, gradientMagnitude: grad, width: W, height: H, maxShift: 6)
    let moved = zip(far, s3).map { hypot($0.x - $1.x, $0.y - $1.y) }.max()!
    check(moved < 1e-9, "(g) ring far from any edge left alone (max move \(moved))")
}

// MARK: smooth / simplify sanity
do {
    let sq: [CGPoint] = [CGPoint(x: 0, y: 0), CGPoint(x: 0, y: 100), CGPoint(x: 100, y: 100), CGPoint(x: 100, y: 0)]
    let s = OutlineEngine.smooth(sq, sigma: 0, step: 2)
    check(s.count == 200 && abs(OutlineEngine.area(s) - 10000) < 1e-6, "smooth(sigma 0) resamples every 2 px (\(s.count) pts)")
    let s2 = OutlineEngine.smooth(sq, sigma: 3, step: 1)
    check(OutlineEngine.area(s2) < 10000 && OutlineEngine.area(s2) > 9900 && OutlineEngine.signedArea(s2) < 0, "smooth(sigma 3) rounds corners slightly, keeps orientation (\(f(OutlineEngine.area(s2))))")
}

// MARK: speed at photo resolution (a SAM mask upsampled to a 12 MP still)
do {
    let W = 4032, H = 3024
    var m = BitMask(width: W, height: H)
    for y in 0..<H { for x in 0..<W where pow((Double(x) - 2016) / 1400, 2) + pow((Double(y) - 1512) / 600, 2) < 1 { m[x, y] = 1 } }
    var t = Date()
    let ring = OutlineEngine.largestComponentPolygon(mask: m, minArea: 100)!
    let tTrace = Date().timeIntervalSince(t) * 1000
    t = Date()
    let back = OutlineEngine.rasterize(ring, width: W, height: H)
    let tRaster = Date().timeIntervalSince(t) * 1000
    check(back.data == m.data && tTrace < 3000, "12 MP ellipse: trace \(f(tTrace, 0)) ms (\(ring.count) pts), rasterize \(f(tRaster, 0)) ms, round trip exact")
}

print(failures == 0 ? "ALL OUTLINE ENGINE CHECKS PASSED" : "\(failures) CHECK(S) FAILED")
exit(failures == 0 ? 0 : 1)
