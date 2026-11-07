import Foundation
import CoreGraphics

// OutlineCorrection.swift on the Mac: the retrace with a FAKE SAM. A 200 x 200 square tool whose true outline has a
// round bulge on its right side (disc r = 40 at (300, 200)); the person traces the bulge a little inside (r = 36).
// A good SAM answer must be adopted (the outline reaches the true r = 40), a wrong one rejected (the drawn r = 36
// stands), and nothing outside the stroke's corridor may move.
var failures = 0
func check(_ ok: Bool, _ msg: String) { print((ok ? "PASS " : "FAIL ") + msg); if !ok { failures += 1 } }
func f(_ v: Double) -> String { String(format: "%.2f", v) }

let W = 400, H = 400
func squareMask() -> BitMask {
    var m = BitMask(width: W, height: H)
    for y in 100..<300 { for x in 100..<300 { m[x, y] = 1 } }
    return m
}
func bulgeMask(radius: Double) -> BitMask {
    var m = squareMask()
    for y in 0..<H { for x in 0..<W where hypot(Double(x) + 0.5 - 300, Double(y) + 0.5 - 200) < radius { m[x, y] = 1 } }
    return m
}
// the person's stroke: the right half of a circle r = 36 about (300, 200), top to bottom, through x = 336
let stroke: [CGPoint] = stride(from: -90.0, through: 90.0, by: 3).map { a in
    let t = a * .pi / 180
    return CGPoint(x: 300 + 36 * cos(t), y: 200 + 36 * sin(t))
}
let state = OutlineCorrection.initialOutline(mask: squareMask(), minAreaPx: 100)!
let r = OutlineCorrection.corridorRadius(imageWidth: W, imageHeight: H)
check(abs(OutlineEngine.area(state.ring) - 40000) / 40000 < 0.01, "initial outline area \(f(OutlineEngine.area(state.ring))) ~ 40000, corridor r = \(f(r))")

func maxX(_ ring: [CGPoint]) -> Double { ring.map { Double($0.x) }.max()! }
func untouched(_ new: [CGPoint]) -> (Int, Double) {
    var n = 0, worst = 0.0
    // outside the corridor AND not on the old right edge between the stroke's ends (that run is what gets replaced)
    for p in state.ring where sqrt(OutlineEngine.polylineDist2(p, stroke)) > r + 2 && !(p.x > 290 && p.y > 150 && p.y < 250) {
        n += 1
        worst = max(worst, sqrt(OutlineEngine.polylineDist2(p, new + [new[0]])))
    }
    return (n, worst)
}

// prompts: foreground inside the corrected outline, background outside
do {
    let geo = OutlineEngine.replaceRun(in: state.ring, with: stroke)
    let geoMask = OutlineEngine.rasterize(geo, width: W, height: H)
    let corridor = OutlineEngine.corridorMask(stroke: stroke, radius: r, width: W, height: H)
    let pr = OutlineCorrection.samPrompts(stroke: stroke, insideMask: geoMask, oldMask: state.mask, corridor: corridor, radius: r)
    let fg = pr.filter { $0.label == 1 }, bg = pr.filter { $0.label == 0 }
    let fgIn = fg.allSatisfy { geoMask[Int($0.x), Int($0.y)] != 0 }, bgOut = bg.allSatisfy { geoMask[Int($0.x), Int($0.y)] == 0 }
    check(pr.count <= 14 && fg.count >= 5 && bg.count == 5, "prompts: \(fg.count) fg (5 stroke + anchors), \(bg.count) bg, \(pr.count) <= 14")
    check(fgIn && bgOut, "prompts: every fg inside the corrected outline, every bg outside")
    let anchors = fg.suffix(2)
    check(anchors.allSatisfy { corridor[Int($0.x), Int($0.y)] == 0 && state.mask[Int($0.x), Int($0.y)] != 0 },
          "prompts: anchors deep in the old mask, outside the corridor: \(anchors.map { "(\(Int($0.x)),\(Int($0.y)))" })")
}

// (1) SAM agrees with the stroke (true bulge r = 40): adopted
do {
    var calls = 0
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: W, imageHeight: H, gradient: nil,
                                                    sam: { _ in calls += 1; return bulgeMask(radius: 40) })
    let mx = maxX(res.state.ring)
    check(calls == 1 && res.method == .sam, "good SAM: called once, method \(res.method)")
    check(abs(mx - 340) < 1.5, "good SAM: outline reaches the TRUE bulge, max x \(f(mx)) ~ 340 (stroke at 336)")
    let (n, worst) = untouched(res.state.ring)
    check(worst < 1e-6, "good SAM: \(n) vertices outside the corridor unchanged (worst \(f(worst)) px)")
    check(res.state.mask == OutlineEngine.rasterize(res.state.ring, width: W, height: H), "good SAM: mask is the ring's raster")
}
// (1b) the same with a JITTERY stroke (+-0.3 r zig-zag every sample): still adopted (coverage, not length, decides)
do {
    let zig: [CGPoint] = stroke.enumerated().map { i, p in
        let d = (i % 2 == 0 ? 0.3 : -0.3) * r, dx = p.x - 300, dy = p.y - 200, l = hypot(dx, dy)
        return CGPoint(x: p.x + dx / l * d, y: p.y + dy / l * d)
    }
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: zig, imageWidth: W, imageHeight: H, gradient: nil,
                                                    sam: { _ in bulgeMask(radius: 40) })
    check(res.method == .sam && abs(maxX(res.state.ring) - 340) < 1.5, "jittery stroke + good SAM: adopted, max x \(f(maxX(res.state.ring)))")
}
// (2) SAM returns garbage far from the stroke: rejected, the drawn line stands
do {
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: W, imageHeight: H, gradient: nil, sam: { _ in
        var m = BitMask(width: W, height: H); for y in 0..<50 { for x in 0..<50 { m[x, y] = 1 } }; return m })
    let mx = maxX(res.state.ring)
    check(res.method == .geometric, "garbage SAM (block in the corner): rejected, method \(res.method)")
    check(abs(mx - 336) < 1.5, "garbage SAM: the stroke stands, max x \(f(mx)) ~ 336")
    let (n, worst) = untouched(res.state.ring)
    check(worst < 1e-6, "garbage SAM: \(n) vertices outside the corridor unchanged (worst \(f(worst)) px)")
}
// (3) SAM returns a much bigger bulge (r = 80) that fills the corridor: its edge is not the stroke -> rejected
do {
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: W, imageHeight: H, gradient: nil,
                                                    sam: { _ in bulgeMask(radius: 80) })
    check(res.method == .geometric && abs(maxX(res.state.ring) - 336) < 1.5, "over-grown SAM (r = 80): rejected, max x \(f(maxX(res.state.ring)))")
}
// (4) SAM throws: geometric
do {
    struct E: Error {}
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: W, imageHeight: H, gradient: nil, sam: { _ in throw E() })
    check(res.method == .geometric && abs(maxX(res.state.ring) - 336) < 1.5, "throwing SAM: geometric, max x \(f(maxX(res.state.ring)))")
}
// (5) no SAM, with a gradient: the line snaps to the photo's edge (bulge r = 40) inside the corridor only
do {
    let img = bulgeMask(radius: 40)
    let gray = img.data.map { $0 != 0 ? UInt8(200) : UInt8(40) }
    let grad = OutlineEngine.gradientMagnitude(gray: gray, width: W, height: H)
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: W, imageHeight: H, gradient: grad, sam: nil)
    let mx = maxX(res.state.ring)
    check(res.method == .snapped && mx > 337.5 && mx < 341.5, "no SAM + gradient: snapped toward the edge, max x \(f(mx)) (338-341)")
    let (n, worst) = untouched(res.state.ring)
    check(worst < 1e-6, "no SAM + gradient: \(n) vertices outside the corridor unchanged (worst \(f(worst)) px)")
}
// (6) an overshooting stroke (both ends 12 px INSIDE the tool, so the joins cross the old outline): the tails are cut
// out locally, the result is a simple ring, and the rest is untouched
do {
    let over: [CGPoint] = stride(from: -110.0, through: 110.0, by: 3).map { a in
        let t = a * .pi / 180
        return CGPoint(x: 300 + 36 * cos(t), y: 200 + 36 * sin(t))
    }
    let res = OutlineCorrection.applyStrokeDetailed(state: state, stroke: over, imageWidth: W, imageHeight: H, gradient: nil, sam: nil)
    let mx = maxX(res.state.ring)
    let a = OutlineEngine.area(res.state.ring)
    check(!OutlineEngine.isSelfIntersecting(res.state.ring) && abs(mx - 336) < 1.5, "overshooting stroke: simple ring, max x \(f(mx))")
    check(a > 40000 && a < 40000 + .pi * 36 * 36 / 2 + 300, "overshooting stroke: area \(f(a)) = square + half disc, no tails")
    var worst = 0.0
    for p in state.ring where p.x < 280 { worst = max(worst, sqrt(OutlineEngine.polylineDist2(p, res.state.ring + [res.state.ring[0]]))) }
    check(worst < 1e-6, "overshooting stroke: left part unchanged (worst \(f(worst)) px)")
}
// (7) a loop drawn with no outline at all becomes the outline
do {
    let loop: [CGPoint] = stride(from: 0.0, to: 360, by: 4).map { a in CGPoint(x: 200 + 60 * cos(a * .pi / 180), y: 200 + 60 * sin(a * .pi / 180)) }
    let s = OutlineCorrection.outlineFromLoop(loop, imageWidth: W, imageHeight: H, minAreaPx: 100)
    let a = s.map { OutlineEngine.area($0.ring) } ?? 0
    check(abs(a - .pi * 3600) / (.pi * 3600) < 0.03, "loop -> outline, area \(f(a)) ~ \(f(.pi * 3600))")
}

print(failures == 0 ? "ALL OUTLINE CORRECTION CHECKS PASSED" : "\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
