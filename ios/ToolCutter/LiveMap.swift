import simd
import Foundation

/// A live top-down height map built on the phone while the rear LiDAR pass is recorded, so you can see what
/// you have covered BEFORE uploading (the first depth pass ever taken missed half the drawer and nobody knew
/// until the Mac showed it). It is a PREVIEW: coverage and rough height on a 5 mm grid. The accurate model is
/// still built on the server from the marker-registered frames — nothing here feeds the outlines.
///
/// Kept free of ARKit/UIKit so `ios/tests/run_live_map_test.sh` can check the geometry on the Mac with swiftc:
/// unprojection conventions are exactly the kind of thing that comes out mirrored or upside down on a device.
///
/// Conventions (ARKit, right-handed, gravity-aligned world because the session uses `.gravity`):
///   camera space  +x right, +y up, -z forward (the camera looks along -z)
///   image pixels  u right, v down, intrinsics fx, fy, cx, cy in the depth map's own pixel units
///   world         y is UP; the drawer floor is a horizontal plane y = floorY; the map is on (x, z)
final class LiveHeightMap {
    /// Map cell size in metres (2.5 mm — a preview, but one that should look like the drawer).
    let cell: Float
    /// Map extent in cells, centred on `origin` (world x, z). 1.5 m square covers any drawer with slack.
    let size: Int
    private(set) var origin = SIMD2<Float>(0, 0)
    private var anchored = false
    /// Highest point seen per cell (metres above the floor), and how many samples landed there.
    private(set) var height: [Float]
    private(set) var count: [UInt16]
    /// Estimated floor height in world y (metres). Refined from the data: most of a drawer is floor.
    private(set) var floorY: Float = .nan
    private var floorSamples: [Float] = []
    private let floorSampleCap = 20_000

    init(cellMetres: Float = 0.0025, sizeCells: Int = 600) {      // 2.5 mm cells over 1.5 m: finer than the LiDAR's ~2 mm/px at 30 cm
        cell = cellMetres
        size = sizeCells
        height = Array(repeating: .nan, count: sizeCells * sizeCells)
        count = Array(repeating: 0, count: sizeCells * sizeCells)
    }

    var cellsCovered: Int { count.reduce(0) { $0 + ($1 > 0 ? 1 : 0) } }
    var hasFloor: Bool { !floorY.isNaN }

    func reset() {
        height = Array(repeating: .nan, count: size * size)
        count = Array(repeating: 0, count: size * size)
        floorY = .nan
        floorSamples.removeAll()
        anchored = false
    }

    /// Unproject a depth map into world points. `depth` is row-major, `w × h`, metres, 0/NaN = no return.
    /// `stride` subsamples the grid (every n-th pixel each way) — a 256×192 LiDAR map at stride 2 is ~12k points.
    static func worldPoints(depth: [Float], w: Int, h: Int, fx: Float, fy: Float, cx: Float, cy: Float,
                            transform: simd_float4x4, stride: Int = 2, maxDepth: Float = 1.5) -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        out.reserveCapacity((w / stride) * (h / stride))
        var v = 0
        while v < h {
            var u = 0
            while u < w {
                let d = depth[v * w + u]
                if d.isFinite && d > 0.05 && d < maxDepth {
                    // camera looks along -z; image v grows DOWN while camera y grows UP
                    let cam = SIMD4<Float>((Float(u) - cx) / fx * d, -(Float(v) - cy) / fy * d, -d, 1)
                    let p = transform * cam
                    out.append(SIMD3<Float>(p.x, p.y, p.z))
                }
                u += stride
            }
            v += stride
        }
        return out
    }

    /// Fold one frame's world points into the map.
    func add(points: [SIMD3<Float>]) {
        guard !points.isEmpty else { return }
        if !anchored {
            // centre the map on the first frame's footprint
            var sx: Float = 0, sz: Float = 0
            for p in points { sx += p.x; sz += p.z }
            origin = SIMD2<Float>(sx / Float(points.count), sz / Float(points.count))
            anchored = true
        }
        // Floor: the lowest dense level. A drawer is mostly floor, so the 15th percentile of all heights seen is
        // on the floor and robust to the floor's own noise; tools only ever push the percentile UP if they cover
        // most of the view, which the 15th percentile tolerates up to ~85 % coverage.
        if floorSamples.count < floorSampleCap {
            let step = max(1, points.count / 400)
            var i = 0
            while i < points.count && floorSamples.count < floorSampleCap { floorSamples.append(points[i].y); i += step }
        }
        if floorSamples.count >= 200 {
            let sorted = floorSamples.sorted()
            floorY = sorted[sorted.count * 15 / 100]
        }
        let half = Float(size) * cell / 2
        for p in points {
            let gx = Int((p.x - origin.x + half) / cell)
            let gz = Int((p.z - origin.y + half) / cell)
            guard gx >= 0, gz >= 0, gx < size, gz < size else { continue }
            let idx = gz * size + gx
            let hgt = floorY.isNaN ? p.y : p.y - floorY
            if count[idx] == 0 || height[idx].isNaN || hgt > height[idx] { height[idx] = hgt }
            if count[idx] < UInt16.max { count[idx] += 1 }
        }
    }

    /// Height (metres above the floor) at a world (x, z), or nil where nothing has been seen.
    func height(atX x: Float, z: Float) -> Float? {
        let half = Float(size) * cell / 2
        let gx = Int((x - origin.x + half) / cell), gz = Int((z - origin.y + half) / cell)
        guard gx >= 0, gz >= 0, gx < size, gz < size, count[gz * size + gx] > 0 else { return nil }
        let h = height[gz * size + gx]
        return floorY.isNaN ? nil : h
    }

    /// Bounding box of covered cells (gx0, gz0, gx1, gz1) or nil.
    func coveredBounds() -> (Int, Int, Int, Int)? {
        var x0 = size, z0 = size, x1 = -1, z1 = -1
        for gz in 0..<size {
            let row = gz * size
            for gx in 0..<size where count[row + gx] > 0 {
                if gx < x0 { x0 = gx }; if gx > x1 { x1 = gx }
                if gz < z0 { z0 = gz }; if gz > z1 { z1 = gz }
            }
        }
        return x1 < 0 ? nil : (x0, z0, x1, z1)
    }

    /// RGBA8 picture of the covered region: unseen = transparent, floor = dark, tools brighten with height
    /// (turbo-like ramp, `maxHeightMetres` = full brightness). Returns (bytes, width, height) or nil.
    func rgba(maxHeightMetres: Float = 0.06, pad: Int = 4) -> (bytes: [UInt8], w: Int, h: Int)? {
        guard let (bx0, bz0, bx1, bz1) = coveredBounds() else { return nil }
        let x0 = max(0, bx0 - pad), z0 = max(0, bz0 - pad), x1 = min(size - 1, bx1 + pad), z1 = min(size - 1, bz1 + pad)
        let w = x1 - x0 + 1, h = z1 - z0 + 1
        var out = [UInt8](repeating: 0, count: w * h * 4)
        for gz in z0...z1 {
            for gx in x0...x1 {
                let idx = gz * size + gx
                let o = ((gz - z0) * w + (gx - x0)) * 4
                guard count[idx] > 0 else { continue }
                let hgt = floorY.isNaN ? 0 : max(0, height[idx])
                let t = min(1, hgt / maxHeightMetres)
                let (r, g, b) = LiveHeightMap.ramp(t)
                out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255
            }
        }
        return (out, w, h)
    }

    /// Floor reads dark slate; anything raised runs blue → green → yellow → red, so a 6 mm tool is visibly
    /// not floor and a 40 mm tool is unmistakable.
    static func ramp(_ t: Float) -> (UInt8, UInt8, UInt8) {
        if t < 0.04 { return (48, 52, 60) }
        let s = (t - 0.04) / 0.96
        let stops: [(Float, SIMD3<Float>)] = [(0, SIMD3(40, 90, 220)), (0.33, SIMD3(40, 200, 120)),
                                              (0.66, SIMD3(240, 210, 40)), (1, SIMD3(235, 60, 40))]
        var c = stops.last!.1
        for i in 1..<stops.count where s <= stops[i].0 {
            let (a, ca) = stops[i - 1], (b, cb) = stops[i]
            let f = (s - a) / max(0.0001, b - a)
            c = ca + (cb - ca) * f
            break
        }
        return (UInt8(min(255, max(0, c.x))), UInt8(min(255, max(0, c.y))), UInt8(min(255, max(0, c.z))))
    }
}
