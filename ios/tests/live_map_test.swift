import simd
import Foundation

// Synthetic scene: floor at world y = 0, a 100 x 40 mm block 30 mm tall with its centre at world (x=+0.10, z=-0.05).
// Camera 0.5 m above the floor looking straight down, then tilted 15 deg, then rotated 90 deg about vertical.
// The map must read the block's height, its footprint size, and put it on the correct SIDE (handedness).
var failures = 0
func check(_ ok: Bool, _ msg: String) { print((ok ? "  ok   " : "  FAIL ") + msg); if !ok { failures += 1 } }

let W = 256, H = 192
let fx: Float = 210, fy: Float = 210, cx: Float = 128, cy: Float = 96

/// Render a depth map from a camera pose by ray-marching the analytic scene (floor + block).
func render(transform T: simd_float4x4) -> [Float] {
    var depth = [Float](repeating: 0, count: W * H)
    let inv = T.inverse
    // scene in world: block occupies x in [0.05,0.15], z in [-0.07,-0.03], y in [0,0.03]
    func sceneHeight(_ x: Float, _ z: Float) -> Float { (x >= 0.05 && x <= 0.15 && z >= -0.07 && z <= -0.03) ? 0.03 : 0 }
    let camPos = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
    for v in 0..<H {
        for u in 0..<W {
            // ray in camera space through pixel (u,v): x right, y up, -z forward
            let dirCam = simd_normalize(SIMD3<Float>((Float(u) - cx) / fx, -(Float(v) - cy) / fy, -1))
            let d4 = T * SIMD4<Float>(dirCam, 0); let dir = SIMD3<Float>(d4.x, d4.y, d4.z)
            // march to the first surface: try block top first (y = 0.03), then floor (y = 0)
            var hitDepth: Float = 0
            for planeY: Float in [0.03, 0] {
                guard dir.y < -1e-6 else { continue }
                let t = (planeY - camPos.y) / dir.y
                let p = camPos + dir * t
                if planeY == 0.03 && sceneHeight(p.x, p.z) < 0.03 { continue }
                // depth as the camera measures it: distance along the optical axis (-z of camera)
                let pc = inv * SIMD4<Float>(p, 1)
                hitDepth = -pc.z
                break
            }
            depth[v * W + u] = hitDepth
        }
    }
    _ = inv
    return depth
}

/// Camera at `pos` looking straight down (-y), with the image's "up" (camera +y) pointing to world -z, rotated `yawDeg`
/// about vertical and tilted `tiltDeg` about the camera x axis.
func pose(pos: SIMD3<Float>, yawDeg: Float, tiltDeg: Float) -> simd_float4x4 {
    // looking down: camera -z -> world -y  =>  camera +z -> world +y ; camera +y -> world -z ; camera +x -> world +x
    var R = simd_float3x3(SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 0, -1), SIMD3<Float>(0, 1, 0))  // columns = camera axes in world
    let yaw = simd_float3x3(simd_quatf(angle: yawDeg * .pi / 180, axis: SIMD3<Float>(0, 1, 0)))
    let tilt = simd_float3x3(simd_quatf(angle: tiltDeg * .pi / 180, axis: SIMD3<Float>(1, 0, 0)))
    R = yaw * R * tilt
    var T = matrix_identity_float4x4
    T.columns.0 = SIMD4<Float>(R.columns.0, 0); T.columns.1 = SIMD4<Float>(R.columns.1, 0); T.columns.2 = SIMD4<Float>(R.columns.2, 0)
    T.columns.3 = SIMD4<Float>(pos, 1)
    return T
}

func run(_ label: String, yaw: Float, tilt: Float, camPos: SIMD3<Float>) {
    print(label)
    let T = pose(pos: camPos, yawDeg: yaw, tiltDeg: tilt)
    let depth = render(transform: T)
    let pts = LiveHeightMap.worldPoints(depth: depth, w: W, h: H, fx: fx, fy: fy, cx: cx, cy: cy, transform: T, stride: 1)
    check(pts.count > 20_000, "unprojected \(pts.count) points")
    // world points must land back on the analytic surfaces
    let onFloor = pts.filter { abs($0.y) < 0.002 }.count, onTop = pts.filter { abs($0.y - 0.03) < 0.002 }.count
    check(onFloor + onTop > pts.count * 97 / 100, "points sit on floor or block top: \(onFloor) floor, \(onTop) top of \(pts.count)")
    let map = LiveHeightMap()
    map.add(points: pts)
    check(map.hasFloor && abs(map.floorY) < 0.003, String(format: "floor found at y = %.4f m (truth 0)", map.floorY))
    // block centre must read ~30 mm; a point outside it ~0
    let hc = map.height(atX: 0.10, z: -0.05) ?? -1
    check(abs(hc - 0.03) < 0.004, String(format: "block centre reads %.1f mm (truth 30)", hc * 1000))
    let hf = map.height(atX: -0.05, z: 0.05) ?? -1
    check(hf >= 0 && hf < 0.004, String(format: "bare floor reads %.1f mm (truth 0)", hf * 1000))
    // handedness: the block is at +x, -z; the mirror image would be at -x. Nothing tall may sit there.
    let hm = map.height(atX: -0.10, z: -0.05) ?? 0
    check(hm < 0.004, String(format: "mirror position (-x) reads %.1f mm — must be floor", hm * 1000))
    // footprint size from the map: count cells above 15 mm along the block's axes
    var tallX = Set<Int>(), tallZ = Set<Int>()
    let half = Float(map.size) * map.cell / 2
    for gz in 0..<map.size { for gx in 0..<map.size {
        let i = gz * map.size + gx
        if map.count[i] > 0 && map.height[i] > 0.015 { tallX.insert(gx); tallZ.insert(gz) }
    } }
    let wx = Float(tallX.count) * map.cell * 1000, wz = Float(tallZ.count) * map.cell * 1000
    check(abs(wx - 100) <= 10 && abs(wz - 40) <= 10, String(format: "block footprint reads %.0f x %.0f mm (truth 100 x 40, +-10)", wx, wz))
    _ = half
    guard let pic = map.rgba() else { check(false, "rgba"); return }
    check(pic.w > 10 && pic.h > 10, "rgba picture \(pic.w) x \(pic.h)")
}

run("straight down, 0.5 m", yaw: 0, tilt: 0, camPos: SIMD3<Float>(0.05, 0.5, 0))
run("tilted 15 deg, offset camera", yaw: 0, tilt: 15, camPos: SIMD3<Float>(-0.05, 0.45, 0.10))
run("yawed 90 deg (phone turned sideways)", yaw: 90, tilt: 0, camPos: SIMD3<Float>(0.1, 0.5, -0.05))
run("yawed 90, tilted 10", yaw: 90, tilt: 10, camPos: SIMD3<Float>(0.1, 0.55, 0.05))

if failures == 0 { print("ALL CHECKS PASSED") } else { print("\(failures) FAILURES"); exit(1) }
