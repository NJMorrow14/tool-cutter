import simd
import Foundation

// World: ARKit gravity alignment, +y up. Phone in portrait, camera on the back looking down,
// screen up, top edge pointing "north" = world -z. Screen right = world +x ("east").
//
// Raising an edge of the phone tips the camera TOWARD that edge: lift the right edge and the
// screen's normal leans west, so the camera (opposite normal) aims east.

var failures = 0
func check(_ cond: Bool, _ what: String) {
    print(cond ? "  ok   \(what)" : "  FAIL \(what)")
    if !cond { failures += 1 }
}

/// view matrix for a phone tilted `deg` about its long axis (+ = right edge up) and
/// `degNS` about its short axis (+ = top edge up).
func view(rightEdgeUp deg: Float, topEdgeUp degNS: Float) -> simd_float4x4 {
    let a = deg * .pi / 180, b = degNS * .pi / 180
    // level basis: right = +x, up(screen) = -z (north), backward = +y
    let r0 = simd_float3(1, 0, 0), u0 = simd_float3(0, 0, -1), b0 = simd_float3(0, 1, 0)
    // lifting the right edge rotates the phone about its up/north axis; lifting the top edge about right.
    // Right-hand rule about u0 = (0,0,-1) sends +x toward -y, i.e. the right edge DOWN — hence the -a.
    let qa = simd_quatf(angle: -a, axis: u0)
    let qb = simd_quatf(angle: b, axis: r0)
    let q = qb * qa
    return LevelMath.viewMatrix(right: q.act(r0), up: q.act(u0), backward: q.act(b0))
}

@main
struct LevelMathTest {
    static func main() {
        print("=== bullseye level")
        let level = LevelMath.tiltVector(view: view(rightEdgeUp: 0, topEdgeUp: 0))
        check(simd_length(level) < 0.01, "pointing straight down -> bubble centred")

        let right = LevelMath.tiltVector(view: view(rightEdgeUp: 10, topEdgeUp: 0))
        check(abs(simd_length(right) - 10) < 0.01, "10 deg tilt -> 10 deg reading (got \(simd_length(right)))")
        check(right.x < -1 && abs(right.y) < 0.01, "right edge up -> dot moves LEFT (got \(right))")

        let left = LevelMath.tiltVector(view: view(rightEdgeUp: -10, topEdgeUp: 0))
        check(left.x > 1 && abs(left.y) < 0.01, "left edge up -> dot moves RIGHT (got \(left))")

        let top = LevelMath.tiltVector(view: view(rightEdgeUp: 0, topEdgeUp: 10))
        check(top.y > 1 && abs(top.x) < 0.01, "top edge up -> dot moves DOWN (got \(top))")

        let bottom = LevelMath.tiltVector(view: view(rightEdgeUp: 0, topEdgeUp: -10))
        check(bottom.y < -1 && abs(bottom.x) < 0.01, "bottom edge up -> dot moves UP (got \(bottom))")

        let diag = LevelMath.tiltVector(view: view(rightEdgeUp: 10, topEdgeUp: 10))
        check(diag.x < -1 && diag.y > 1, "two axes at once -> diagonal, both components live (got \(diag))")
        check(abs(simd_length(diag) - 14.1) < 0.5, "diagonal magnitude ~14 deg (got \(simd_length(diag)))")

        // the old code could only ever move the bubble one way; every direction must now be reachable
        let dirs = [right, left, top, bottom].map { atan2($0.y, $0.x) }
        check(Set(dirs.map { Int(($0 * 180 / .pi).rounded()) }).count == 4, "all four directions are distinct")

        print(failures == 0 ? "\nALL CHECKS PASSED" : "\n\(failures) FAILED")
        exit(failures == 0 ? 0 : 1)
    }
}
