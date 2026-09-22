import simd

/// The bullseye-level maths, kept free of ARKit/UIKit so it can be tested on the Mac
/// (`ios/tests/run_level_math_test.sh`) — the sign of the offset is very easy to get backwards.
enum LevelMath {
    /// Where straight down sits relative to where the camera is aiming, in SCREEN axes
    /// (+x right, +y down), in degrees. Its length is the angle off straight down.
    ///
    /// `view` is `ARCamera.viewMatrix(for:)`: world -> view space, with the interface rotation already
    /// applied, so view space is +x right on screen, +y up on screen, -z the way the camera looks.
    ///
    /// The dot is a TARGET, not a floating bubble: it marks straight down in the scene you are looking
    /// at, so you steer the camera onto it. Aim the camera east of straight down and the dot sits to
    /// the west — i.e. on the LOW side, where a spirit level's bubble would not be. Negate to swap.
    static func tiltVector(view: simd_float4x4) -> simd_float2 {
        let g = simd_normalize(simd_make_float3(view * simd_float4(0, -1, 0, 0)))
        let tilt = Float(acos(max(-1, min(1, -g.z))) * 180 / .pi)
        let planar = simd_float2(g.x, -g.y)          // screen y grows downward
        let len = simd_length(planar)
        return len > 1e-5 ? planar / len * tilt : simd_float2(0, 0)
    }

    /// Build a world->view matrix from the camera's screen-right / screen-up / backward axes in world.
    /// Only the rotation matters here (gravity is a direction), so translation is left at zero.
    static func viewMatrix(right r: simd_float3, up u: simd_float3, backward b: simd_float3) -> simd_float4x4 {
        // rows are the basis vectors: the transpose of the camera-to-world rotation
        simd_float4x4(rows: [simd_float4(r, 0), simd_float4(u, 0), simd_float4(b, 0), simd_float4(0, 0, 0, 1)])
    }
}
