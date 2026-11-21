import Foundation
import CoreGraphics
import ImageIO

// SAM2Engine.swift on the Mac: load the compiled SAM 2.1 Small models, encode a real rear-camera drawer photo
// (capture a237eba87dba, frame_000.jpg), prompt one foreground point on an object, check the mask, write an overlay.
var failures = 0
func check(_ ok: Bool, _ msg: String) { print((ok ? "PASS " : "FAIL ") + msg); if !ok { failures += 1 } }
func ms(_ t: Date) -> String { String(format: "%.0f ms", Date().timeIntervalSince(t) * 1000) }

let args = CommandLine.arguments
let modelsDir = URL(fileURLWithPath: args[1])
let photoURL = URL(fileURLWithPath: args[2])
let outURL = URL(fileURLWithPath: args[3])

guard let src = CGImageSourceCreateWithURL(photoURL as CFURL, nil), let full = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    print("FAIL cannot read \(photoURL.path)"); exit(1)
}
print("photo \(full.width) x \(full.height)")
// A square crop around the pill bottle and the USB stick (full-res pixels), so SAM sees them large.
let cropRect = CGRect(x: 600, y: 400, width: 1400, height: 1400)
let crop = full.cropping(to: cropRect)!

var t = Date()
let engine: SAM2Engine
do { engine = try SAM2Engine(modelsDirectory: modelsDir) } catch { print("FAIL load: \(error)"); exit(1) }
print("     load: \(ms(t))")
check(!engine.isReady, "not ready before encode")
do { _ = try engine.predictMask(prompts: [.init(x: 1, y: 1, label: 1)]); check(false, "predict before encode throws") }
catch { check(true, "predict before encode throws (\(error))") }

do {
    t = Date(); try engine.encode(crop); print("     encode (first, includes warm-up): \(ms(t))")
    t = Date(); try engine.encode(crop); print("     encode (second): \(ms(t))")
    check(engine.isReady, "ready after encode")

    // objects: pill bottle label centre, USB stick body (crop pixels)
    // The pill bottle is THE check (one point at its centre). The USB stick is small (~1 % of the crop) and a single
    // point on its red slider gets the slider alone (SAM ranks the part highest), so it takes two points on the body.
    let objects: [(String, [CGPoint], (UInt8, UInt8, UInt8), Double)] = [
        ("pill bottle", [CGPoint(x: 773, y: 645)], (30, 144, 255), 0.01),
        ("USB stick", [CGPoint(x: 1206, y: 653), CGPoint(x: 1206, y: 871)], (255, 140, 0), 0.005),
    ]
    var masks: [(BitMask, (UInt8, UInt8, UInt8), [CGPoint]?)] = []
    for (name, pts, colour, minFrac) in objects {
        let prompts = pts.map { SAM2Engine.Prompt(x: $0.x, y: $0.y, label: 1) }
        t = Date()
        let m = try engine.predictMask(prompts: prompts)
        print("     predictMask (\(name), \(pts.count) point(s)): \(ms(t))  scores \(engine.lastScores.map { String(format: "%.3f", $0) })")
        t = Date(); _ = try engine.predictMask(prompts: prompts); print("     predictMask again: \(ms(t))")
        check(m.width == crop.width && m.height == crop.height, "\(name): mask is at the original size \(m.width) x \(m.height)")
        let frac = Double(m.count) / Double(m.width * m.height)
        check(frac > minFrac && frac < 0.60, "\(name): mask covers \(String(format: "%.2f", frac * 100)) % of the image (\(minFrac * 100)-60 %)")
        check(pts.allSatisfy { m[Int($0.x), Int($0.y)] == 1 }, "\(name): mask contains the prompt point(s)")
        let ring = OutlineEngine.largestComponentPolygon(mask: m, minArea: 100)
        if let ring { print("     \(name): outline \(ring.count) pts, bbox \(OutlineEngine.boundingBox(ring).integral)") }
        masks.append((m, colour, ring))
    }
    let (logits, side) = try engine.predictLogits(prompts: [.init(x: 773, y: 645, label: 1), .init(x: 1195, y: 744, label: 0)])
    check(side == 256 && logits.count == 256 * 256, "predictLogits returns 256 x 256 logits (fg + bg prompt)")
    t = Date()
    let many = (0..<14).map { SAM2Engine.Prompt(x: 700 + Double($0) * 10, y: 645, label: 1) }
    _ = try engine.predictMask(prompts: many)
    check(true, "14 prompts accepted (\(ms(t)))")
    do { _ = try engine.predictMask(prompts: many + many); check(false, "28 prompts rejected") } catch { check(true, "28 prompts rejected (\(error))") }
    let box = try engine.predictMask(prompts: [.init(x: 640, y: 480, label: 2), .init(x: 900, y: 780, label: 3)])
    check(box[773, 645] == 1, "box prompt (2/3) around the bottle contains its centre (\(String(format: "%.1f", Double(box.count) / 19600)) %)")

    // overlay
    let W = crop.width, H = crop.height
    var px = [UInt8](repeating: 0, count: W * H * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(crop, in: CGRect(x: 0, y: 0, width: W, height: H))
    for (m, c, _) in masks {
        for i in 0..<(W * H) where m.data[i] != 0 {
            px[i * 4] = UInt8((Int(px[i * 4]) + Int(c.0)) / 2)
            px[i * 4 + 1] = UInt8((Int(px[i * 4 + 1]) + Int(c.1)) / 2)
            px[i * 4 + 2] = UInt8((Int(px[i * 4 + 2]) + Int(c.2)) / 2)
        }
    }
    // outlines (traced by OutlineEngine) in green, prompt points in red; CGContext is y-up, rings are y-down
    ctx.translateBy(x: 0, y: CGFloat(H)); ctx.scaleBy(x: 1, y: -1)
    ctx.setStrokeColor(red: 0, green: 1, blue: 0.2, alpha: 1); ctx.setLineWidth(3)
    for (_, _, ring) in masks { if let ring { ctx.addLines(between: ring); ctx.closePath(); ctx.strokePath() } }
    ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
    for (_, pts, _, _) in objects { for p in pts { ctx.fill(CGRect(x: p.x - 8, y: p.y - 8, width: 16, height: 16)) } }
    let img = ctx.makeImage()!
    try? FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    let dst = CGImageDestinationCreateWithURL(outURL as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    check(CGImageDestinationFinalize(dst), "overlay written to \(outURL.path)")
} catch {
    check(false, "SAM2 error: \(error)")
}

print(failures == 0 ? "ALL SAM2 CHECKS PASSED" : "\(failures) CHECK(S) FAILED")
exit(failures == 0 ? 0 : 1)
