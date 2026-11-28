import SwiftUI
import CoreGraphics

/// The live top-down height map, drawn in a corner of the capture screen during the rear LiDAR pass.
/// Floor is dark slate; anything raised runs blue → red with height. Unseen cells are transparent, so the
/// shape of the picture IS the coverage — a hole means "you have not been there yet".
struct LiveMapView: View {
    let image: CGImage?
    let cellsCovered: Int
    let frames: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.medium)                  // 2.5 mm cells, lightly smoothed: reads as a surface
                    .aspectRatio(CGFloat(image.width) / CGFloat(image.height), contentMode: .fit)
                    .frame(maxWidth: 210, maxHeight: 280)
                    .background(Color.black.opacity(0.35))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.white.opacity(0.35), lineWidth: 1))
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.black.opacity(0.35))
                    .frame(width: 210, height: 140)
                    .overlay(Text("Live map appears\nas you sweep").font(.caption2).multilineTextAlignment(.center).foregroundStyle(.white.opacity(0.7)))
            }
            Text(String(format: "%.0f cm² seen · %d frames", Double(cellsCovered) * 0.0625, frames))
                .font(.caption2.monospaced()).foregroundStyle(.white.opacity(0.85))
        }
        .padding(6)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityLabel("Live coverage map")
    }
}

enum LiveMapImage {
    /// RGBA8 bytes → CGImage. Rows are map z (world "down the page"), columns map x.
    static func make(_ pic: (bytes: [UInt8], w: Int, h: Int)) -> CGImage? {
        var bytes = pic.bytes
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        return bytes.withUnsafeMutableBytes { buf -> CGImage? in
            guard let ctx = CGContext(data: buf.baseAddress, width: pic.w, height: pic.h, bitsPerComponent: 8,
                                      bytesPerRow: pic.w * 4, space: cs, bitmapInfo: info.rawValue) else { return nil }
            return ctx.makeImage()
        }
    }
}
