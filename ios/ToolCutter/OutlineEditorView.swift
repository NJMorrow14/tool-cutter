import SwiftUI
import UIKit

/// The photo with its outline, editable by finger. No modes:
///   ONE finger drags  -> a correction stroke (drawn live in white; on release the outline is recalculated there)
///   ONE finger taps   -> a point prompt (used when there is no outline yet, or to pick another object)
///   TWO fingers       -> pan / pinch-zoom (UIScrollView; its pan needs 2 touches, so one finger never scrolls)
/// Everything is drawn in IMAGE PIXEL coordinates inside the zoomed content view, so a touch's location in that view
/// is already the image pixel (x right, y down) the engine works in.
struct OutlineEditorView: UIViewRepresentable {
    let image: CGImage?
    let ring: [CGPoint]?
    let busy: Bool
    var onStroke: ([CGPoint]) -> Void
    var onTap: (CGPoint) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> EditorScrollView {
        let v = EditorScrollView()
        context.coordinator.install(in: v)
        return v
    }

    func updateUIView(_ v: EditorScrollView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.update(image: image, ring: ring, busy: busy)
    }

    final class EditorScrollView: UIScrollView {
        var onLayout: (() -> Void)?
        override func layoutSubviews() { super.layoutSubviews(); onLayout?() }
    }

    final class Coordinator: NSObject, UIScrollViewDelegate, UIGestureRecognizerDelegate {
        var parent: OutlineEditorView
        private weak var scroll: EditorScrollView?
        private let content = UIView()
        private let imageView = UIImageView()
        private let fillLayer = CAShapeLayer()
        private let outlineLayer = CAShapeLayer()
        private let strokeLayer = CAShapeLayer()
        private var currentImage: CGImage?
        private var imageSize: CGSize = .zero
        private var stroke: [CGPoint] = []
        private var strokeScreenLength: CGFloat = 0
        private var lastBoundsSize: CGSize = .zero
        private var strokePan: UIPanGestureRecognizer!
        private var lastRing: [CGPoint]?
        private var pulsing = false

        init(_ p: OutlineEditorView) { parent = p }

        func install(in v: EditorScrollView) {
            scroll = v
            v.delegate = self
            v.backgroundColor = .black
            v.showsHorizontalScrollIndicator = false
            v.showsVerticalScrollIndicator = false
            v.contentInsetAdjustmentBehavior = .never
            v.bouncesZoom = true
            v.delaysContentTouches = false
            v.panGestureRecognizer.minimumNumberOfTouches = 2          // one finger draws, two pan
            v.onLayout = { [weak self] in self?.layoutChanged() }
            v.addSubview(content)
            imageView.contentMode = .scaleToFill
            content.addSubview(imageView)
            let orange = UIColor(red: 0xf2 / 255, green: 0x6a / 255, blue: 0x1b / 255, alpha: 1)
            fillLayer.fillColor = orange.withAlphaComponent(0.22).cgColor
            fillLayer.strokeColor = nil
            outlineLayer.fillColor = nil
            outlineLayer.strokeColor = orange.cgColor
            outlineLayer.lineJoin = .round
            strokeLayer.fillColor = nil
            strokeLayer.strokeColor = UIColor.white.cgColor
            strokeLayer.lineCap = .round
            strokeLayer.lineJoin = .round
            strokeLayer.shadowColor = UIColor.black.cgColor
            strokeLayer.shadowOpacity = 0.6
            strokeLayer.shadowRadius = 1
            strokeLayer.shadowOffset = .zero
            for l in [fillLayer, outlineLayer, strokeLayer] { content.layer.addSublayer(l) }

            strokePan = UIPanGestureRecognizer(target: self, action: #selector(handleStroke(_:)))
            strokePan.maximumNumberOfTouches = 1
            strokePan.delegate = self
            content.addGestureRecognizer(strokePan)
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            tap.numberOfTouchesRequired = 1
            content.addGestureRecognizer(tap)
            v.pinchGestureRecognizer?.addTarget(self, action: #selector(handlePinch(_:)))
        }

        func update(image: CGImage?, ring: [CGPoint]?, busy: Bool) {
            if image !== currentImage {
                currentImage = image
                stroke = []; strokeLayer.path = nil
                if let image {
                    imageSize = CGSize(width: image.width, height: image.height)
                    imageView.image = UIImage(cgImage: image)
                } else {
                    imageSize = .zero
                    imageView.image = nil
                }
                lastBoundsSize = .zero                                 // force a fresh fit
                layoutChanged()
            }
            if ring != lastRing {
                lastRing = ring
                let path = CGMutablePath()
                if let ring, ring.count >= 3 { path.addLines(between: ring); path.closeSubpath() }
                fillLayer.path = path
                outlineLayer.path = path
            }
            setPulsing(busy)
        }

        private func setPulsing(_ on: Bool) {
            guard on != pulsing else { return }
            pulsing = on
            if on {
                let a = CABasicAnimation(keyPath: "opacity")
                a.fromValue = 1; a.toValue = 0.25; a.duration = 0.35
                a.autoreverses = true; a.repeatCount = .infinity
                outlineLayer.add(a, forKey: "busy"); fillLayer.add(a, forKey: "busy")
            } else {
                outlineLayer.removeAnimation(forKey: "busy"); fillLayer.removeAnimation(forKey: "busy")
            }
        }

        // MARK: zoom / layout

        private func layoutChanged() {
            guard let v = scroll, imageSize.width > 0, v.bounds.width > 0 else { return }
            if v.bounds.size != lastBoundsSize {
                lastBoundsSize = v.bounds.size
                let fit = min(v.bounds.width / imageSize.width, v.bounds.height / imageSize.height)
                v.minimumZoomScale = fit
                v.maximumZoomScale = max(fit, 6)
                v.zoomScale = 1
                content.frame = CGRect(origin: .zero, size: imageSize)
                imageView.frame = content.bounds
                for l in [fillLayer, outlineLayer, strokeLayer] { l.frame = content.bounds }
                v.contentSize = imageSize
                v.zoomScale = fit
            }
            centre()
            updateLineWidths()
        }

        func viewForZooming(in scrollView: UIScrollView) -> UIView? { content }
        func scrollViewDidZoom(_ scrollView: UIScrollView) { centre(); updateLineWidths() }

        private func centre() {
            guard let v = scroll else { return }
            let dx = max(0, (v.bounds.width - v.contentSize.width) / 2), dy = max(0, (v.bounds.height - v.contentSize.height) / 2)
            v.contentInset = UIEdgeInsets(top: dy, left: dx, bottom: dy, right: dx)
        }

        /// Constant ON-SCREEN widths: the layers live in image pixels, so divide by the zoom.
        private func updateLineWidths() {
            guard let z = scroll?.zoomScale, z > 0 else { return }
            CATransaction.begin(); CATransaction.setDisableActions(true)
            outlineLayer.lineWidth = 2 / z
            strokeLayer.lineWidth = 3 / z
            strokeLayer.shadowRadius = 1
            CATransaction.commit()
        }

        // MARK: gestures

        @objc private func handleStroke(_ g: UIPanGestureRecognizer) {
            let p = g.location(in: content)
            switch g.state {
            case .began:
                // the pan is recognised only after ~10 pt of movement: put back where the finger came down
                let t = g.translation(in: content)
                let start = CGPoint(x: p.x - t.x, y: p.y - t.y)
                stroke = [start, p]
                strokeScreenLength = hypot(t.x, t.y) * (scroll?.zoomScale ?? 1)
            case .changed:
                if g.numberOfTouches > 1 { cancelStroke(); return }
                if let last = stroke.last {
                    let z = scroll?.zoomScale ?? 1
                    let d = hypot(p.x - last.x, p.y - last.y)
                    if d * z < 1.5 { return }                           // < 1.5 screen points: not worth a vertex
                    strokeScreenLength += d * z
                }
                stroke.append(p)
                let path = CGMutablePath(); path.addLines(between: stroke)
                CATransaction.begin(); CATransaction.setDisableActions(true)
                strokeLayer.path = path
                CATransaction.commit()
            case .ended:
                let s = stroke
                let long = strokeScreenLength >= 10
                clearStroke()
                if long && s.count >= 2 { parent.onStroke(s) }
            default:
                cancelStroke()
            }
        }

        private func clearStroke() {
            stroke = []
            CATransaction.begin(); CATransaction.setDisableActions(true)
            strokeLayer.path = nil
            CATransaction.commit()
        }

        /// A second finger landing turns the gesture into a zoom: drop the stroke the first finger had started.
        private func cancelStroke() {
            clearStroke()
            strokePan.isEnabled = false
            strokePan.isEnabled = true
        }

        @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
            if g.state == .began, !stroke.isEmpty { cancelStroke() }
        }

        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            guard g.state == .ended else { return }
            parent.onTap(g.location(in: content))
        }

        func gestureRecognizer(_ g: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            g === strokePan && other === scroll?.pinchGestureRecognizer
        }
    }
}
