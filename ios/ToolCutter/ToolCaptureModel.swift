import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import Combine
import simd

/// One tool at a time (docs/tool-library.md, "Phone"): shutter -> automatic outline -> finger corrections ->
/// name + thickness -> save into the current job -> next tool.
///
/// Degrade, don't block: no LiDAR -> type the longest side; no SAM -> corrections are geometric (+ edge snap);
/// no Vision result -> SAM box prompt, then "tap the tool", then "draw around it"; no network -> queued on disk.
@MainActor
final class ToolCaptureModel: ObservableObject {
    enum Phase: Equatable { case aiming, processing, editing, details, saving }

    @Published var phase: Phase = .aiming
    @Published private(set) var still: CaptureController.StillCapture?
    /// The photo everything is outlined on: the upright still, longest side <= `workSide` px.
    @Published private(set) var workImage: CGImage?
    private(set) var workIntrinsics: CameraIntrinsics?
    @Published private(set) var outline: OutlineState?
    @Published private(set) var findingOutline = false
    /// A correction or tap is being recalculated.
    @Published private(set) var busy = false
    /// One line telling the person what to do next / what just happened.
    @Published var hint: String?
    @Published private(set) var plane: Plane?
    @Published private(set) var scaleStatus = ""
    @Published private(set) var hasLidarScale = false
    @Published private(set) var encodedForCurrentPhoto = false
    @Published private(set) var samStatus = "loading"          // loading | ready | unavailable
    @Published private(set) var canUndo = false
    @Published var toolsInJob = 0
    @Published private(set) var lastMethod: OutlineCorrection.Method?

    private(set) var visionMask: BitMask?
    private var automaticOutline: OutlineState?
    /// Rings before each edit (an empty ring = "no outline"); masks are re-rasterised on undo to keep memory small.
    private var undoStack: [[CGPoint]] = []
    private let maxUndo = 20
    private var gradient: [Float]?
    private var photoToken = 0
    private var encodeTask: Task<Bool, Never>?
    private var tapPoints: [CGPoint] = []
    private var outlineFromTaps = false
    private var capturedAt = Date()
    private var engine: SAM2Engine?

    nonisolated static let workSide = 2048

    init() {
        Task { [weak self] in
            let e = await Self.sharedSAM().value
            guard let self else { return }
            self.engine = e
            self.samStatus = e == nil ? "unavailable" : "ready"
        }
    }

    var edited: Bool { !undoStack.isEmpty }
    var imageSize: CGSize { workImage.map { CGSize(width: $0.width, height: $0.height) } ?? .zero }
    private var minAreaPx: Int { max(400, Int(imageSize.width * imageSize.height) / 500) }

    // MARK: - SAM (one engine for the app, one serial queue for everything that touches it)

    private static var samLoad: Task<SAM2Engine?, Never>?
    private static let samQueue = DispatchQueue(label: "toolcutter.sam2", qos: .userInitiated)

    /// The three Core ML models are compiled by Xcode into `<Name>.mlmodelc` at the bundle root, which is the
    /// directory SAM2Engine expects (it appends `SAM2_1Small<Part>FLOAT16.mlmodelc`).
    static func sharedSAM() -> Task<SAM2Engine?, Never> {
        if let t = samLoad { return t }
        let t = Task.detached(priority: .utility) { () -> SAM2Engine? in
            guard let dir = Bundle.main.resourceURL else { return nil }
            do { return try await SAM2Engine.load(modelsDirectory: dir) }
            catch { print("SAM2 unavailable: \(error)"); return nil }
        }
        samLoad = t
        return t
    }

    private static func onSAMQueue<T>(_ work: @escaping () -> T) async -> T {
        await withCheckedContinuation { c in samQueue.async { c.resume(returning: work()) } }
    }

    private static func offMain<T>(_ work: @escaping () -> T) async -> T {
        await withCheckedContinuation { c in DispatchQueue.global(qos: .userInitiated).async { c.resume(returning: work()) } }
    }

    // MARK: - Shutter -> automatic outline

    func shutter(controller: CaptureController) async {
        guard phase == .aiming else { return }
        phase = .processing
        clearPhoto()
        photoToken += 1
        let token = photoToken
        let s: CaptureController.StillCapture
        do { s = try await controller.captureStillWithDepth() }
        catch {
            hint = "Could not take the photo: \(error.localizedDescription)"
            phase = .aiming
            return
        }
        guard token == photoToken else { return }
        still = s
        capturedAt = Date()
        findingOutline = true
        let (img, K) = await Self.offMain { Self.makeWorkImage(s) }
        guard token == photoToken else { return }
        guard let img else { hint = "Could not read the photo"; phase = .aiming; findingOutline = false; return }
        workImage = img
        workIntrinsics = K
        phase = .editing
        startEncode(img, token: token)
        Task { [weak self] in
            let g = await Self.offMain { Self.gradient(of: img) }
            if let self, self.photoToken == token { self.gradient = g }
        }

        // 1. Vision foreground instance mask
        let vmask = await Self.offMain { Self.foregroundMask(img) }
        guard token == photoToken else { return }
        visionMask = vmask
        var first = vmask.flatMap { OutlineCorrection.initialOutline(mask: $0, minAreaPx: minAreaPx) }
        // 2. SAM with a box over the central 70 % + a centre point (waits for the encode: only on this fallback)
        if first == nil, let engine, await encodeTask?.value == true, token == photoToken {
            let w = Double(img.width), h = Double(img.height)
            let prompts = [SAM2Engine.Prompt(x: 0.15 * w, y: 0.15 * h, label: 2), SAM2Engine.Prompt(x: 0.85 * w, y: 0.85 * h, label: 3),
                           SAM2Engine.Prompt(x: 0.5 * w, y: 0.5 * h, label: 1)]
            let minA = minAreaPx
            first = await Self.onSAMQueue {
                (try? engine.predictMask(prompts: prompts)).flatMap { OutlineCorrection.initialOutline(mask: $0, minAreaPx: minA) }
            }
        }
        guard token == photoToken else { return }
        automaticOutline = first
        outline = first
        findingOutline = false
        if first == nil {
            hint = engine != nil ? "Couldn't find the tool — tap it (tap again to add a part it missed)."
                                 : "Couldn't find the tool — draw around it with one finger."
        } else {
            hint = "Retrace any wrong part with one finger. Two fingers zoom."
        }
        // 3. scale: the resting plane from the LiDAR depth, the tool excluded
        let mask = first?.mask
        let fitted = await Self.offMain { Self.fitScalePlane(s, workK: K, toolMask: mask) }
        guard token == photoToken else { return }
        setScale(fitted, still: s)
    }

    private func clearPhoto() {
        still = nil; workImage = nil; workIntrinsics = nil; outline = nil; automaticOutline = nil; visionMask = nil
        undoStack = []; canUndo = false; gradient = nil; tapPoints = []; outlineFromTaps = false
        plane = nil; hasLidarScale = false; scaleStatus = ""; encodedForCurrentPhoto = false; lastMethod = nil
        busy = false; findingOutline = false
        encodeTask = nil
    }

    private func startEncode(_ img: CGImage, token: Int) {
        encodeTask = Task { [weak self] in
            guard let engine = await Self.sharedSAM().value else { return false }
            let ok = await Self.onSAMQueue { (try? engine.encode(img)) != nil }
            guard let self, self.photoToken == token else { return false }
            if ok { self.encodedForCurrentPhoto = true }
            return ok
        }
    }

    // MARK: - Corrections

    /// One finger stroke, in work-image pixels. Never waits for SAM: it is used only once the photo is encoded.
    func applyStroke(_ stroke: [CGPoint]) {
        guard phase == .editing, !busy, !findingOutline, stroke.count >= 2 else { return }
        let (w, h) = (Int(imageSize.width), Int(imageSize.height))
        guard let state = outline else {
            // nothing to correct: a loop drawn around the tool IS the outline
            if let s = OutlineCorrection.outlineFromLoop(stroke, imageWidth: w, imageHeight: h, minAreaPx: minAreaPx) {
                push([]); outline = s; outlineFromTaps = false
                hint = "Retrace any wrong part with one finger. Two fingers zoom."
                refitScaleIfNeeded()
            } else {
                hint = "Draw all the way around the tool."
            }
            return
        }
        busy = true
        let token = photoToken
        let grad = gradient
        let samEngine = encodedForCurrentPhoto ? engine : nil
        Task { [weak self] in
            let work: () -> (state: OutlineState, method: OutlineCorrection.Method) = {
                let sam: (([SAM2Engine.Prompt]) throws -> BitMask)? = samEngine.map { e in { try e.predictMask(prompts: $0) } }
                return OutlineCorrection.applyStrokeDetailed(state: state, stroke: stroke, imageWidth: w, imageHeight: h,
                                                             gradient: grad, sam: sam)
            }
            let res = samEngine != nil ? await Self.onSAMQueue(work) : await Self.offMain(work)
            guard let self, self.photoToken == token else { return }
            self.push(state.ring)
            self.outline = res.state
            self.lastMethod = res.method
            self.busy = false
        }
    }

    /// A tap: prompts SAM with a foreground point. With no outline (or one made by taps) every tap adds a point — a
    /// single tap on a small object can pick only a part, so a second tap is cheap. A tap OUTSIDE an automatic
    /// outline means "this one": the taps start over there. A tap inside an automatic outline does nothing.
    func tap(at p: CGPoint) {
        guard phase == .editing, !busy, !findingOutline else { return }
        if let o = outline, !outlineFromTaps {
            let x = Int(p.x), y = Int(p.y)
            if x >= 0, y >= 0, x < o.mask.width, y < o.mask.height, o.mask[x, y] != 0 { return }
            tapPoints = []
        }
        guard let engine else { hint = "Draw around the tool with one finger."; return }
        tapPoints.append(p)
        if tapPoints.count > SAM2Engine.maxPrompts { tapPoints.removeFirst() }
        busy = true
        hint = encodedForCurrentPhoto ? nil : "Preparing the photo…"
        let token = photoToken
        let pts = tapPoints, minA = minAreaPx
        let previous = outline
        Task { [weak self] in
            guard let self else { return }
            guard await self.encodeTask?.value == true, self.photoToken == token else {
                if self.photoToken == token { self.busy = false; self.hint = "Draw around the tool with one finger." }
                return
            }
            let prompts = pts.map { SAM2Engine.Prompt(x: Double($0.x), y: Double($0.y), label: 1) }
            let s = await Self.onSAMQueue { (try? engine.predictMask(prompts: prompts)).flatMap { OutlineCorrection.initialOutline(mask: $0, minAreaPx: minA) } }
            guard self.photoToken == token else { return }
            self.busy = false
            if let s {
                self.push(previous?.ring ?? [])
                self.outline = s
                self.outlineFromTaps = true
                self.hint = "Tap a part it missed, or retrace with one finger."
                self.refitScaleIfNeeded()
            } else {
                self.hint = "Nothing found there — tap the middle of the tool, or draw around it."
            }
        }
    }

    func undo() {
        guard let ring = undoStack.popLast(), let img = workImage else { return }
        canUndo = !undoStack.isEmpty
        outline = ring.count >= 3 ? OutlineState(ring: ring, mask: OutlineEngine.rasterize(ring, width: img.width, height: img.height)) : nil
    }

    /// Back to the automatic outline (the edits are forgotten).
    func reset() {
        outline = automaticOutline
        undoStack = []; canUndo = false; tapPoints = []; outlineFromTaps = false
    }

    func retake() {
        photoToken += 1
        clearPhoto()
        hint = nil
        phase = .aiming
    }

    private func push(_ ring: [CGPoint]) {
        undoStack.append(ring)
        if undoStack.count > maxUndo { undoStack.removeFirst() }
        canUndo = true
    }

    // MARK: - Scale

    private func refitScaleIfNeeded() {
        guard plane == nil, let s = still, let K = workIntrinsics, let m = outline?.mask else { return }
        let token = photoToken
        Task { [weak self] in
            let p = await Self.offMain { Self.fitScalePlane(s, workK: K, toolMask: m) }
            if let self, self.photoToken == token { self.setScale(p, still: s) }
        }
    }

    private func setScale(_ p: Plane?, still s: CaptureController.StillCapture) {
        if let p, p.inlierFraction >= 0.3, p.tiltDeg <= 25 {
            plane = p
            hasLidarScale = true
            scaleStatus = String(format: "Scale from LiDAR · %.0f mm · tilt %.0f°", p.distanceM * 1000, p.tiltDeg)
        } else {
            plane = nil
            hasLidarScale = false
            if s.depth == nil {
                scaleStatus = "No LiDAR depth — enter the tool's longest side in mm"
            } else if let p {
                scaleStatus = String(format: "Surface too tilted (%.0f°) or uneven for LiDAR scale — enter the longest side in mm", p.tiltDeg)
            } else {
                scaleStatus = "LiDAR could not find the surface — enter the tool's longest side in mm"
            }
        }
    }

    /// The outline in millimetres + the `scale` record (docs/tool-library.md). Order: LiDAR plane; else the typed
    /// longest side; else an ASSUMED 40 cm distance, labelled as such so the website can show it is not to scale.
    func polygonMm(thicknessMm: Double?, longestSideMm: Double?) -> (ring: [CGPoint], scale: [String: Any])? {
        guard let ring = outline?.ring, ring.count >= 3, let K = workIntrinsics else { return nil }
        if let plane, hasLidarScale {
            let (mm, mmPerPx) = OutlineEngine.projectToPlane(ring, intrinsics: K, plane: plane, raisedByMm: thicknessMm ?? 0)
            guard mm.count >= 3 else { return nil }
            return (mm, ["method": "lidar_plane", "mm_per_px": round4(mmPerPx), "distance_mm": round1(plane.distanceM * 1000),
                         "tilt_deg": round1(plane.tiltDeg), "plane_inliers": round4(plane.inlierFraction)])
        }
        let bb = OutlineEngine.boundingBox(ring)
        let longPx = max(bb.width, bb.height)
        guard longPx > 0 else { return nil }
        let mmPerPx: Double
        var scale: [String: Any]
        if let L = longestSideMm, L > 0 {
            mmPerPx = L / longPx
            scale = ["method": "manual_length", "mm_per_px": round4(mmPerPx), "longest_side_mm": L]
        } else {
            mmPerPx = 400.0 / K.fx
            scale = ["method": "assumed_distance", "mm_per_px": round4(mmPerPx), "distance_mm": 400.0]
        }
        let mm = ring.map { CGPoint(x: (Double($0.x) - Double(bb.minX)) * mmPerPx, y: (Double($0.y) - Double(bb.minY)) * mmPerPx) }
        return (mm, scale)
    }

    /// Size of the outline in mm for the details sheet ("212 × 48 mm"), nil when nothing gives a scale.
    func sizeMm(thicknessMm: Double?, longestSideMm: Double?) -> CGSize? {
        guard hasLidarScale || (longestSideMm ?? 0) > 0, let r = polygonMm(thicknessMm: thicknessMm, longestSideMm: longestSideMm) else { return nil }
        return OutlineEngine.boundingBox(r.ring).size
    }

    // MARK: - Save

    /// Builds the ToolRecord + photo, queues it (disk first), and kicks an upload. Returns the saved name.
    func save(name rawName: String, thicknessMm: Double?, longestSideMm: Double?, jobId: String, jobName: String,
              uploads: PendingUploads, settings: AppSettings) async -> String? {
        guard let poly = polygonMm(thicknessMm: thicknessMm, longestSideMm: longestSideMm), let s = still else {
            hint = "There is no outline to save yet."
            return nil
        }
        phase = .saving
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Tool \(toolsInJob + 1)" : String(rawName.prefix(80))
        let id = Self.newToolId()
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        iso.timeZone = TimeZone(identifier: "UTC")
        let record: [String: Any] = [
            "id": id, "name": name,
            "thickness_mm": thicknessMm.map { $0 as Any } ?? NSNull(),
            "polygon_mm": poly.ring.map { [round2(Double($0.x)), round2(Double($0.y))] },
            "captured_at": iso.string(from: capturedAt),
            "source": "phone_photo",
            "scale": poly.scale,
            "edited": edited,
            "notes": "",
        ]
        let image = s.image
        let jpeg = await Self.offMain { Self.jpeg(image, longestSide: 2000, quality: 0.9) }
        guard let json = try? JSONSerialization.data(withJSONObject: record), let jpeg else {
            hint = "Could not prepare the tool for upload."
            phase = .details
            return nil
        }
        do {
            try uploads.enqueue(toolId: id, jobId: jobId, jobName: jobName, toolJSON: json, photoJPEG: jpeg)
        } catch {
            hint = "Could not save on the phone: \(error.localizedDescription)"
            phase = .details
            return nil
        }
        toolsInJob += 1
        uploads.flush(settings: settings)
        photoToken += 1
        clearPhoto()
        hint = nil
        phase = .aiming
        return name
    }

    /// "N tools in <job>": the server's count plus what is still queued here; offline, the queue alone is a floor.
    func refreshCount(jobId: String, settings: AppSettings, uploads: PendingUploads) async {
        let queued = uploads.pending(forJob: jobId)
        if jobId.hasPrefix("local-") { toolsInJob = queued; return }
        if let tools = try? await LibraryClient(base: settings.apiBase).listTools(jobId: jobId) {
            toolsInJob = tools.count + uploads.pending(forJob: jobId)
        } else {
            toolsInJob = max(toolsInJob, queued)
        }
    }

    // MARK: - Pure helpers (off the main actor)

    nonisolated static func newToolId() -> String {
        var g = SystemRandomNumberGenerator()
        return String(format: "%012llx", g.next() & 0xFFFF_FFFF_FFFF)
    }

    nonisolated static func makeWorkImage(_ s: CaptureController.StillCapture) -> (CGImage?, CameraIntrinsics) {
        let img = s.image
        let long = max(img.width, img.height)
        guard long > workSide else { return (img, s.intrinsics) }
        let f = Double(workSide) / Double(long)
        let w = max(1, Int((Double(img.width) * f).rounded())), h = max(1, Int((Double(img.height) * f).rounded()))
        return (resized(img, width: w, height: h), s.intrinsics.scaled(toWidth: w, height: h))
    }

    nonisolated static func resized(_ img: CGImage, width w: Int, height h: Int) -> CGImage? {
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    nonisolated static func jpeg(_ img: CGImage, longestSide: Int, quality: Double) -> Data? {
        let long = max(img.width, img.height)
        var out = img
        if long > longestSide {
            let f = Double(longestSide) / Double(long)
            guard let r = resized(img, width: Int((Double(img.width) * f).rounded()), height: Int((Double(img.height) * f).rounded())) else { return nil }
            out = r
        }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, out, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }

    nonisolated static func gradient(of img: CGImage) -> [Float]? {
        let w = img.width, h = img.height
        var gray = [UInt8](repeating: 0, count: w * h)
        let ok: Bool = gray.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? OutlineEngine.gradientMagnitude(gray: gray, width: w, height: h) : nil
    }

    /// Vision's foreground instance mask (iOS 17): the instance under the image centre, else the largest, at image size.
    nonisolated static func foregroundMask(_ img: CGImage) -> BitMask? {
        let req = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: img, orientation: .up, options: [:])
        do { try handler.perform([req]) } catch { return nil }
        guard let obs = req.results?.first, !obs.allInstances.isEmpty else { return nil }
        // pick the instance from the low-res label map
        let labels = obs.instanceMask
        CVPixelBufferLockBaseAddress(labels, .readOnly)
        let lw = CVPixelBufferGetWidth(labels), lh = CVPixelBufferGetHeight(labels), lrb = CVPixelBufferGetBytesPerRow(labels)
        var chosen: Int?
        if let base = CVPixelBufferGetBaseAddress(labels)?.assumingMemoryBound(to: UInt8.self) {
            let c = Int(base[(lh / 2) * lrb + lw / 2])
            if c != 0, obs.allInstances.contains(c) { chosen = c }
            if chosen == nil {
                var counts = [Int](repeating: 0, count: 256)
                for y in 0..<lh { for x in 0..<lw { counts[Int(base[y * lrb + x])] += 1 } }
                chosen = obs.allInstances.max { counts[min(255, $0)] < counts[min(255, $1)] }
            }
        }
        CVPixelBufferUnlockBaseAddress(labels, .readOnly)
        guard let inst = chosen,
              let buf = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: inst), from: handler) else { return nil }
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let bw = CVPixelBufferGetWidth(buf), bh = CVPixelBufferGetHeight(buf), rb = CVPixelBufferGetBytesPerRow(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf), CVPixelBufferGetPixelFormatType(buf) == kCVPixelFormatType_OneComponent32Float else { return nil }
        let W = img.width, H = img.height
        var m = BitMask(width: W, height: H)
        for y in 0..<H {
            let sy = bh == H ? y : min(bh - 1, y * bh / H)
            let row = base.advanced(by: sy * rb).assumingMemoryBound(to: Float.self)
            for x in 0..<W {
                let sx = bw == W ? x : min(bw - 1, x * bw / W)
                if row[sx] > 0.5 { m.data[y * W + x] = 1 }
            }
        }
        return m.count > 0 ? m : nil
    }

    /// Fit the surface the tool lies on: depth pixels whose RAY lands on the tool in the photo are excluded (mapped
    /// through both intrinsics — same camera centre — so a different photo / depth field of view does not matter),
    /// dilated by 2 depth pixels.
    nonisolated static func fitScalePlane(_ s: CaptureController.StillCapture, workK: CameraIntrinsics, toolMask: BitMask?) -> Plane? {
        guard let depth = s.depth, let dK = s.depthIntrinsics, s.depthWidth > 0, s.depthHeight > 0 else { return nil }
        let dw = s.depthWidth, dh = s.depthHeight
        var exclude: BitMask?
        if let m = toolMask {
            var ex = BitMask(width: dw, height: dh)
            for y in 0..<dh {
                for x in 0..<dw {
                    let z = Double(depth[y * dw + x])
                    guard z.isFinite, z > 0 else { continue }
                    let X = (Double(x) - dK.cx) / dK.fx, Y = (Double(y) - dK.cy) / dK.fy
                    // integer-centre photo coordinate -> mask pixel floor(u + 0.5)
                    let u = Int(floor(workK.fx * X + workK.cx + 0.5)), v = Int(floor(workK.fy * Y + workK.cy + 0.5))
                    if u >= 0, v >= 0, u < m.width, v < m.height, m[u, v] != 0 { ex[x, y] = 1 }
                }
            }
            var dil = ex
            for y in 0..<dh { for x in 0..<dw where ex[x, y] != 0 {
                for yy in max(0, y - 2)...min(dh - 1, y + 2) { for xx in max(0, x - 2)...min(dw - 1, x + 2) { dil[xx, yy] = 1 } }
            } }
            exclude = dil
        }
        return OutlineEngine.fitPlane(depth: depth, width: dw, height: dh, intrinsics: dK, exclude: exclude)
    }
}

private func round1(_ v: Double) -> Double { (v * 10).rounded() / 10 }
private func round2(_ v: Double) -> Double { (v * 100).rounded() / 100 }
private func round4(_ v: Double) -> Double { (v * 10000).rounded() / 10000 }
