import Foundation
import CoreGraphics

struct CaptureIntrinsics: Codable {
    var fx: Double, fy: Double, cx: Double, cy: Double
    var width: Int, height: Int
}

struct DepthPayload {
    var data: Data          // row-major; float32 LE metres ("f32", default) or uint16 millimetres ("u2mm")
    var width: Int
    var height: Int
    var dtype: String = "f32"
}

struct SessionInfo: Decodable {
    struct Size: Decodable { var width: Double; var height: Double }
    struct Rect: Decodable { var width: Int; var height: Int; var mm_per_px: Double; var has_height: Bool }
    var id: String
    var source_kind: String
    var auto_calibrated: Bool?
    var mat_mm: Size?
    var rectified: Rect?
    var scan: [String: AnyCodableValue]?
}

/// Minimal JSON value wrapper so we can read loosely-typed metadata.
enum AnyCodableValue: Decodable {
    case double(Double), string(String), bool(Bool), array([AnyCodableValue]), null
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let d = try? c.decode(Double.self) { self = .double(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([AnyCodableValue].self) { self = .array(a); return }
        self = .null
    }
    var intValue: Int? { if case .double(let d) = self { return Int(d) } else { return nil } }
    var count: Int? { if case .array(let a) = self { return a.count } else { return nil } }
}

enum APIError: LocalizedError {
    case server(String)
    case transport(String)
    var errorDescription: String? {
        switch self {
        case .server(let m): return m
        case .transport(let m): return m
        }
    }
}

struct JobInfo: Decodable {
    var id: String
    var status: String          // queued | running | done | error
    var progress: Double?
    var message: String?
    var session_id: String?
    var error: String?
}

struct APIClient {
    let base: URL

    /// POST /api/sweeps: all frames + manifest in one multipart body; returns a job to poll.
    enum SweepMode { case lidarFusion, photogrammetry }

    /// Multipart body shared by /api/sweeps (photogrammetry job) and /api/captures/multi (LiDAR fusion, immediate).
    private func sweepBody(frames: [SweepFrame], markerSizeMm: Double, insetMm: Double, detail: String, filename: String,
                           drawerSize: (Double, Double)?, boundary: String) throws -> Data {
        var mp = MultipartBody(boundary: boundary)
        func field(_ name: String, _ value: String) { mp.field(name, value) }
        func file(_ name: String, filename: String, type: String, data: Data) { mp.file(name, filename: filename, type: type, data: data) }
        var manifest: [[String: Any]] = []
        for (i, f) in frames.enumerated() {
            let img = String(format: "frame_%03d.jpg", i)
            var entry: [String: Any] = ["image": img, "sensor": f.sensor, "use": f.use, "pose_quality": f.poseQuality, "timestamp": f.timestamp,
                                        "intrinsics": ["fx": f.intrinsics.fx, "fy": f.intrinsics.fy, "cx": f.intrinsics.cx, "cy": f.intrinsics.cy,
                                                       "width": f.intrinsics.width, "height": f.intrinsics.height]]
            if let m = f.transform {
                let tf: [Float] = [m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
                               m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
                               m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
                               m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w]
                entry["transform"] = tf
                entry["transform_units"] = "m"
            }
            if let g = f.gravity { entry["gravity"] = [g.x, g.y, g.z] }
            if let lens = f.lens { entry["lens_calibration"] = lens.manifest }
            file(img, filename: img, type: "image/jpeg", data: f.jpeg)
            if let d = f.depth {
                let dn = String(format: "depth_%03d.f32", i)
                entry["depth"] = dn
                entry["depth_width"] = d.width
                entry["depth_height"] = d.height
                entry["depth_dtype"] = d.dtype          // "f32" (metres) or "u2mm" (uint16 millimetres)
                file(dn, filename: dn, type: "application/octet-stream", data: d.data)
            }
            manifest.append(entry)
        }
        let manifestData = try JSONSerialization.data(withJSONObject: ["frames": manifest])
        file("manifest", filename: "manifest.json", type: "application/json", data: manifestData)
        field("marker_size_mm", String(markerSizeMm))
        field("inset_mm", String(insetMm))
        field("detail", detail)
        field("filename", filename)
        if let (w, h) = drawerSize {
            field("drawer_width_mm", String(w))
            field("drawer_height_mm", String(h))
        }
        return mp.finished()
    }

    /// LiDAR fusion of an arc of frames: returns the finished session right away.
    struct MarkerCheck: Decodable { let ids: [Int]; let count: Int; let adjacent_pair: Bool; let ok: Bool }

    /// Which corner markers the server sees in one small frame. Used before a sweep starts: the drawer can only be
    /// measured exactly from a view with 3+ markers, and every failed scan so far began without one.
    func markers(jpeg: Data) async throws -> MarkerCheck {
        var req = URLRequest(url: base.appendingPathComponent("api/markers"))
        req.httpMethod = "POST"
        req.timeoutInterval = 8
        let boundary = "ToolCutter-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"preview.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpeg)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        let (data, resp): (Data, URLResponse)
        do { (data, resp) = try await URLSession.shared.upload(for: req, from: body) }
        catch { throw APIError.transport("Could not reach the ToolCutter server: \(error.localizedDescription)") }
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let msg = obj["error"] as? String { throw APIError.server(msg) }
            throw APIError.server("Server returned HTTP \((resp as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        return try JSONDecoder().decode(MarkerCheck.self, from: data)
    }

    func uploadLidarArc(frames: [SweepFrame], markerSizeMm: Double, insetMm: Double, drawerSize: (Double, Double)?) async throws -> SessionInfo {
        if frames.contains(where: { $0.sensor == "truedepth" }) {
            var check = URLRequest(url: base.appendingPathComponent("health"))
            check.timeoutInterval = 10
            let (data, response) = try await URLSession.shared.data(for: check)
            let health = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  (health?["capture_capabilities"] as? [String])?.contains("truedepth_lens_v1") == true else {
                throw APIError.server("Update and restart the ToolCutter server before uploading a TrueDepth scan.")
            }
        }
        var req = URLRequest(url: base.appendingPathComponent("api/captures/multi"))
        req.httpMethod = "POST"
        req.timeoutInterval = 600
        let boundary = "ToolCutter-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let body = try sweepBody(frames: frames, markerSizeMm: markerSizeMm, insetMm: insetMm, detail: "medium", filename: "arc.jpg",
                                 drawerSize: drawerSize, boundary: boundary)
        let (data, resp): (Data, URLResponse)
        do { (data, resp) = try await URLSession.shared.upload(for: req, from: body) }
        catch { throw APIError.transport("Could not reach the ToolCutter server: \(error.localizedDescription)") }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if status != 201 {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let msg = obj["error"] as? String { throw APIError.server(msg) }
            throw APIError.server("Server returned HTTP \(status)")
        }
        return try JSONDecoder().decode(SessionInfo.self, from: data)
    }

    func uploadSweep(frames: [SweepFrame], markerSizeMm: Double, insetMm: Double, detail: String = "medium",
                     filename: String = "sweep.obj", drawerSize: (Double, Double)? = nil) async throws -> JobInfo {
        var req = URLRequest(url: base.appendingPathComponent("api/sweeps"))
        req.httpMethod = "POST"
        req.timeoutInterval = 600
        let boundary = "ToolCutter-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let body = try sweepBody(frames: frames, markerSizeMm: markerSizeMm, insetMm: insetMm, detail: detail, filename: filename,
                                 drawerSize: drawerSize, boundary: boundary)
        let (data, resp): (Data, URLResponse)
        do { (data, resp) = try await URLSession.shared.upload(for: req, from: body) }
        catch { throw APIError.transport("Could not reach the ToolCutter server: \(error.localizedDescription)") }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if status != 202 {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let msg = obj["error"] as? String { throw APIError.server(msg) }
            throw APIError.server("Server returned HTTP \(status)")
        }
        return try JSONDecoder().decode(JobInfo.self, from: data)
    }

    func job(_ id: String) async throws -> JobInfo {
        let (data, _) = try await URLSession.shared.data(from: base.appendingPathComponent("api/jobs/\(id)"))
        return try JSONDecoder().decode(JobInfo.self, from: data)
    }

    func health() async throws -> Bool {
        let (_, resp) = try await URLSession.shared.data(from: base.appendingPathComponent("health"))
        return (resp as? HTTPURLResponse)?.statusCode == 200
    }

    /// POST /api/captures with the photo, optional depth map and intrinsics.
    func uploadCapture(jpeg: Data, depth: DepthPayload?, intrinsics: CaptureIntrinsics?, markerSizeMm: Double,
                       insetMm: Double, progress: ((Double) -> Void)? = nil) async throws -> SessionInfo {
        var req = URLRequest(url: base.appendingPathComponent("api/captures"))
        req.httpMethod = "POST"
        req.timeoutInterval = 180
        let boundary = "ToolCutter-\(UUID().uuidString)"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var mp = MultipartBody(boundary: boundary)
        func field(_ name: String, _ value: String) { mp.field(name, value) }
        func file(_ name: String, filename: String, type: String, data: Data) { mp.file(name, filename: filename, type: type, data: data) }
        field("marker_size_mm", String(markerSizeMm))
        field("inset_mm", String(insetMm))
        if let k = intrinsics, let j = try? JSONEncoder().encode(k), let s = String(data: j, encoding: .utf8) {
            field("intrinsics", s)
        }
        if let d = depth {
            field("depth_width", String(d.width))
            field("depth_height", String(d.height))
            file("depth", filename: "depth.f32", type: "application/octet-stream", data: d.data)
        }
        file("image", filename: "capture.jpg", type: "image/jpeg", data: jpeg)
        let body = mp.finished()
        let (data, resp): (Data, URLResponse)
        do {
            (data, resp) = try await URLSession.shared.upload(for: req, from: body)
        } catch {
            throw APIError.transport("Could not reach the ToolCutter server at \(base.absoluteString): \(error.localizedDescription)")
        }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if status != 201 {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let msg = obj["error"] as? String {
                throw APIError.server(msg)
            }
            throw APIError.server("Server returned HTTP \(status)")
        }
        return try JSONDecoder().decode(SessionInfo.self, from: data)
    }
}

// MARK: - Multipart

/// multipart/form-data builder shared by every upload (it used to be three copies of the same two helpers).
struct MultipartBody {
    let boundary: String
    private(set) var data = Data()
    init(boundary: String = "ToolCutter-\(UUID().uuidString)") { self.boundary = boundary }
    var contentType: String { "multipart/form-data; boundary=\(boundary)" }
    mutating func field(_ name: String, _ value: String) {
        data.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
    }
    mutating func file(_ name: String, filename: String, type: String, data payload: Data) {
        data.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\nContent-Type: \(type)\r\n\r\n".data(using: .utf8)!)
        data.append(payload)
        data.append("\r\n".data(using: .utf8)!)
    }
    /// The body with its closing boundary.
    func finished() -> Data { data + "--\(boundary)--\r\n".data(using: .utf8)! }
}

// MARK: - Tool library (docs/tool-library.md)

struct LibraryJob: Decodable, Identifiable, Equatable {
    var id: String
    var name: String
    var created: Double?
    var tool_count: Int?
}

struct LibraryToolRecord: Decodable, Identifiable, Equatable {
    var id: String
    var job_id: String
    var name: String
    var thickness_mm: Double?
    var area_mm2: Double?
    var bbox_mm: [Double]?
    var thumb_url: String?
}

/// /api/library/... on the Mac: jobs (one per customer / toolboard) and the tools captured into them.
struct LibraryClient {
    let base: URL

    func listJobs() async throws -> [LibraryJob] {
        struct R: Decodable { var jobs: [LibraryJob] }
        return try await send(request(path: "api/library/jobs", method: "GET"), as: R.self).jobs
    }

    func createJob(name: String) async throws -> LibraryJob {
        var req = request(path: "api/library/jobs", method: "POST")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["name": name])
        return try await send(req, as: LibraryJob.self)
    }

    func listTools(jobId: String) async throws -> [LibraryToolRecord] {
        struct R: Decodable { var tools: [LibraryToolRecord] }
        return try await send(request(path: "api/library/jobs/\(jobId)/tools", method: "GET"), as: R.self).tools
    }

    /// POST the tool (JSON as the `tool` form field) and its photo. 201 = new, 200 = replaced the same id (a retry).
    func uploadTool(jobId: String, toolJSON: Data, photoJPEG: Data) async throws -> LibraryToolRecord {
        var req = request(path: "api/library/jobs/\(jobId)/tools", method: "POST")
        req.timeoutInterval = 60
        var mp = MultipartBody()
        req.setValue(mp.contentType, forHTTPHeaderField: "Content-Type")
        mp.field("tool", String(decoding: toolJSON, as: UTF8.self))
        mp.file("photo", filename: "photo.jpg", type: "image/jpeg", data: photoJPEG)
        req.httpBody = mp.finished()
        return try await send(req, as: LibraryToolRecord.self)
    }

    private func request(path: String, method: String) -> URLRequest {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = method
        req.timeoutInterval = 15
        return req
    }

    private func send<T: Decodable>(_ req: URLRequest, as: T.Type) async throws -> T {
        let (data, resp): (Data, URLResponse)
        do { (data, resp) = try await URLSession.shared.data(for: req) }
        catch { throw APIError.transport("Could not reach the ToolCutter server: \(error.localizedDescription)") }
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let msg = obj["error"] as? String { throw APIError.server(msg) }
            throw APIError.server("Server returned HTTP \(status)")
        }
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw APIError.server("Unexpected reply from the server: \(error.localizedDescription)") }
    }
}
