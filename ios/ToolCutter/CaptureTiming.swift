import CoreMedia

/// Compare the camera's exact rational durations, never rounded reciprocal FPS.
/// 33,333/1,000,000 is faster than 30 fps and is illegal for a 30 fps format.
enum CaptureTiming {
    struct Range {
        let minimum: CMTime
        let maximum: CMTime
        func contains(_ duration: CMTime) -> Bool {
            duration.isNumeric && CMTimeCompare(duration, minimum) >= 0 && CMTimeCompare(duration, maximum) <= 0
        }
    }

    static func duration(video: [Range], depth: [Range], preferred: CMTime = CMTime(value: 1, timescale: 30)) -> CMTime? {
        var candidates: [CMTime] = []
        for v in video {
            for d in depth {
                guard v.minimum.isNumeric, v.maximum.isNumeric, d.minimum.isNumeric, d.maximum.isNumeric else { continue }
                let minimum = CMTimeCompare(v.minimum, d.minimum) >= 0 ? v.minimum : d.minimum
                let value = CMTimeCompare(preferred, minimum) >= 0 ? preferred : minimum
                if CMTimeCompare(value, .zero) > 0 && v.contains(value) && d.contains(value) { candidates.append(value) }
            }
        }
        return candidates.min { CMTimeCompare($0, $1) < 0 }
    }
}
