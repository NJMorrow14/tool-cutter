import CoreMedia
import Foundation

@main struct TimingTests {
    static func main() {
        typealias Range = CaptureTiming.Range
        let thirty = CMTime(value: 1, timescale: 30)
        let range = Range(minimum: thirty, maximum: CMTime(value: 1, timescale: 15))
        precondition(!range.contains(CMTime(seconds: 1.0 / 30, preferredTimescale: 1_000_000)), "Reproduce the old rounded duration failure")
        let exact = CaptureTiming.duration(video: [range], depth: [range])!
        precondition(CMTimeCompare(exact, thirty) == 0)
        let fractional = Range(minimum: CMTime(value: 1001, timescale: 30_000), maximum: CMTime(value: 1, timescale: 15))
        precondition(CMTimeCompare(CaptureTiming.duration(video: [range], depth: [fractional])!, fractional.minimum) == 0)
        let fixed = Range(minimum: thirty, maximum: thirty)
        precondition(CMTimeCompare(CaptureTiming.duration(video: [fixed], depth: [range])!, thirty) == 0)
        let slow = Range(minimum: CMTime(value: 1, timescale: 10), maximum: CMTime(value: 1, timescale: 5))
        precondition(CaptureTiming.duration(video: [fixed], depth: [slow]) == nil)
        precondition(CaptureTiming.duration(video: [], depth: [range]) == nil)
        let invalid = Range(minimum: .invalid, maximum: .invalid)
        precondition(CaptureTiming.duration(video: [invalid], depth: [range]) == nil)
        print("All 7 capture timing checks passed")
    }
}
