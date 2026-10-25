import Foundation

/// One accelerometer reading. `timestamp` is seconds since session start on the monotonic clock;
/// acceleration is in g along the device axes, exactly as reported by Core Motion.
struct SensorSample: Equatable {
    var timestamp: Double
    var x: Double
    var y: Double
    var z: Double

    var magnitude: Double { (x * x + y * y + z * z).squareRoot() }
}

/// File names inside a session directory and inside the exported ZIP.
enum SessionFiles {
    static let sensorData = "sensor_data.csv"
    static let stepLabels = "step_labels.csv"
    static let metadata = "session_metadata.json"
}
