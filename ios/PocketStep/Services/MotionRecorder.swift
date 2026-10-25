import CoreMotion
import Foundation
import UIKit

/// Source of raw accelerometer readings. Timestamps are device uptime in seconds.
protocol AccelerometerSource: AnyObject {
    var isAvailable: Bool { get }
    func start(interval: TimeInterval, queue: OperationQueue,
               handler: @escaping (Result<SensorSample, Error>) -> Void)
    func stop()
}

final class CoreMotionAccelerometer: AccelerometerSource {
    private let manager = CMMotionManager()

    var isAvailable: Bool { manager.isAccelerometerAvailable }

    func start(interval: TimeInterval, queue: OperationQueue,
               handler: @escaping (Result<SensorSample, Error>) -> Void) {
        manager.accelerometerUpdateInterval = interval
        manager.startAccelerometerUpdates(to: queue) { data, error in
            if let error {
                handler(.failure(error))
                return
            }
            guard let data else { return }
            let a = data.acceleration
            handler(.success(SensorSample(timestamp: data.timestamp, x: a.x, y: a.y, z: a.z)))
        }
    }

    func stop() {
        manager.stopAccelerometerUpdates()
    }
}

/// Thread-safe bridge between the sensor queue and the CSV file.
final class RecordingSink: @unchecked Sendable {
    private let lock = NSLock()
    private let writer: SensorCSVWriter
    private let startUptime: Double
    private var stats: SamplingStatistics
    private var lastTimestamp = -Double.infinity
    private var failed = false

    init(writer: SensorCSVWriter, startUptime: Double, requestedInterval: Double) {
        self.writer = writer
        self.startUptime = startUptime
        self.stats = SamplingStatistics(requestedInterval: requestedInterval)
    }

    /// Converts an uptime-stamped reading to session time and writes it.
    /// Returns an error only the first time writing fails; later readings are ignored.
    func receive(_ raw: SensorSample) -> Error? {
        let t = raw.timestamp - startUptime
        lock.lock()
        defer { lock.unlock() }
        // Readings from before the session start or out of order are not written.
        guard !failed, t >= 0, t > lastTimestamp else { return nil }
        do {
            try writer.append(SensorSample(timestamp: t, x: raw.x, y: raw.y, z: raw.z))
            stats.add(timestamp: t)
            lastTimestamp = t
            return nil
        } catch {
            failed = true
            return error
        }
    }

    /// Flushes buffered rows and marks a deliberate break so the pause is not counted as a gap.
    func breakContinuity() {
        lock.lock()
        defer { lock.unlock() }
        try? writer.flush()
        stats.breakContinuity()
    }

    func close() throws {
        lock.lock()
        defer { lock.unlock() }
        try writer.close()
    }

    var statistics: SamplingStatistics {
        lock.lock()
        defer { lock.unlock() }
        return stats
    }
}

enum RecorderState: Equatable {
    case ready
    case countingDown(remaining: Int)
    case recording
    case paused
    case finished
    case failed(String)
}

@MainActor
final class MotionRecorder: ObservableObject {
    @Published private(set) var state: RecorderState = .ready
    @Published private(set) var metadata: SessionMetadata
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var sampleCount = 0
    @Published private(set) var measuredRateHz: Double?
    @Published private(set) var gapCount = 0
    @Published private(set) var pedometerSteps: Int?
    @Published private(set) var pedometerCadence: Double?
    @Published private(set) var pedometerMessage: String?
    @Published private(set) var warning: String?
    /// True once a session folder exists on disk, even if recording later failed.
    @Published private(set) var hasSavedData = false

    private let store: SessionStore
    private let accelerometer: AccelerometerSource
    private let pedometer = CMPedometer()
    private let sensorQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "PocketStep.sensor"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        return queue
    }()
    private var sink: RecordingSink?
    private var startUptime: Double = 0
    private var countdownTask: Task<Void, Never>?
    private var tickTimer: Timer?
    private var lastCheckpoint: Double = 0

    init(metadata: SessionMetadata, store: SessionStore,
         accelerometer: AccelerometerSource = CoreMotionAccelerometer()) {
        self.metadata = metadata
        self.store = store
        self.accelerometer = accelerometer
    }

    /// True while the researcher should not navigate away.
    var isBusy: Bool {
        switch state {
        case .countingDown, .recording, .paused: return true
        default: return false
        }
    }

    private var currentElapsed: Double {
        ProcessInfo.processInfo.systemUptime - startUptime
    }

    // MARK: Permissions

    /// Raw accelerometer access needs no permission; CMPedometer needs Motion & Fitness access.
    func requestMotionPermissionIfNeeded() {
        guard CMPedometer.isStepCountingAvailable() else {
            pedometerMessage = "Step counting is not available on this device. Raw accelerometer recording is unaffected."
            return
        }
        if CMPedometer.authorizationStatus() == .notDetermined {
            let now = Date()
            pedometer.queryPedometerData(from: now.addingTimeInterval(-60), to: now) { [weak self] _, _ in
                Task { @MainActor in self?.updatePedometerAuthorizationMessage() }
            }
        } else {
            updatePedometerAuthorizationMessage()
        }
    }

    private func updatePedometerAuthorizationMessage() {
        switch CMPedometer.authorizationStatus() {
        case .denied, .restricted:
            pedometerMessage = "Motion & Fitness access is off, so pedometer reference counts are unavailable. Raw accelerometer recording still works. To enable it, open Settings > Privacy & Security > Motion & Fitness."
        default:
            pedometerMessage = nil
        }
    }

    // MARK: Controls

    func start(countdownSeconds: Int) {
        guard state == .ready else { return }
        guard accelerometer.isAvailable else {
            state = .failed("No accelerometer is available. Recording needs a physical iPhone; the Simulator has no motion sensors.")
            return
        }
        setScreenGuards(enabled: true)
        guard countdownSeconds > 0 else {
            beginRecording()
            return
        }
        state = .countingDown(remaining: countdownSeconds)
        countdownTask = Task { [weak self] in
            for remaining in stride(from: countdownSeconds - 1, through: 0, by: -1) {
                try? await Task.sleep(for: .seconds(1))
                guard let self, !Task.isCancelled else { return }
                if remaining == 0 {
                    self.beginRecording()
                } else {
                    self.state = .countingDown(remaining: remaining)
                }
            }
        }
    }

    func cancelCountdown() {
        guard case .countingDown = state else { return }
        countdownTask?.cancel()
        countdownTask = nil
        setScreenGuards(enabled: false)
        state = .ready
    }

    func pause(reason: String = "Paused by researcher") {
        guard state == .recording else { return }
        accelerometer.stop()
        metadata.pauses.append(PauseInterval(startSeconds: currentElapsed, endSeconds: nil, reason: reason))
        sensorQueue.addOperation { [sink] in sink?.breakContinuity() }
        state = .paused
        checkpoint()
    }

    func resume() {
        guard state == .paused else { return }
        if let last = metadata.pauses.indices.last, metadata.pauses[last].endSeconds == nil {
            metadata.pauses[last].endSeconds = currentElapsed
        }
        warning = nil
        startSensor()
        state = .recording
        checkpoint()
    }

    func stop() {
        guard state == .recording || state == .paused else { return }
        closeOpenPause()
        finish(status: .completed, failureMessage: nil)
    }

    /// The accelerometer stops delivering data when the app is backgrounded, so pause explicitly
    /// and log it instead of leaving an unexplained gap.
    func handleBackgrounding() {
        switch state {
        case .countingDown:
            cancelCountdown()
        case .recording:
            metadata.interruptions.append(InterruptionEvent(
                atSeconds: currentElapsed,
                reason: "App moved to the background; recording paused automatically"))
            pause(reason: "App moved to the background")
            warning = "Recording paused because the app left the foreground (for example, the side button was pressed). Check that the phone is still in the same position, then resume or stop."
        default:
            break
        }
    }

    // MARK: Internals

    private func beginRecording() {
        countdownTask = nil
        do {
            _ = try store.createDirectory(for: metadata)
            let writer = try SensorCSVWriter(url: store.fileURL(SessionFiles.sensorData, for: metadata.sessionId))
            startUptime = ProcessInfo.processInfo.systemUptime
            metadata.sessionStart = Date()
            metadata.iosVersion = DeviceInfo.iosVersion
            metadata.recordingStatus = .inProgress
            sink = RecordingSink(writer: writer, startUptime: startUptime,
                                 requestedInterval: metadata.requestedSamplingIntervalSeconds)
            try store.save(metadata)
            hasSavedData = true
        } catch {
            setScreenGuards(enabled: false)
            state = .failed("Could not create the session files: \(error.localizedDescription)")
            return
        }
        startSensor()
        startPedometer()
        state = .recording
        startTicking()
    }

    private func startSensor() {
        guard let sink else { return }
        accelerometer.start(interval: metadata.requestedSamplingIntervalSeconds, queue: sensorQueue) { [weak self] result in
            switch result {
            case .success(let raw):
                if let error = sink.receive(raw) {
                    Task { @MainActor in
                        self?.fail("Writing sensor data failed: \(error.localizedDescription). Samples saved before this point are kept.")
                    }
                }
            case .failure(let error):
                Task { @MainActor in
                    self?.fail("The accelerometer reported an error: \(error.localizedDescription). Samples saved before this point are kept.")
                }
            }
        }
    }

    private func startPedometer() {
        guard CMPedometer.isStepCountingAvailable() else { return }
        let status = CMPedometer.authorizationStatus()
        guard status != .denied, status != .restricted else { return }
        pedometer.startUpdates(from: Date()) { [weak self] data, error in
            Task { @MainActor in
                guard let self else { return }
                if let error {
                    self.pedometerMessage = "Pedometer error: \(error.localizedDescription). Raw recording continues."
                    return
                }
                guard let data else { return }
                self.pedometerSteps = data.numberOfSteps.intValue
                self.pedometerCadence = data.currentCadence?.doubleValue
            }
        }
    }

    private func fail(_ message: String) {
        guard state == .recording || state == .paused else { return }
        metadata.interruptions.append(InterruptionEvent(atSeconds: currentElapsed, reason: message))
        closeOpenPause()
        finish(status: .failed, failureMessage: message)
    }

    private func closeOpenPause() {
        if let last = metadata.pauses.indices.last, metadata.pauses[last].endSeconds == nil {
            metadata.pauses[last].endSeconds = currentElapsed
        }
    }

    private func finish(status: RecordingStatus, failureMessage: String?) {
        accelerometer.stop()
        pedometer.stopUpdates()
        stopTicking()
        setScreenGuards(enabled: false)
        sensorQueue.waitUntilAllOperationsAreFinished()

        var message = failureMessage
        var finalStatus = status
        if let sink {
            do {
                try sink.close()
            } catch {
                finalStatus = .failed
                message = message ?? "Could not finish writing the sensor file: \(error.localizedDescription)"
            }
            let stats = sink.statistics
            metadata.sampleCount = stats.sampleCount
            metadata.samplingIntervalSeconds = stats.medianInterval
            metadata.samplingRateHzEstimate = stats.estimatedRateHz
            metadata.unexpectedGapCount = stats.gapCount
        }
        metadata.sessionEnd = Date()
        metadata.pedometerSteps = pedometerSteps
        metadata.recordingStatus = finalStatus
        do {
            try store.save(metadata)
            try StepLabelStore.createEmptyIfMissing(at: store.fileURL(SessionFiles.stepLabels, for: metadata.sessionId))
        } catch {
            message = message ?? "Could not save the session metadata: \(error.localizedDescription)"
        }
        sink = nil
        refreshLiveStatistics()
        state = message.map { .failed($0) } ?? .finished
    }

    private func startTicking() {
        lastCheckpoint = 0
        tickTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func stopTicking() {
        tickTimer?.invalidate()
        tickTimer = nil
    }

    private func tick() {
        elapsedSeconds = currentElapsed
        refreshLiveStatistics()
        if elapsedSeconds - lastCheckpoint >= 5 {
            checkpoint()
        }
    }

    private func refreshLiveStatistics() {
        guard let stats = sink?.statistics else { return }
        sampleCount = stats.sampleCount
        measuredRateHz = stats.estimatedRateHz
        gapCount = stats.gapCount
    }

    /// Saves metadata periodically so a terminated session still has a usable description.
    private func checkpoint() {
        lastCheckpoint = currentElapsed
        if let stats = sink?.statistics {
            metadata.sampleCount = stats.sampleCount
            metadata.unexpectedGapCount = stats.gapCount
        }
        try? store.save(metadata)
    }

    /// Keeps the app in the foreground while the phone is pocketed: no auto-lock, and the
    /// proximity sensor turns the screen off so the pocket cannot press buttons.
    private func setScreenGuards(enabled: Bool) {
        UIApplication.shared.isIdleTimerDisabled = enabled
        UIDevice.current.isProximityMonitoringEnabled = enabled
    }
}
