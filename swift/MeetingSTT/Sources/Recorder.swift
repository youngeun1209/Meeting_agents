import AVFoundation
import Combine

/// Microphone recording + a live input level, for in-person meetings.
///
/// Records straight to m4a (AAC) rather than WAV: an hour of 16kHz mono WAV is ~115MB, and a
/// phone recording has to survive in a phone's storage. The Mac side decodes any format via
/// afconvert before transcribing, so the container choice here costs nothing downstream.
@MainActor
final class Recorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var level: Double = 0          // 0...1, for the meter
    @Published var elapsed: TimeInterval = 0
    @Published var recordings: [URL] = []
    @Published var errorMessage: String?

    private var recorder: AVAudioRecorder?
    private var timer: Timer?

    private var docsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    override init() {
        super.init()
        refreshRecordings()
    }

    func toggle() {
        isRecording ? stop() : start()
    }

    private func start() {
        // Ask before touching the session, so a denial surfaces as a message instead of silence.
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else {
                    self.errorMessage = "Microphone access denied. Enable it in Settings › Privacy › Microphone."
                    return
                }
                self.beginRecording()
            }
        }
    }

    private func beginRecording() {
        do {
            let session = AVAudioSession.sharedInstance()
            // Deliberately no .allowBluetooth: routing to a Bluetooth headset forces the HFP
            // profile, which is narrowband (8/16kHz, heavily compressed voice) and measurably
            // hurts transcription accuracy. The built-in mic is the better recording source for
            // an in-person meeting, so let the session use it.
            try session.setCategory(.playAndRecord, mode: .default)
            try session.setActive(true)

            let name = Self.timestampName()
            let url = docsURL.appendingPathComponent("\(name).m4a")
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                // 16kHz mono matches what Whisper wants, so the Mac side never has to resample up.
                AVSampleRateKey: 16000.0,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
            ]
            let rec = try AVAudioRecorder(url: url, settings: settings)
            rec.isMeteringEnabled = true
            rec.delegate = self
            guard rec.record() else {
                errorMessage = "Could not start recording."
                return
            }
            recorder = rec
            isRecording = true
            errorMessage = nil
            startTimer()
        } catch {
            errorMessage = "Recording failed: \(error.localizedDescription)"
        }
    }

    private func stop() {
        recorder?.stop()
        recorder = nil
        timer?.invalidate()
        timer = nil
        isRecording = false
        level = 0
        elapsed = 0
        try? AVAudioSession.sharedInstance().setActive(false)
        refreshRecordings()
    }

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let rec = self.recorder else { return }
                rec.updateMeters()
                // averagePower is dBFS (-160...0). Map the useful -50...0 band onto 0...1.
                let db = Double(rec.averagePower(forChannel: 0))
                self.level = max(0, min(1, (db + 50) / 50))
                self.elapsed = rec.currentTime
            }
        }
    }

    func refreshRecordings() {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: docsURL, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        recordings = files
            .filter { $0.pathExtension == "m4a" }
            .sorted { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return da > db
            }
    }

    func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        refreshRecordings()
    }

    /// Same naming shape as the Mac app's transcripts, so files from both sides sort together.
    private static func timestampName() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HH-mm"
        return "meeting_\(f.string(from: Date()))"
    }
}

extension Recorder: AVAudioRecorderDelegate {
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task { @MainActor in self.refreshRecordings() }
    }
}
