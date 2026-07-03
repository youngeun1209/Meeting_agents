import Foundation
import ScreenCaptureKit
import AVFoundation
import CoreMedia

// Captures the whole system audio (output of every app) and streams 16kHz mono int16 PCM to stdout.
// Captured directly via ScreenCaptureKit (macOS 13+), no BlackHole / Multi-Output Device.
// The Python side (audio_capture.SystemAudioCapture) reads stdout and pushes it onto audio_queue.

func err(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

@available(macOS 13.0, *)
final class SystemAudioCapturer: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let out = FileHandle.standardOutput
    private var loggedFormat = false

    func start() async throws {
        // Whole-display filter -> audio of every app on it is captured together.
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            err("no available display"); exit(1)
        }
        let filter = SCContentFilter(display: display,
                                     excludingApplications: [],
                                     exceptingWindows: [])

        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.sampleRate = 16000       // Whisper expects this. Verify the actual value via the stderr log.
        cfg.channelCount = 1         // request mono
        // Only audio is needed -> minimize video size (minimize overhead).
        cfg.width = 2
        cfg.height = 2
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        cfg.queueDepth = 6

        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .audio,
                              sampleHandlerQueue: DispatchQueue(label: "sysaudio.audio"))
        try await s.startCapture()
        self.stream = s
        err("system audio capture started (requested: 16kHz mono)")
    }

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferDataIsReady(sampleBuffer) else { return }
        guard let fmtDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmtDesc)?.pointee
        else { return }

        if !loggedFormat {
            loggedFormat = true
            // If the actual sample rate isn't 16000 (e.g. 48000) it won't match the Python pipeline -> warn.
            err("input format: \(Int(asbd.mSampleRate))Hz, \(asbd.mChannelsPerFrame)ch")
            if Int(asbd.mSampleRate) != 16000 {
                err("warning: actual \(Int(asbd.mSampleRate))Hz. Mismatch with config.SAMPLE_RATE -> resampling needed.")
            }
        }

        var blockBuffer: CMBlockBuffer?
        var abl = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer)
        guard status == noErr else { return }

        let channels = Int(asbd.mChannelsPerFrame)
        let list = UnsafeMutableAudioBufferListPointer(&abl)

        // ScreenCaptureKit gives float32 PCM. If channels > 1, downmix to mono.
        var mono: [Float32] = []
        if list.count == 1 {
            // interleaved (mono or stereo-interleaved)
            let b = list[0]
            guard let base = b.mData else { return }
            let total = Int(b.mDataByteSize) / MemoryLayout<Float32>.size
            let ptr = base.bindMemory(to: Float32.self, capacity: total)
            if channels <= 1 {
                mono = Array(UnsafeBufferPointer(start: ptr, count: total))
            } else {
                let frames = total / channels
                mono = [Float32](repeating: 0, count: frames)
                for f in 0..<frames {
                    var acc: Float32 = 0
                    for c in 0..<channels { acc += ptr[f * channels + c] }
                    mono[f] = acc / Float32(channels)
                }
            }
        } else {
            // non-interleaved: one buffer per channel -> average the channels
            let frames = Int(list[0].mDataByteSize) / MemoryLayout<Float32>.size
            mono = [Float32](repeating: 0, count: frames)
            var used = 0
            for b in list {
                guard let base = b.mData else { continue }
                let ptr = base.bindMemory(to: Float32.self, capacity: frames)
                for f in 0..<frames { mono[f] += ptr[f] }
                used += 1
            }
            if used > 1 { for f in 0..<frames { mono[f] /= Float32(used) } }
        }

        // float32 [-1,1] -> int16 PCM
        var pcm16 = [Int16](repeating: 0, count: mono.count)
        for i in 0..<mono.count {
            let v = max(-1.0, min(1.0, mono[i]))
            pcm16[i] = Int16(v * 32767.0)
        }
        pcm16.withUnsafeBytes { out.write(Data($0)) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        err("stream stopped: \(error)")
        exit(1)
    }
}

if #available(macOS 13.0, *) {
    let cap = SystemAudioCapturer()
    Task {
        do { try await cap.start() }
        catch { err("failed to start: \(error)"); exit(1) }
    }
    signal(SIGINT)  { _ in exit(0) }
    signal(SIGTERM) { _ in exit(0) }
    RunLoop.main.run()
} else {
    err("macOS 13.0+ required")
    exit(1)
}
