import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = Recorder()

    var body: some View {
        NavigationStack {
            VStack(spacing: 28) {
                LevelMeter(level: recorder.level, active: recorder.isRecording)
                    .frame(height: 56)
                    .padding(.horizontal)

                Text(timeString(recorder.elapsed))
                    .font(.system(size: 44, weight: .semibold, design: .monospaced))
                    .foregroundStyle(recorder.isRecording ? .primary : .secondary)
                    .contentTransition(.numericText())

                Button(action: recorder.toggle) {
                    ZStack {
                        Circle()
                            .fill(recorder.isRecording ? Color.red : Color.accentColor)
                            .frame(width: 92, height: 92)
                        Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 34, weight: .bold))
                            .foregroundStyle(.white)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(recorder.isRecording ? "Stop recording" : "Start recording")

                if let msg = recorder.errorMessage {
                    Text(msg)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                recordingsList
            }
            .padding(.top, 24)
            .navigationTitle("Meeting STT")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var recordingsList: some View {
        List {
            Section("Recordings") {
                if recorder.recordings.isEmpty {
                    Text("Nothing recorded yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(recorder.recordings, id: \.self) { url in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(url.deletingPathExtension().lastPathComponent)
                                    .font(.callout)
                                Text(sizeString(url))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            ShareLink(item: url) {
                                Image(systemName: "square.and.arrow.up")
                            }
                        }
                    }
                    .onDelete { idx in
                        idx.map { recorder.recordings[$0] }.forEach(recorder.delete)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }

    private func timeString(_ t: TimeInterval) -> String {
        String(format: "%02d:%02d", Int(t) / 60, Int(t) % 60)
    }

    private func sizeString(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

/// Simple bar meter. Driven by the recorder's averagePower, so it moves with real input —
/// which also makes it the quickest way to see that the microphone is actually live.
struct LevelMeter: View {
    let level: Double
    let active: Bool

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 3) {
                ForEach(0..<28, id: \.self) { i in
                    let threshold = Double(i) / 28.0
                    RoundedRectangle(cornerRadius: 2)
                        .fill(active && level > threshold ? barColor(threshold) : Color.secondary.opacity(0.18))
                        .frame(width: (geo.size.width - 81) / 28)
                }
            }
            .frame(height: geo.size.height)
        }
    }

    private func barColor(_ t: Double) -> Color {
        t > 0.85 ? .red : (t > 0.65 ? .orange : .accentColor)
    }
}
