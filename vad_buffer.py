"""Splits the block stream into utterance chunks on silence. Uses webrtcvad."""
import numpy as np
import webrtcvad

import config


def _float32_to_pcm16(audio):
    """float32 [-1,1] -> int16 PCM bytes. webrtcvad only accepts 16-bit PCM."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


class VADBuffer:
    """
    Accumulates audio blocks and decides speech/silence with webrtcvad.
    When an utterance ends (sustained silence) it returns the finished chunk (float32).
    """

    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.sample_rate = config.SAMPLE_RATE
        self.frame_len = int(self.sample_rate * config.VAD_FRAME_MS / 1000)  # sample count
        self.silence_frames_limit = int(
            config.VAD_SILENCE_SEC * 1000 / config.VAD_FRAME_MS
        )
        self.max_chunk_samples = int(config.MAX_CHUNK_SEC * self.sample_rate)
        self.min_chunk_samples = int(config.MIN_CHUNK_SEC * self.sample_rate)

        self._residual = np.empty(0, dtype=np.float32)  # leftover shorter than a frame
        self._speech = []        # current utterance accumulator (list of frames)
        self._silence_run = 0    # consecutive silent frames
        self._in_speech = False

    def add(self, block):
        """
        Feed one audio block (float32 1D).
        Returns a list of finished utterance chunks (empty list if none).
        """
        chunks = []
        buf = np.concatenate([self._residual, block])
        n_frames = len(buf) // self.frame_len

        for i in range(n_frames):
            frame = buf[i * self.frame_len:(i + 1) * self.frame_len]
            is_speech = self.vad.is_speech(_float32_to_pcm16(frame), self.sample_rate)

            if is_speech:
                self._speech.append(frame)
                self._silence_run = 0
                self._in_speech = True
            else:
                if self._in_speech:
                    self._speech.append(frame)  # include a bit of trailing silence
                    self._silence_run += 1
                    if self._silence_run >= self.silence_frames_limit:
                        done = self._flush()
                        if done is not None:
                            chunks.append(done)

            # Force-split if the utterance gets too long
            if self._in_speech and self._speech_len() >= self.max_chunk_samples:
                done = self._flush()
                if done is not None:
                    chunks.append(done)

        # Keep the leftover that doesn't fill a frame
        self._residual = buf[n_frames * self.frame_len:]
        return chunks

    def _speech_len(self):
        return sum(len(f) for f in self._speech)

    def _flush(self):
        """Finalize the accumulated utterance into a chunk. None if too short."""
        if not self._speech:
            self._reset_speech()
            return None
        audio = np.concatenate(self._speech)
        self._reset_speech()
        if len(audio) < self.min_chunk_samples:
            return None
        return audio

    def _reset_speech(self):
        self._speech = []
        self._silence_run = 0
        self._in_speech = False

    def finalize(self):
        """On shutdown, return the trailing utterance as a final chunk (None if none)."""
        return self._flush()
