import re
import string
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from marvin.config import (
    MIN_VALID_AUDIO_SECONDS,
    MIN_VALID_RMS,
    MIN_VALID_TRANSCRIPT_CHARS,
    SAMPLE_RATE,
    ACTIVE_SESSION_END_PHRASES,
    WAKE_ALLOW_BARGE_IN,
    WAKE_CHECK_LANGUAGE,
    WAKE_ENGINE,
    WAKE_MAX_UTTERANCE_SECONDS,
    WAKE_POSITIVE_WORDS,
    WAKE_PREROLL_SECONDS,
    WAKE_SILENCE_STOP_SECONDS,
)
from marvin.tts import is_speaking


WAKE_AUDIO_FRAME_SIZE = 512
VAD_MIN_RMS = 0.001
VAD_SPEECH_RATIO = 3.0
VAD_NOISE_ALPHA = 0.95
VALID_TRANSCRIPT_RE = re.compile(r"[A-Za-z0-9\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")

def is_valid_transcript(text: str) -> bool:
    normalized = (text or "").lower().strip()
    if len(normalized) < MIN_VALID_TRANSCRIPT_CHARS:
        return False
    return bool(VALID_TRANSCRIPT_RE.search(normalized))


def _invalid_transcript_reason(text, duration, rms):
    normalized = (text or "").lower().strip()
    reasons = []

    if not normalized:
        reasons.append("empty transcript")
    elif len(normalized) < MIN_VALID_TRANSCRIPT_CHARS:
        reasons.append(f"transcript shorter than {MIN_VALID_TRANSCRIPT_CHARS} chars")
    elif not VALID_TRANSCRIPT_RE.search(normalized):
        reasons.append("transcript has no letters, numbers, or Arabic characters")

    if duration < MIN_VALID_AUDIO_SECONDS:
        reasons.append(f"candidate shorter than {MIN_VALID_AUDIO_SECONDS:.2f}s")

    if rms < MIN_VALID_RMS:
        reasons.append(f"candidate RMS below {MIN_VALID_RMS:.4f}")

    return "; ".join(reasons) if reasons else "unknown"


class WakeWordListener(QThread):
    status_changed = Signal(str)
    utterance_accepted = Signal(str)
    utterance_rejected = Signal()
    session_ended = Signal()
    error = Signal(str)

    def __init__(self, stt=None):
        super().__init__()
        self.stt = stt
        self._stop_requested = threading.Event()
        self._paused = threading.Event()
        self._stream_active = threading.Event()
        self._activity_active = threading.Event()
        self._active_session = False
        self._normalized_end_phrases = [self._clean_transcript(p) for p in ACTIVE_SESSION_END_PHRASES]

    def pause(self, wait=False, timeout=1.0):
        self._paused.set()
        if wait:
            self.wait_until_paused(timeout)

    def resume(self):
        if not self._stop_requested.is_set():
            self._paused.clear()

    def stop(self):
        self._stop_requested.set()
        self._paused.clear()

    def set_active_session(self, enabled):
        self._active_session = enabled

    def wait_until_paused(self, timeout=1.0):
        deadline = time.time() + timeout
        while (
            self._stream_active.is_set() or self._activity_active.is_set()
        ) and time.time() < deadline:
            time.sleep(0.01)
        return not self._stream_active.is_set() and not self._activity_active.is_set()

    def run(self):
        if WAKE_ENGINE.lower() != "vad_whisper_gate":
            self.error.emit(f"Unsupported active wake engine: {WAKE_ENGINE}")
            return

        if self.stt is None or self.stt.model is None:
            self.error.emit("Wake listener needs the already-loaded STT model.")
            return

        while not self._stop_requested.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            self.status_changed.emit("Passive listening")
            self._listen_for_utterance()

        self._stream_active.clear()
        self._activity_active.clear()

    def _listen_for_utterance(self):
        self._activity_active.set()
        wav_path = None
        duration = 0.0
        candidate_rms = 0.0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=WAKE_AUDIO_FRAME_SIZE,
                channels=1,
                dtype="int16",
            ) as stream:
                self._stream_active.set()
                wav_path, duration, candidate_rms = self._capture_utterance(stream)
        except Exception as e:
            if not self._stop_requested.is_set():
                self.error.emit(f"Wake listener failed: {e}")
                self._stop_requested.set()
            self._activity_active.clear()
            return
        finally:
            self._stream_active.clear()

        if not wav_path or self._stop_requested.is_set() or self._paused.is_set():
            self._activity_active.clear()
            return

        print(f"[Wake] candidate duration: {duration:.2f}s")
        print(f"[Wake] candidate RMS: {candidate_rms:.4f}")
        self.status_changed.emit("Checking wake phrase")

        try:
            transcript = self._transcribe_utterance(wav_path)
        except Exception as e:
            if not self._stop_requested.is_set():
                self.error.emit(f"Wake transcription failed: {e}")
                self._stop_requested.set()
            self._remove_file(wav_path)
            self._activity_active.clear()
            return
        finally:
            self._remove_file(wav_path)

        if self._stop_requested.is_set() or self._paused.is_set():
            self._activity_active.clear()
            return

        normalized = self._clean_transcript(transcript)

        print(f"[Wake] raw transcript: {transcript!r}")
        print(f"[Wake] normalized transcript: {normalized!r}")
        print(f"[Wake] mode: {'active session' if self._active_session else 'passive'}")

        if not is_valid_transcript(transcript):
            reason = _invalid_transcript_reason(transcript, duration, candidate_rms)
            print(f"[Wake] transcript rejected: {reason}")
            print("[Wake] TTS interrupted: no")
            self.utterance_rejected.emit()
            self._activity_active.clear()
            return

        if self._active_session:
            if self._is_end_phrase(normalized):
                print("[Wake] active session ended by end phrase")
                print("[Wake] transcript accepted: end phrase")
                self._paused.set()
                self.session_ended.emit()
                self._activity_active.clear()
                return
            print("[Wake] active session: auto-accepting utterance")
            print("[Wake] transcript accepted: active session valid speech")
            self._paused.set()
            self.utterance_accepted.emit(transcript.strip())
            self._activity_active.clear()
            return

        matched_word = self._matched_positive_word(normalized)

        if matched_word is None:
            print("[Wake] transcript rejected: no positive wake word found in passive mode")
            print("[Wake] TTS interrupted: no")
            self.utterance_rejected.emit()
            self._activity_active.clear()
            return

        print(f"[Wake] matched positive word: {matched_word!r}")
        print("[Wake] transcript accepted: passive wake word matched")
        self._paused.set()
        self.utterance_accepted.emit(transcript.strip())
        self._activity_active.clear()

    def _capture_utterance(self, stream):
        frame_seconds = WAKE_AUDIO_FRAME_SIZE / SAMPLE_RATE
        pre_roll_frames = max(1, int(WAKE_PREROLL_SECONDS / frame_seconds))
        pre_roll = deque(maxlen=pre_roll_frames)
        frames = []
        speech_started = False
        speech_duration = 0.0
        silent_for = 0.0
        noise_floor = VAD_MIN_RMS
        barge_in_logged = False

        while not self._stop_requested.is_set() and not self._paused.is_set():
            speaking_now = is_speaking()
            if speaking_now and not WAKE_ALLOW_BARGE_IN:
                self._paused.set()
                return None, 0.0, 0.0
            if speaking_now and not barge_in_logged:
                print("[Wake] barge-in listening while TTS is speaking")
                barge_in_logged = True

            frame, _overflowed = stream.read(WAKE_AUDIO_FRAME_SIZE)
            pcm = np.asarray(frame[:, 0], dtype=np.int16).copy()
            frame_rms = self._rms(pcm)
            speech_threshold = max(VAD_MIN_RMS, noise_floor * VAD_SPEECH_RATIO)

            if not speech_started:
                pre_roll.append(pcm)
                if frame_rms >= speech_threshold:
                    speech_started = True
                    frames.extend(pre_roll)
                    speech_duration = len(frames) * frame_seconds
                    silent_for = 0.0
                    print(
                        "[Wake] VAD candidate started "
                        f"(rms={frame_rms:.4f}, threshold={speech_threshold:.4f})"
                    )
                    self.status_changed.emit("Speech detected")
                    continue

                noise_floor = (VAD_NOISE_ALPHA * noise_floor) + ((1.0 - VAD_NOISE_ALPHA) * frame_rms)
                continue

            frames.append(pcm)
            speech_duration += frame_seconds

            if frame_rms >= speech_threshold:
                silent_for = 0.0
            else:
                silent_for += frame_seconds

            if speech_duration >= WAKE_MAX_UTTERANCE_SECONDS:
                break

            if silent_for >= WAKE_SILENCE_STOP_SECONDS:
                break

        if not frames:
            return None, 0.0, 0.0

        audio = np.concatenate(frames)
        candidate_rms = self._rms(audio)
        print(f"[Wake] VAD candidate ended (silent_for={silent_for:.2f}s)")
        wav_path = self._save_wav(audio)
        return wav_path, speech_duration, candidate_rms

    def _transcribe_utterance(self, wav_path):
        text, _segments, _info = self.stt.transcribe(
            wav_path,
            language=WAKE_CHECK_LANGUAGE,
            beam_size=1,
        )
        return text.strip()

    def _matched_positive_word(self, normalized):
        if not normalized:
            return None

        positives = [self._clean_transcript(word) for word in WAKE_POSITIVE_WORDS]

        for phrase in positives:
            if " " in phrase and self._contains_phrase(normalized, phrase):
                return phrase

        tokens = normalized.split()
        direct_words = {word for word in positives if " " not in word}

        for token in tokens:
            if token in direct_words:
                return token

        return None

    @staticmethod
    def _contains_phrase(normalized, phrase):
        return f" {phrase} " in f" {normalized} "

    def _is_end_phrase(self, normalized):
        for phrase in self._normalized_end_phrases:
            if phrase == normalized:
                return True
            if " " in phrase and self._contains_phrase(normalized, phrase):
                return True
        return False

    @staticmethod
    def _clean_transcript(text):
        punctuation = string.punctuation + "؟،؛"
        text = text.lower().translate(str.maketrans("", "", punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _save_wav(audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        return wav_path

    @staticmethod
    def _remove_file(path):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _rms(pcm):
        audio = pcm.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio * audio)))
