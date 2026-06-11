import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from marvin.config import SAMPLE_RATE, MIN_RECORDING


class Recorder:
    def __init__(self):
        self._chunks = []
        self._stream = None

    def start(self):
        self._chunks = []
        def _callback(indata, _frames, _time_info, _status):
            self._chunks.append(indata.copy())
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_callback)
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()
        audio = np.concatenate(self._chunks, axis=0)
        duration = len(audio) / SAMPLE_RATE
        return audio, duration

    @staticmethod
    def is_too_short(duration):
        return duration < MIN_RECORDING

    @staticmethod
    def save_temp_wav(audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        sf.write(path, audio, SAMPLE_RATE)
        return path
