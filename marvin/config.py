import os

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SAMPLE_RATE = 16000
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Marvin")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
STT_DEVICE = os.getenv("MARVIN_STT_DEVICE", "auto").strip().lower()
TRANSCRIBE_LANGUAGE = None
WHISPER_INITIAL_PROMPT = (
    "Mixed-language dictation: English, Arabic, and Lebanese Arabic. "
    "Preserve Arabic words in Arabic script. Transcribe exactly what is said. "
    "Do not translate. Lebanese Arabic words may include شو، هيدا، عم، وين، ليش، كيف، وقف."
)
WHISPER_HOTWORDS = (
    "Marvin marven marvyn marlin "
    "مارفن شو هيدا عم وين ليش كيف وقف screen stop open"
)
WHISPER_TEMPERATURE = 0.0
MIXED_LANGUAGE_TRANSCRIPTION = True
MIXED_LANGUAGE_RESCUE_LANGUAGES = ["ar"]
MIXED_LANGUAGE_RESCUE_MARGIN = 0.75
MIN_RECORDING = 0.3
MIN_VALID_TRANSCRIPT_CHARS = 1
MIN_VALID_AUDIO_SECONDS = 0.4
MIN_VALID_RMS = 0.003
MAX_HISTORY = 6

# Wake word
ENABLE_VOICE_INPUT = _env_bool("MARVIN_ENABLE_VOICE_INPUT", True)
ENABLE_WAKE_WORD = _env_bool("MARVIN_ENABLE_WAKE_WORD", True)
WAKE_ENGINE = "vad_whisper_gate"
WAKE_POSITIVE_WORDS = [
    "marvin",
    "marven",
    "marvyn",
    "marlin",
    "مارفن",
]
WAKE_SILENCE_STOP_SECONDS = 1
WAKE_MAX_UTTERANCE_SECONDS = 20
WAKE_PREROLL_SECONDS = 0.3
WAKE_CHECK_LANGUAGE = None
WAKE_ALLOW_BARGE_IN = True

# Active session
ENABLE_ACTIVE_SESSION = True
ACTIVE_SESSION_STARTS_AFTER_WAKE = True
ACTIVE_SESSION_TIMEOUT_SECONDS = 120
ACTIVE_SESSION_END_PHRASES = [
    "marvin stop",
    "marvin end",
    "ok marvin stop",
    "okay marvin stop",
    "ok marvin end",
    "okay marvin end",
    "stop marvin",
    "end marvin",
    "stop session",
    "end session",
    "stop listening",
    "end active session",
]

# Kokoro TTS
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "model/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "model/voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_USE_GPU = _env_bool("KOKORO_USE_GPU", True)
KOKORO_WARMUP_ON_STARTUP = _env_bool("KOKORO_WARMUP_ON_STARTUP", True)
# Other voices: am_michael, af_heart, af_bella, af_nicole, af_sarah, af_sky,
#               bf_emma, bm_george, am_liam, am_onyx, am_fenrir, am_echo, am_eric, am_puck
