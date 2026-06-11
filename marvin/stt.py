import importlib.util
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from marvin.config import (
    MIXED_LANGUAGE_RESCUE_LANGUAGES,
    MIXED_LANGUAGE_RESCUE_MARGIN,
    MIXED_LANGUAGE_TRANSCRIPTION,
    TRANSCRIBE_LANGUAGE,
    STT_DEVICE,
    WHISPER_HOTWORDS,
    WHISPER_INITIAL_PROMPT,
    WHISPER_MODEL,
    WHISPER_TEMPERATURE,
)


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
PHONETIC_ENGLISH_RE = re.compile(
    r"\b(show you|show me|italian|little bit more|the speaker may)\b",
    re.IGNORECASE,
)
_dll_directory_handles = []


@dataclass
class TranscriptionCandidate:
    label: str
    text: str
    segments: list
    info: object
    avg_logprob: float | None


class STT:
    def __init__(self):
        self.model = None
        self.device = None
        self.startup_warning = None

    def load_model(self):
        print("Loading model...")
        start = time.time()
        _add_nvidia_dll_directories()
        requested_device = STT_DEVICE if STT_DEVICE in {"auto", "cuda", "cpu"} else "auto"

        if requested_device in {"auto", "cuda"}:
            try:
                self.model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                self.device = "cuda"
            except Exception as e:
                if _is_cuda_runtime_error(e):
                    message = (
                        "CUDA STT failed because required NVIDIA CUDA runtime files are missing. "
                        "Text chat still works. Install GPU requirements or set MARVIN_STT_DEVICE=cpu."
                    )
                else:
                    message = (
                        "CUDA STT failed. Text chat still works. "
                        "Marvin will try CPU STT next, or you can set MARVIN_STT_DEVICE=cpu."
                    )

                if requested_device == "cuda":
                    raise RuntimeError(f"{message} Details: {e}") from e

                self.startup_warning = message
                print(f"{message} Details: {e}")
                print("Retrying Whisper on CPU. First run may download the model.")

        if self.model is None:
            self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            self.device = "cpu"

        print(f"Model loaded on {self.device} ({time.time() - start:.2f}s).")

    def transcribe(self, wav_path, language=TRANSCRIBE_LANGUAGE, beam_size=None):
        if language is not None or not MIXED_LANGUAGE_TRANSCRIPTION:
            candidate = self._decode_once(wav_path, language, beam_size, label=language or "auto")
            return candidate.text, candidate.segments, candidate.info

        candidates = [self._decode_once(wav_path, None, beam_size, label="auto")]
        auto = candidates[0]

        if self._should_try_language_rescue(auto):
            for rescue_language in MIXED_LANGUAGE_RESCUE_LANGUAGES:
                candidates.append(
                    self._decode_once(
                        wav_path,
                        rescue_language,
                        beam_size,
                        label=f"forced_{rescue_language}",
                    )
                )

        selected = self._select_best_candidate(candidates)
        if len(candidates) > 1:
            self._log_candidates(candidates, selected)

        return selected.text, selected.segments, selected.info

    def _decode_once(self, wav_path, language, beam_size, label):
        options = {
            "vad_filter": True,
            "task": "transcribe",
            "initial_prompt": WHISPER_INITIAL_PROMPT,
            "hotwords": WHISPER_HOTWORDS,
            "temperature": WHISPER_TEMPERATURE,
            "condition_on_previous_text": False,
            "multilingual": True,
        }
        if language is not None:
            options["language"] = language
        if beam_size is not None:
            options["beam_size"] = beam_size

        segments, info = self.model.transcribe(wav_path, **options)
        segments = list(segments)
        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()

        return TranscriptionCandidate(
            label=label,
            text=text,
            segments=segments,
            info=info,
            avg_logprob=self._average_logprob(segments),
        )

    def _should_try_language_rescue(self, auto):
        if self._has_arabic(auto.text):
            return False

        detected_language = getattr(auto.info, "language", None)
        return detected_language != "ar" or self._looks_phonetic_english(auto.text)

    def _select_best_candidate(self, candidates):
        selected = candidates[0]

        for candidate in candidates[1:]:
            if not self._has_rescue_arabic_content(candidate.text):
                continue
            if not self._is_close_enough(candidate, selected):
                continue
            selected = candidate
            break

        return selected

    def _is_close_enough(self, candidate, reference):
        if candidate.avg_logprob is None or reference.avg_logprob is None:
            return self._looks_phonetic_english(reference.text)

        return candidate.avg_logprob >= reference.avg_logprob - MIXED_LANGUAGE_RESCUE_MARGIN

    @staticmethod
    def _average_logprob(segments):
        weighted_sum = 0.0
        total_weight = 0.0

        for segment in segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is None:
                continue

            start = getattr(segment, "start", 0.0) or 0.0
            end = getattr(segment, "end", start) or start
            weight = max(end - start, 0.01)
            weighted_sum += avg_logprob * weight
            total_weight += weight

        if total_weight == 0.0:
            return None

        return weighted_sum / total_weight

    @staticmethod
    def _has_arabic(text):
        return bool(ARABIC_RE.search(text or ""))

    @staticmethod
    def _has_rescue_arabic_content(text):
        arabic_wake_words = {"مارفن"}
        punctuation = ".,!?؟،؛:;\"'()[]{}"
        tokens = (text or "").split()
        content_tokens = [
            token for token in tokens if token.strip(punctuation) not in arabic_wake_words
        ]
        return bool(ARABIC_RE.search(" ".join(content_tokens)))

    @staticmethod
    def _looks_phonetic_english(text):
        return bool(PHONETIC_ENGLISH_RE.search(text or ""))

    def _log_candidates(self, candidates, selected):
        parts = []
        for candidate in candidates:
            detected_language = getattr(candidate.info, "language", None)
            language_probability = getattr(candidate.info, "language_probability", None)
            probability_text = (
                f"{language_probability:.2f}"
                if isinstance(language_probability, (int, float))
                else "?"
            )
            score_text = (
                f"{candidate.avg_logprob:.2f}"
                if isinstance(candidate.avg_logprob, (int, float))
                else "?"
            )
            parts.append(
                f"{candidate.label}(detected={detected_language}, p={probability_text}, "
                f"score={score_text}, arabic={self._has_arabic(candidate.text)}, "
                f"arabic_content={self._has_rescue_arabic_content(candidate.text)}): {candidate.text!r}"
            )

        print("[STT] mixed candidates: " + " | ".join(parts) + f" | selected={selected.label}")


def _add_nvidia_dll_directories():
    spec = importlib.util.find_spec("nvidia")
    search_locations = getattr(spec, "submodule_search_locations", None)
    if not search_locations:
        return

    for package_root in search_locations:
        for dll_dir in Path(package_root).glob("*/bin"):
            if not dll_dir.is_dir():
                continue

            dll_dir_text = str(dll_dir)
            if hasattr(os, "add_dll_directory"):
                _dll_directory_handles.append(os.add_dll_directory(dll_dir_text))
            os.environ["PATH"] = dll_dir_text + os.pathsep + os.environ.get("PATH", "")


def _is_cuda_runtime_error(error):
    text = str(error).lower()
    cuda_markers = (
        "cublas",
        "cudnn",
        "cuda",
        "cudart",
        "nvrtc",
        ".dll",
    )
    load_markers = (
        "not found",
        "cannot be loaded",
        "could not be loaded",
        "load library",
        "dll load failed",
        "failed to load",
    )
    return any(marker in text for marker in cuda_markers) and any(
        marker in text for marker in load_markers
    )
