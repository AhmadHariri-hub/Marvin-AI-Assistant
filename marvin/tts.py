import os
import ctypes
import importlib.util
import re
from pathlib import Path
import shutil
import tempfile
import threading
import time
import sounddevice as sd
import pyttsx3
from marvin.config import (
    KOKORO_MODEL_PATH,
    KOKORO_VOICES_PATH,
    KOKORO_VOICE,
    KOKORO_USE_GPU,
    KOKORO_WARMUP_ON_STARTUP,
)

_kokoro = None
_provider_used = "CPU"
_dll_directory_handles = []
_stop_requested = threading.Event()
_speaking = threading.Event()
_MIN_PHONEMIZER_TEMP_BYTES = 25 * 1024 * 1024
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _add_nvidia_dll_directories():
    spec = importlib.util.find_spec("nvidia")
    search_locations = getattr(spec, "submodule_search_locations", None)
    if not search_locations:
        return []

    added = []
    for package_root in search_locations:
        for dll_dir in Path(package_root).glob("*/bin"):
            if not dll_dir.is_dir():
                continue

            dll_dir_text = str(dll_dir)
            if hasattr(os, "add_dll_directory"):
                _dll_directory_handles.append(os.add_dll_directory(dll_dir_text))
            os.environ["PATH"] = dll_dir_text + os.pathsep + os.environ.get("PATH", "")
            added.append(dll_dir_text)

    return added


def _preload_cudnn_sublibraries(dll_dirs):
    cudnn_dir = next((Path(path) for path in dll_dirs if Path(path).parts[-2:] == ("cudnn", "bin")), None)
    if cudnn_dir is None:
        return

    for dll_path in sorted(cudnn_dir.glob("cudnn*.dll")):
        try:
            ctypes.CDLL(str(dll_path))
        except OSError as e:
            print(f"[TTS] Could not preload {dll_path.name}: {e}")


def _preload_onnx_cuda_dlls(rt):
    preload_dlls = getattr(rt, "preload_dlls", None)
    if preload_dlls is None:
        print("[TTS] ONNX Runtime cannot preload CUDA DLLs; install onnxruntime-gpu>=1.21")
        return

    try:
        print("[TTS] Preloading ONNX CUDA/cuDNN DLLs...")
        dll_dirs = _add_nvidia_dll_directories()
        preload_dlls(directory="")
        _preload_cudnn_sublibraries(dll_dirs)
    except Exception as e:
        print(f"[TTS] CUDA DLL preload failed: {e}")


def preload_kokoro():
    if not KOKORO_WARMUP_ON_STARTUP:
        print("[TTS] Startup preload skipped.")
        return

    try:
        _get_kokoro()
    except Exception as e:
        print(f"[TTS] Kokoro preload skipped: {e}")
        return

    if _kokoro is not None:
        if not _phonemizer_temp_has_space():
            print(
                "[TTS] Warmup skipped: not enough free temp space for espeak/phonemizer. "
                "Free some disk space to re-enable Kokoro speech generation."
            )
            return

        print("[TTS] Running warmup...")
        t0 = time.time()
        try:
            _kokoro.create("Ready.", voice=KOKORO_VOICE)
            print(f"[TTS] Warmup done in {time.time() - t0:.2f}s")
        except OSError as e:
            print(f"[TTS] Warmup skipped: {e}")
        except Exception as e:
            print(f"[TTS] Warmup failed, continuing without startup warmup: {e}")


def is_speaking():
    return _speaking.is_set()


def stop_speaking():
    _stop_requested.set()
    try:
        sd.stop()
    except Exception as e:
        print(f"[TTS] Could not stop playback: {e}")


def sanitize_for_speech(text):
    text = str(text or "")
    had_code_block = bool(_CODE_FENCE_RE.search(text))
    text = _CODE_FENCE_RE.sub(" ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s[-*•]\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if had_code_block:
        code_note = "I wrote the code in the chat."
        text = f"{text} {code_note}".strip() if text else code_note

    return text


def _phonemizer_temp_has_space():
    try:
        free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
    except OSError:
        return True

    if free_bytes < _MIN_PHONEMIZER_TEMP_BYTES:
        print(
            f"[TTS] Temp free space is low ({free_bytes / (1024 * 1024):.1f} MB). "
            "Kokoro phonemizer needs temp space for espeak."
        )
        return False

    return True


def _get_kokoro():
    global _kokoro, _provider_used
    if _kokoro is not None:
        print(f"[TTS] Kokoro already loaded (reusing) | provider: {_provider_used}")
        return _kokoro

    missing = [
        path
        for path in (Path(KOKORO_MODEL_PATH), Path(KOKORO_VOICES_PATH))
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"missing Kokoro model files: {missing_text}. "
            "Download them into model/ or rely on the pyttsx3 fallback."
        )

    print("[TTS] Loading Kokoro model...")
    t0 = time.time()

    import onnxruntime as rt
    if KOKORO_USE_GPU:
        rt.set_default_logger_severity(3)
        _preload_onnx_cuda_dlls(rt)

    providers = rt.get_available_providers()
    print(f"[TTS] ONNX providers available: {providers}")

    if KOKORO_USE_GPU and "CUDAExecutionProvider" in providers:
        os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
        _provider_used = "CUDA"
        print("[TTS] Trying: CUDAExecutionProvider (GPU)")
    else:
        if KOKORO_USE_GPU:
            print("[TTS] CUDA not available, falling back to CPU")
        _provider_used = "CPU"

    from kokoro_onnx import Kokoro
    _kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    actual = _kokoro.sess.get_providers()
    if "CUDAExecutionProvider" in actual:
        _provider_used = "CUDA"
    else:
        _provider_used = "CPU"
        if KOKORO_USE_GPU:
            print("[TTS] CUDA did not load; using CPU. Check onnxruntime-gpu[cuda,cudnn] dependencies.")
    print(f"[TTS] Kokoro loaded in {time.time() - t0:.2f}s | provider: {_provider_used} (session: {actual})")
    return _kokoro


def speak(text):
    text = sanitize_for_speech(text)
    if not text:
        print("[TTS] speak() skipped; no speech-friendly text")
        return

    t_total = time.time()
    _stop_requested.clear()
    print(f"[TTS] speak() called | text length: {len(text)} chars | provider: {_provider_used}")

    try:
        t0 = time.time()
        koko = _get_kokoro()
        print(f"[TTS] Model/voice ready: {time.time() - t0:.3f}s")

        t0 = time.time()
        if not _phonemizer_temp_has_space():
            raise RuntimeError("not enough free temp space for Kokoro phonemizer")

        audio, sr = koko.create(text, voice=KOKORO_VOICE)
        t_gen = time.time() - t0
        audio_dur = len(audio) / sr
        print(f"[TTS] Audio generated: {t_gen:.2f}s | audio duration: {audio_dur:.2f}s | RTF: {t_gen / audio_dur:.2f}x")

        if _stop_requested.is_set():
            print("[TTS] Playback skipped; speech was interrupted before audio started")
            print(f"[TTS] Total speak() time: {time.time() - t_total:.2f}s")
            return

        t0 = time.time()
        _speaking.set()
        try:
            sd.play(audio, samplerate=sr)
            print(f"[TTS] Playback started in: {time.time() - t0:.3f}s")

            t0 = time.time()
            sd.wait()
        finally:
            _speaking.clear()

        if _stop_requested.is_set():
            print(f"[TTS] Playback interrupted in: {time.time() - t0:.2f}s")
        else:
            print(f"[TTS] Playback finished in: {time.time() - t0:.2f}s")
    except Exception as e:
        if _stop_requested.is_set():
            print(f"[TTS] Playback stopped: {e}")
            print(f"[TTS] Total speak() time: {time.time() - t_total:.2f}s")
            return

        print(f"[TTS] Kokoro failed: {e}, falling back to pyttsx3")
        try:
            t0 = time.time()
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            print(f"[TTS] pyttsx3 fallback finished in: {time.time() - t0:.2f}s")
        except Exception as e2:
            print(f"[TTS] pyttsx3 fallback also failed: {e2}")

    print(f"[TTS] Total speak() time: {time.time() - t_total:.2f}s")
