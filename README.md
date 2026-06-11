# Marvin

Marvin is a work-in-progress Windows desktop AI assistant. The long-term goal is to build a local assistant that can help users interact with their PC through chat, voice, and practical tools.

The current prototype includes a PySide6 desktop chat UI, system tray support, manual text chat, microphone recording, Whisper speech-to-text, DeepSeek-compatible LLM responses, text-to-speech output through Kokoro or pyttsx3 fallback, and demo mode for running without live API calls.

![Marvin desktop assistant UI](assets/marvin-screenshot.png)

## Status

Active prototype / work in progress.

The core assistant flow is implemented, but Marvin is still being improved. Voice and wake behavior, memory, PC-control tools, and always-on assistant behavior are still in progress.

## Current Features

* Desktop chat interface built with PySide6
* System tray behavior
* Manual text chat
* Microphone recording enabled by default
* Wake listening enabled by default
* Whisper STT with CUDA auto-detect and CPU fallback
* DeepSeek-compatible LLM chat through an OpenAI-compatible API client
* Demo mode with local sample responses
* Kokoro TTS support when local model files exist
* pyttsx3 fallback TTS

## Planned / In Progress

* More reliable wake-word and voice flow
* Better assistant memory
* PC-control and local tool integrations
* File/app automation tools
* Cleaner assistant actions
* More stable always-on behavior

## Quick Start - Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m app.main
```

The default `.env.example` runs Marvin in demo mode, so the app opens without an API key. Voice input and wake listening are enabled by default.

Base install:

```powershell
pip install -r requirements.txt
```

Optional GPU STT install:

```powershell
pip install -r requirements-gpu.txt
```

## How to Use Marvin

When Marvin starts, the desktop chat window opens and the system tray icon is enabled. If `MARVIN_ENABLE_VOICE_INPUT=1`, Marvin loads Whisper STT. If `MARVIN_ENABLE_WAKE_WORD=1` and STT loads successfully, wake listening starts automatically. If voice startup fails, Marvin shows a system message and keeps text chat available.

For text chat, type a message in the input box and press Enter or click the send button. Shift+Enter inserts a new line. In demo mode, Marvin returns local sample responses. In live mode, Marvin sends the message to the configured DeepSeek-compatible API.

For manual voice input, click the microphone button to start recording, then click it again to stop. The flow is:

```text
microphone recording -> Whisper transcription -> Marvin response -> TTS output if available
```

Wake listening uses VAD plus Whisper transcription. In passive listening mode, Marvin accepts an utterance when the transcript contains one of these configured wake words: `marvin`, `marven`, `marvyn`, `marlin`, or `مارفن`. The accepted utterance is sent to Marvin as the command, so say the wake word as part of the request, such as "Marvin, what can you do?"

After a wake command is accepted, Marvin starts an active session. During an active session, valid speech is accepted without repeating the wake word. The active session remains open while used and times out after 120 seconds of inactivity.

Supported active-session end phrases are: `marvin stop`, `marvin end`, `ok marvin stop`, `okay marvin stop`, `ok marvin end`, `okay marvin end`, `stop marvin`, `end marvin`, `stop session`, `end session`, `stop listening`, and `end active session`. When an end phrase is detected, Marvin ends the active session and returns to passive listening.

Closing the Marvin window hides it to the tray instead of quitting. Double-click the tray icon, or use the tray menu's Show action, to restore it. The tray menu also supports Hide, Mute voice, and Quit.

Marvin is an active prototype. Wake/listening behavior is still being improved. If voice or STT fails, text chat should still work. Demo mode does not call DeepSeek.

## API / Config

Demo mode works without an API key. Live mode requires the user to add their own DeepSeek API key to a local `.env` file. `.env.example` is included as the configuration template, and the local `.env` file is excluded from version control.

Default demo configuration:

```env
DEEPSEEK_API_KEY=
MARVIN_DEMO_MODE=1
MARVIN_ENABLE_VOICE_INPUT=1
MARVIN_ENABLE_WAKE_WORD=1
WHISPER_MODEL=large-v3-turbo
MARVIN_STT_DEVICE=auto
KOKORO_WARMUP_ON_STARTUP=0
```

Live mode:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MARVIN_DEMO_MODE=0
```

If demo mode is off and no API key exists, Marvin shows a clear setup message instead of crashing.

## Optional: Install Kokoro TTS

Kokoro is optional. Marvin falls back to pyttsx3 if Kokoro model files are missing.

To install the optional Kokoro model files:

```powershell
.\scripts\install_kokoro.ps1
```

The script downloads the files into the local `model/` folder:

```text
model/kokoro-v1.0.onnx
model/voices-v1.0.bin
```

The model files are large, so they are not included in the repository.

### Kokoro Warmup

By default, Kokoro startup warmup is disabled:

```env
KOKORO_WARMUP_ON_STARTUP=0
```

With this setting, Marvin launches faster, but the first Kokoro spoken response may be slower because the TTS model loads on first use.

To preload Kokoro during startup, set:

```env
KOKORO_WARMUP_ON_STARTUP=1
```

This can make startup slower, but the first spoken response should be faster.


## Voice Mode

Voice input and wake listening are enabled by default in the public demo config:

```env
MARVIN_ENABLE_VOICE_INPUT=1
MARVIN_ENABLE_WAKE_WORD=1
WHISPER_MODEL=large-v3-turbo
MARVIN_STT_DEVICE=auto
```

The first voice run may take longer because Whisper may download and load the configured model. `MARVIN_STT_DEVICE=auto` tries CUDA first and falls back to CPU if CUDA is unavailable. CPU is slower, but it is useful as a reliable fallback.

If CUDA DLL errors occur, install the optional GPU requirements or set:

```env
MARVIN_STT_DEVICE=cpu
```

For a faster text-only startup, set:

```env
MARVIN_ENABLE_VOICE_INPUT=0
MARVIN_ENABLE_WAKE_WORD=0
```

In demo mode, the voice flow is:

```text
wake/manual microphone -> Whisper STT -> demo response -> TTS if available
```

## Troubleshooting: CUDA STT error

If you see an error like:

```text
cublas64_12.dll is not found or cannot be loaded
```

it means the NVIDIA CUDA/cuBLAS runtime needed by faster-whisper/CTranslate2 is missing from the environment.

Fix options:

```powershell
pip install -r requirements-gpu.txt
```

or use CPU fallback:

```env
MARVIN_STT_DEVICE=cpu
```

## Privacy / Data

Marvin records microphone audio only when voice input is used. Temporary WAV files are deleted after processing.

In live mode, typed or transcribed text is sent to the configured LLM provider. Demo mode does not make live API calls.

## Run Checks

```powershell
python -m compileall app marvin
python -m app.main
```
