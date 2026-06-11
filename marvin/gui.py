import os
import time
from threading import Event
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QSize, QThread, Signal, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QPainter, QPixmap

from marvin.audio_recorder import Recorder
from marvin.stt import STT
from marvin.brain import Brain
from marvin.config import (
    ASSISTANT_NAME,
    ENABLE_VOICE_INPUT,
    ENABLE_WAKE_WORD,
    ENABLE_ACTIVE_SESSION,
    ACTIVE_SESSION_STARTS_AFTER_WAKE,
    ACTIVE_SESSION_TIMEOUT_SECONDS,
    WAKE_ALLOW_BARGE_IN,
)
from marvin.tts import speak, preload_kokoro, stop_speaking, is_speaking
from marvin.wake_word import WakeWordListener


MICROPHONE_ICON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "images", "microphone-black-shape.png")
)
SEND_ICON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "images", "Arrowmessage.png")
)
SPEAKER_ICON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "images", "speaker-filled-audio-tool.png")
)
MUTE_ICON_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "images", "volume-mute.png")
)


class CommandInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not event.modifiers() & Qt.ShiftModifier:
            self.submitted.emit()
            event.accept()
            return

        super().keyPressEvent(event)


class TranscriptionWorker(QThread):
    status_changed = Signal(str)
    message_received = Signal(str, str)
    finished = Signal()

    def __init__(self, wav_path, brain, stt, speech_enabled, source="manual"):
        super().__init__()
        self.wav_path = wav_path
        self.brain = brain
        self.stt = stt
        self.speech_enabled = speech_enabled
        self.source = source

    def run(self):
        try:
            self.status_changed.emit("Transcribing...")
            if self.source == "wake":
                print("[Wake] command transcription started")
            text, segments, _info = self.stt.transcribe(self.wav_path)
            if self.source == "wake":
                print(f"[Wake] command transcription: {text!r}")

            if self.isInterruptionRequested():
                return

            if not text.strip():
                self.status_changed.emit("Ready")
                return

            for seg in segments:
                self.message_received.emit("You", seg.text)

            if self.isInterruptionRequested():
                return

            self.status_changed.emit("Thinking...")
            reply = self.brain.chat(text)

            if self.isInterruptionRequested():
                return

            self.message_received.emit(ASSISTANT_NAME, reply)

            if not self.speech_enabled.is_set():
                print("[GUI] Speech muted; skipping speak()")
                self.status_changed.emit("Ready")
                return

            self.status_changed.emit("Speaking...")
            t0 = time.time()
            speak(reply)
            print(f"[GUI] AI reply received → speak() done: {time.time() - t0:.2f}s")

            self.status_changed.emit("Ready")
        except Exception as e:
            self.message_received.emit("System", f"Voice request failed: {e}")
            self.status_changed.emit("Ready")
        finally:
            if os.path.exists(self.wav_path):
                os.unlink(self.wav_path)
            self.finished.emit()


class TextCommandWorker(QThread):
    status_changed = Signal(str)
    message_received = Signal(str, str)
    finished = Signal()

    def __init__(self, text, brain, speech_enabled):
        super().__init__()
        self.text = text
        self.brain = brain
        self.speech_enabled = speech_enabled

    def run(self):
        try:
            text = self.text.strip()
            if not text:
                self.status_changed.emit("Ready")
                return

            self.message_received.emit("You", text)

            if self.isInterruptionRequested():
                return

            self.status_changed.emit("Thinking")
            reply = self.brain.chat(text)

            if self.isInterruptionRequested():
                return

            self.message_received.emit(ASSISTANT_NAME, reply)
            if not self.speech_enabled.is_set():
                print("[GUI] Speech muted; skipping speak()")
                self.status_changed.emit("Ready")
                return

            self.status_changed.emit("Speaking")
            t0 = time.time()
            speak(reply)
            print(f"[GUI] AI reply received -> speak() done: {time.time() - t0:.2f}s")
            self.status_changed.emit("Ready")
        except Exception as e:
            self.message_received.emit("System", f"Text request failed: {e}")
            self.status_changed.emit("Ready")
        finally:
            self.finished.emit()


class MarvinWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(ASSISTANT_NAME)
        self.setMinimumSize(980, 660)
        self.resize(1180, 760)

        self.recorder = Recorder()
        self.brain = None
        self.stt = None
        self.voice_input_available = False
        self.is_recording = False
        self.is_ai_speaking = False
        self.is_speech_muted = False
        self.speech_enabled = Event()
        self.speech_enabled.set()
        self.is_wake_recording = False
        self.worker = None
        self.wake_listener = None
        self.wake_disabled_for_session = False
        self._really_quitting = False
        self._pending_wake_transcript = None
        self._active_session = False
        self._last_active_session_time = None

        self._session_timer = QTimer(self)
        self._session_timer.setSingleShot(True)
        self._session_timer.timeout.connect(self._on_session_timeout)

        self._setup_ui()
        self._setup_tray()
        self._setup_backend()

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._setup_header())
        root_layout.addWidget(self._setup_chat(), 1)
        root_layout.addWidget(self._setup_composer())

        self._setup_styles()
        self._update_speech_mute_controls()
        self._set_status("Ready")

    def _setup_header(self):
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(48)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)
        header_layout.setSpacing(12)

        name = QLabel(ASSISTANT_NAME)
        name.setObjectName("headerName")
        header_layout.addWidget(name)

        header_layout.addStretch(1)

        self.header_status_dot = QLabel()
        self.header_status_dot.setObjectName("statusDot")
        self.header_status_dot.setFixedSize(8, 8)
        header_layout.addWidget(self.header_status_dot)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("headerStatus")
        header_layout.addWidget(self.status_label)

        return header

    def _setup_chat(self):
        self.chat_area = QScrollArea()
        self.chat_area.setObjectName("chatArea")
        self.chat_area.viewport().setObjectName("chatAreaViewport")
        self.chat_area.setFrameShape(QFrame.NoFrame)
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.chat_viewport = QWidget()
        self.chat_viewport.setObjectName("chatViewport")
        self.chat_layout = QVBoxLayout(self.chat_viewport)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_area.setWidget(self.chat_viewport)
        self._message_bubbles = []

        return self.chat_area

    def _setup_composer(self):
        outer = QWidget()
        outer.setObjectName("composerWrapper")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(16, 0, 16, 16)

        composer_panel = QFrame()
        composer_panel.setObjectName("composerPanel")
        composer_layout = QHBoxLayout(composer_panel)
        composer_layout.setContentsMargins(14, 12, 14, 12)
        composer_layout.setSpacing(8)

        self.command_input = CommandInput()
        self.command_input.setObjectName("commandInput")
        self.command_input.setPlaceholderText(f"Ask {ASSISTANT_NAME} anything...")
        self.command_input.setMinimumHeight(48)
        self.command_input.setMaximumHeight(160)
        self.command_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.command_input.submitted.connect(self._send_typed_message)
        self.command_input.textChanged.connect(self._update_action_state)
        composer_layout.addWidget(self.command_input, 1)

        self.speaker_button = QPushButton()
        self.speaker_button.setObjectName("speakerButton")
        self.speaker_button.setCheckable(True)
        self.speaker_button.setIconSize(QSize(20, 20))
        self.speaker_button.setFixedSize(44, 44)
        self.speaker_button.clicked.connect(self._toggle_speech_mute)
        composer_layout.addWidget(self.speaker_button)

        self.record_button = QPushButton()
        self.record_button.setObjectName("micButton")
        self.record_button.setToolTip("Start / stop recording")
        self.record_button.setIcon(self._microphone_icon())
        self.record_button.setIconSize(QSize(20, 20))
        self.record_button.setFixedSize(44, 44)
        self.record_button.clicked.connect(self._on_record)
        composer_layout.addWidget(self.record_button)

        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setToolTip("Send message")
        self.send_button.setIcon(self._send_icon())
        self.send_button.setIconSize(QSize(20, 20))
        self.send_button.setFixedSize(44, 44)
        self.send_button.clicked.connect(self._send_typed_message)
        composer_layout.addWidget(self.send_button)

        outer_layout.addWidget(composer_panel)
        return outer

    def _tinted_icon(self, path, color):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return QIcon()

        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.transparent)

        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color))
        painter.end()

        return QIcon(tinted)

    def _microphone_icon(self):
        icon = self._tinted_icon(MICROPHONE_ICON_PATH, "#d4d4d8")
        if icon.isNull():
            return QApplication.style().standardIcon(QStyle.SP_MediaPlay)

        return icon

    def _send_icon(self):
        icon = self._tinted_icon(SEND_ICON_PATH, "#d4d4d8")
        if icon.isNull():
            return QApplication.style().standardIcon(QStyle.SP_ArrowForward)

        return icon

    def _speaker_icon(self, muted):
        path = MUTE_ICON_PATH if muted else SPEAKER_ICON_PATH
        color = "#fca5a5" if muted else "#d4d4d8"
        icon = self._tinted_icon(path, color)
        if icon.isNull():
            fallback = getattr(QStyle, "SP_MediaVolume", QStyle.SP_MediaPlay)
            return QApplication.style().standardIcon(fallback)

        return icon

    def _update_speech_mute_controls(self):
        if hasattr(self, "speaker_button"):
            self.speaker_button.setChecked(self.is_speech_muted)
            self.speaker_button.setIcon(self._speaker_icon(self.is_speech_muted))
            self.speaker_button.setToolTip(
                "Unmute voice" if self.is_speech_muted else "Mute voice"
            )

        if hasattr(self, "mute_speech_action"):
            self.mute_speech_action.blockSignals(True)
            self.mute_speech_action.setChecked(self.is_speech_muted)
            self.mute_speech_action.blockSignals(False)

    def _toggle_speech_mute(self, checked=None):
        muted = not self.is_speech_muted if checked is None else bool(checked)
        self._set_speech_muted(muted)

    def _set_speech_muted(self, muted):
        muted = bool(muted)
        if self.is_speech_muted == muted:
            self._update_speech_mute_controls()
            return

        self.is_speech_muted = muted
        if muted:
            self.speech_enabled.clear()
            if self.is_ai_speaking or is_speaking():
                stop_speaking()
                self.is_ai_speaking = False
                self._set_status("Muted")
        else:
            self.speech_enabled.set()
            if not self.is_recording and self.worker is None:
                self._set_status("Ready")

        self._update_speech_mute_controls()
        self._update_action_state()

    def _set_record_button_idle(self):
        self.record_button.setIcon(self._microphone_icon())
        self.record_button.setStyleSheet(
            "QPushButton#micButton { background-color: #1e1e1e; border: none; border-radius: 10px; }"
            "QPushButton#micButton:hover { background-color: #2a2a2a; }"
        )

    def _set_record_button_recording(self):
        self.record_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaStop))
        self.record_button.setStyleSheet(
            "QPushButton#micButton { background-color: #3b0a0a; border: 1.5px solid #ef4444; border-radius: 10px; }"
            "QPushButton#micButton:hover { background-color: #4a0d0d; border-color: #f87171; }"
        )

    def _setup_styles(self):
        self.setStyleSheet("""
            QWidget#appRoot {
                background-color: #0f0f0f;
                color: #ececec;
                font-family: "Segoe UI", "Inter", system-ui;
                font-size: 14px;
            }

            QFrame#headerBar {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1f1f1f;
            }

            QLabel#headerName {
                color: #f5f5f5;
                font-size: 14px;
                font-weight: 700;
            }

            QLabel#statusDot {
                background-color: #22c55e;
                border-radius: 4px;
            }

            QLabel#headerStatus {
                color: #9ca3af;
                font-size: 11px;
                font-weight: 500;
            }

            QScrollArea#chatArea, QWidget#chatAreaViewport, QWidget#chatViewport {
                background-color: #0f0f0f;
                border: none;
                color: #d1d5db;
            }

            QWidget#composerWrapper {
                background-color: #0f0f0f;
            }

            QFrame#composerPanel {
                background-color: #1a1a1a;
                border: 1px solid #262626;
                border-radius: 14px;
            }

            QPlainTextEdit#commandInput {
                background-color: transparent;
                border: none;
                color: #e5e5e5;
                font-size: 14px;
                padding: 6px 2px 6px 8px;
                selection-background-color: #1d4ed8;
            }

            QPlainTextEdit#commandInput:focus {
                border: none;
            }

            QPushButton#sendButton, QPushButton#speakerButton {
                background-color: #2a2a2a;
                border: none;
                border-radius: 10px;
                color: #9ca3af;
            }

            QPushButton#sendButton:hover, QPushButton#speakerButton:hover {
                background-color: #2563eb;
                color: #ffffff;
            }

            QPushButton#sendButton:pressed, QPushButton#speakerButton:pressed {
                background-color: #1d4ed8;
            }

            QPushButton#sendButton:disabled, QPushButton#speakerButton:disabled {
                background-color: #1e1e1e;
                color: #3a3a3a;
            }

            QPushButton#speakerButton:checked {
                background-color: #3b0a0a;
                border: 1.5px solid #ef4444;
            }

            QPushButton#speakerButton:checked:hover {
                background-color: #4a0d0d;
                border-color: #f87171;
            }

            QPushButton#micButton {
                background-color: #1e1e1e;
                border: none;
                border-radius: 10px;
            }

            QPushButton#micButton:hover {
                background-color: #2a2a2a;
            }

            QPushButton#micButton:disabled {
                background-color: #1a1a1a;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #333333;
                border-radius: 3px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #444444;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }

            QScrollBar:horizontal {
                height: 0;
            }

            QToolTip {
                background-color: #222222;
                color: #d4d4d8;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)

    def _set_status(self, status):
        display_status = status.rstrip(".")
        if hasattr(self, "status_label"):
            self.status_label.setText(display_status)

        if hasattr(self, "header_status_dot"):
            lower = display_status.lower()
            if "error" in lower or "stopping" in lower:
                color = "#ef4444"
            elif "record" in lower or "detected" in lower or "checking" in lower:
                color = "#f59e0b"
            elif "speaking" in lower or "working" in lower or "thinking" in lower:
                color = "#3b82f6"
            elif "active" in lower or "ready" in lower or "listening" in lower:
                color = "#22c55e"
            else:
                color = "#6b7280"
            self.header_status_dot.setStyleSheet(
                f"background-color: {color}; border-radius: 4px;"
            )

        self._update_action_state()

    def _update_action_state(self):
        if not hasattr(self, "send_button"):
            return

        has_text = hasattr(self, "command_input") and bool(self.command_input.toPlainText().strip())
        speaking_now = self.is_ai_speaking or is_speaking()
        can_send = (
            self.brain is not None
            and not self.is_recording
            and has_text
            and (self.worker is None or speaking_now)
        )
        self.send_button.setEnabled(can_send)

        if hasattr(self, "command_input"):
            self.command_input.setEnabled(self.brain is not None and not self._really_quitting)

    def _set_record_button_enabled(self, enabled):
        if not hasattr(self, "record_button"):
            return

        can_use_voice = self.voice_input_available or self.is_recording
        self.record_button.setEnabled(bool(enabled) and can_use_voice and not self._really_quitting)

    def _send_typed_message(self):
        if not hasattr(self, "command_input"):
            return

        text = self.command_input.toPlainText().strip()
        if not text:
            self._update_action_state()
            return

        if self.brain is None or self.is_recording:
            self._update_action_state()
            return

        if self.worker is not None:
            speaking_now = self.is_ai_speaking or is_speaking()
            if not speaking_now:
                self._update_action_state()
                return

            self._set_status("Stopping speech")
            old_worker = self.worker
            if not self._interrupt_current_speech_worker():
                self._update_action_state()
                return

            if old_worker is not None and self.worker is old_worker:
                self.worker = None
                old_worker.deleteLater()

            self._pending_wake_transcript = None

        self.command_input.clear()
        self._start_text_command(text)

    def _setup_tray(self):
        QApplication.instance().setQuitOnLastWindowClosed(False)

        self.tray_icon = QSystemTrayIcon(self._create_tray_icon(), self)
        self.tray_icon.setToolTip(ASSISTANT_NAME)
        self.tray_icon.activated.connect(self._on_tray_activated)

        tray_menu = QMenu(self)

        show_action = QAction(f"Show {ASSISTANT_NAME}", self)
        show_action.triggered.connect(self._show_from_tray)
        tray_menu.addAction(show_action)

        hide_action = QAction(f"Hide {ASSISTANT_NAME}", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)

        self.mute_speech_action = QAction("Mute voice", self)
        self.mute_speech_action.setCheckable(True)
        self.mute_speech_action.triggered.connect(self._set_speech_muted)
        tray_menu.addAction(self.mute_speech_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit_app)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self._update_speech_mute_controls()
        self.tray_icon.show()

    def _create_tray_icon(self):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#20242b"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
        painter.setPen(QColor("#7dd3fc"))
        painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "M")
        painter.end()

        return QIcon(pixmap)

    def _setup_backend(self):
        self.brain = Brain()
        try:
            self.brain.setup()
        except Exception as e:
            self.brain = None
            self._add_message("System", f"LLM setup failed: {e}")
            self._set_status("LLM setup failed")
            self._set_record_button_enabled(False)
            self._update_action_state()
            return

        if self.brain.setup_message:
            self._add_message("System", self.brain.setup_message)

        if not ENABLE_VOICE_INPUT:
            self.voice_input_available = False
            self._set_record_button_idle()
            self._set_record_button_enabled(False)
            self._add_message("System", "Voice input is disabled. Text chat is ready.")
            preload_kokoro()
            self._set_status("Ready")
            self._update_action_state()
            return

        self.stt = STT()
        try:
            self.stt.load_model()
            self.voice_input_available = True
            if self.stt.startup_warning:
                self._add_message("System", self.stt.startup_warning)
        except Exception as e:
            self.stt = None
            self.voice_input_available = False
            self._set_record_button_idle()
            self._set_record_button_enabled(False)
            self._add_message(
                "System",
                "Voice input could not start, but text chat is still available.",
            )
            preload_kokoro()
            self._set_status("Ready")
            self._update_action_state()
            return

        preload_kokoro()
        self._setup_wake_word()
        self._set_record_button_enabled(True)
        self._set_status("Ready")

    def _setup_wake_word(self):
        if not ENABLE_WAKE_WORD or self.wake_disabled_for_session:
            return

        self.wake_listener = WakeWordListener(self.stt)
        self.wake_listener.status_changed.connect(self._on_wake_status)
        self.wake_listener.utterance_accepted.connect(self._on_wake_utterance_accepted)
        self.wake_listener.utterance_rejected.connect(self._on_wake_utterance_rejected)
        self.wake_listener.session_ended.connect(self._on_session_ended)
        self.wake_listener.error.connect(self._on_wake_error)
        self.wake_listener.start()

    def _on_record(self):
        if not self.is_recording:
            self._pause_wake_listener(wait=True)
            if self.is_ai_speaking or is_speaking():
                if not self._interrupt_current_speech_worker():
                    self._set_status("Stopping speech")
                    self._set_record_button_enabled(True)
                    return
            self._start_recording()
        else:
            try:
                audio, duration = self.recorder.stop()
            except Exception as e:
                print(f"[GUI] Microphone stop failed: {e}")
                self.is_recording = False
                self._disable_voice_for_session(
                    "Voice input could not start, but text chat is still available."
                )
                self._set_status("Ready")
                return

            self.is_recording = False
            self._set_record_button_idle()
            self.record_button.setEnabled(False)

            if Recorder.is_too_short(duration):
                self._set_status("Ready")
                self._set_record_button_enabled(True)
                self._resume_wake_listener()
                return

            wav_path = Recorder.save_temp_wav(audio)
            self._start_transcription(wav_path)

    def _chat_bubble_max_width(self):
        if not hasattr(self, "chat_area"):
            return 520

        viewport_width = self.chat_area.viewport().width()
        if viewport_width <= 0:
            viewport_width = self.width()

        available_width = max(0, viewport_width - 32)
        return max(260, int(available_width * 0.48))

    def _chat_bubble_width(self, role_label, text_label, role, text, max_width):
        role_metrics = QFontMetrics(role_label.font())
        text_metrics = QFontMetrics(text_label.font())
        text_lines = text.splitlines() or [""]
        content_width = max(
            role_metrics.horizontalAdvance(role),
            max(text_metrics.horizontalAdvance(line) for line in text_lines),
        )

        return min(max_width, max(64, content_width + 28))

    def _wrap_long_text_chunks(self, text):
        wrapped_chunks = []
        for chunk in text.split(" "):
            if len(chunk) > 24:
                wrapped_chunks.append("\u200b".join(chunk))
            else:
                wrapped_chunks.append(chunk)

        return " ".join(wrapped_chunks)

    def _update_message_bubble_widths(self):
        if not hasattr(self, "_message_bubbles"):
            return

        max_bubble_width = self._chat_bubble_max_width()
        for bubble, role_label, text_label, role, text in self._message_bubbles:
            bubble_width = self._chat_bubble_width(
                role_label,
                text_label,
                role,
                text,
                max_bubble_width,
            )
            label_width = max(36, bubble_width - 28)
            bubble.setFixedWidth(bubble_width)
            text_label.setFixedWidth(label_width)
            role_label.setMaximumWidth(label_width)
            bubble.setMaximumWidth(bubble_width)
            role_label.updateGeometry()
            text_label.updateGeometry()
            bubble.updateGeometry()

    def _add_message(self, role, text):
        if role == "You":
            align = "right"
            bubble_bg = "#1c3150"
            border = "#2d4a77"
            role_color = "#93c5fd"
            text_color = "#e0f0ff"
        elif role == "System":
            align = "center"
            bubble_bg = "#1a1a1a"
            border = "#2e2e2e"
            role_color = "#71717a"
            text_color = "#a1a1aa"
        else:
            align = "left"
            bubble_bg = "#1c1c1c"
            border = "#2a2a2a"
            role_color = "#60a5fa"
            text_color = "#d4d4d8"

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        bubble = QFrame()
        bubble.setObjectName("messageBubble")
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        bubble.setStyleSheet(
            f"""
            QFrame#messageBubble {{
                background-color: {bubble_bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            """
        )

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 10, 12, 10)
        bubble_layout.setSpacing(4)

        role_label = QLabel(role)
        role_label.setTextFormat(Qt.PlainText)
        role_label.setStyleSheet(
            f"color: {role_color}; font-size: 11px; font-weight: 700;"
        )

        text_label = QLabel(self._wrap_long_text_chunks(text))
        text_label.setTextFormat(Qt.PlainText)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        text_label.setStyleSheet(f"color: {text_color}; font-size: 14px;")

        bubble_layout.addWidget(role_label)
        bubble_layout.addWidget(text_label)

        if align == "right":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        elif align == "center":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)

        self._message_bubbles.append((bubble, role_label, text_label, role, text))
        self._update_message_bubble_widths()
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)

        QTimer.singleShot(
            0,
            lambda: self.chat_area.verticalScrollBar().setValue(
                self.chat_area.verticalScrollBar().maximum()
            ),
        )

    def _start_recording(self):
        try:
            self.recorder.start()
        except Exception as e:
            print(f"[GUI] Microphone start failed: {e}")
            self.is_recording = False
            self._disable_voice_for_session(
                "Voice input could not start, but text chat is still available."
            )
            self._set_status("Ready")
            return

        self.is_recording = True
        self._set_status("Recording")
        self._set_record_button_recording()
        self._set_record_button_enabled(True)

    def _start_transcription(self, wav_path, source="manual"):
        if self.stt is None:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            self._add_message("System", "Voice input is unavailable. Use text chat instead.")
            self._set_status("Ready")
            self._set_record_button_idle()
            self._set_record_button_enabled(False)
            self._resume_wake_listener()
            return

        self._pause_wake_listener(wait=True)
        self._set_record_button_idle()
        self.record_button.setEnabled(False)
        self._update_action_state()

        self.worker = TranscriptionWorker(
            wav_path,
            self.brain,
            self.stt,
            self.speech_enabled,
            source=source,
        )
        self.worker.status_changed.connect(self._on_worker_status)
        self.worker.message_received.connect(self._add_message)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _start_text_command(self, text):
        self._pause_wake_listener(wait=True)
        self._set_record_button_idle()
        self.record_button.setEnabled(False)
        self._update_action_state()

        self.worker = TextCommandWorker(text, self.brain, self.speech_enabled)
        self.worker.status_changed.connect(self._on_worker_status)
        self.worker.message_received.connect(self._add_message)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.start()

    def _on_worker_status(self, status):
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return

        if status in {"Speaking...", "Speaking"}:
            self.is_ai_speaking = True
            if WAKE_ALLOW_BARGE_IN:
                print("[Wake] barge-in enabled during TTS")
                self._resume_wake_listener()
            else:
                self._pause_wake_listener(wait=True)
            self._set_status("Speaking")
            self._set_record_button_idle()
            self._set_record_button_enabled(True)
            return

        if status == "Thinking...":
            status = "Thinking"

        if status == "Ready":
            self.is_ai_speaking = False
            if self.is_recording:
                return

        self._set_status(status)

    def _on_worker_done(self):
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return

        self.is_ai_speaking = False
        old_worker = self.worker
        self.worker = None
        if old_worker is not None:
            old_worker.deleteLater()

        if self._pending_wake_transcript:
            transcript = self._pending_wake_transcript
            self._pending_wake_transcript = None
            self._mark_wake_command_activity()
            self._start_text_command(transcript)
            return

        if self._active_session:
            self._last_active_session_time = time.time()
            remaining = ACTIVE_SESSION_TIMEOUT_SECONDS * 1000
            self._session_timer.start(int(remaining))

        if not self.is_recording:
            self._set_record_button_idle()
            self._set_record_button_enabled(True)
            self._set_status("Ready")
            self._resume_wake_listener()

    def _on_wake_status(self, status):
        if self._really_quitting or self.is_recording or self.worker is not None or self.is_ai_speaking:
            return

        self.is_wake_recording = status in {"Speech detected", "Checking wake phrase"}
        self._set_status(status)

    def _on_wake_utterance_accepted(self, transcript):
        self.is_wake_recording = False

        speaking_now = self.is_ai_speaking or is_speaking()
        barge_in_active = WAKE_ALLOW_BARGE_IN and speaking_now

        if self._really_quitting or self.is_recording:
            return

        if self.worker is not None and not barge_in_active:
            return

        if speaking_now:
            print("[Wake] barge-in accepted; interrupting current TTS")
            print("[Wake] TTS interrupted: yes")
            had_worker = self.worker is not None
            if not self._interrupt_current_speech_worker():
                self._set_status("Stopping speech")
                return
            if had_worker:
                self._pending_wake_transcript = transcript
                return
            self._mark_wake_command_activity()
            self._start_text_command(transcript)
            return

        print("[Wake] TTS interrupted: no")
        self._mark_wake_command_activity()

        self._start_text_command(transcript)

    def _on_wake_utterance_rejected(self):
        self.is_wake_recording = False
        if not self.is_recording and self.worker is None:
            if self._active_session:
                self._set_status("Active session")
            else:
                self._set_status("Passive listening")
            self._set_record_button_idle()
            self._set_record_button_enabled(True)
            self._resume_wake_listener()

    def _on_wake_error(self, message):
        print(f"[Wake] {message}")
        voice_failed = message.startswith(("Wake transcription failed:", "Wake listener failed:"))
        if voice_failed:
            self._disable_voice_for_session(
                "Voice input could not start, but text chat is still available."
            )
        else:
            self._disable_wake_for_session(
                "Wake listening stopped for this session, but manual microphone and text chat are still available."
            )
        if not self.is_recording and self.worker is None and not self.is_ai_speaking:
            self.is_wake_recording = False
            self._set_status("Ready")
            self._set_record_button_enabled(self.voice_input_available)

    def _disable_wake_for_session(self, message):
        if self.wake_disabled_for_session:
            return

        self.wake_disabled_for_session = True
        self.is_wake_recording = False
        if self.wake_listener is not None:
            self.wake_listener.stop()
        self._add_message("System", message)

    def _disable_voice_for_session(self, message):
        self.voice_input_available = False
        self.stt = None
        self._disable_wake_for_session(message)
        self._set_record_button_idle()
        self._set_record_button_enabled(False)

    def _pause_wake_listener(self, wait=False):
        if self.wake_listener is not None and self.wake_listener.isRunning():
            self.wake_listener.pause(wait=wait)

    def _resume_wake_listener(self):
        if not self._can_resume_wake_listener():
            return

        self.wake_listener.resume()
        if not self.is_ai_speaking and not is_speaking():
            if self._active_session:
                self._set_status("Active session")
            else:
                self._set_status("Passive listening")

    def _can_resume_wake_listener(self):
        if (
            self.wake_listener is None
            or not self.wake_listener.isRunning()
            or self.wake_disabled_for_session
            or self._really_quitting
            or self.is_recording
            or self.is_wake_recording
        ):
            return False

        speaking_now = self.is_ai_speaking or is_speaking()
        if self.worker is not None and not (WAKE_ALLOW_BARGE_IN and speaking_now):
            return False

        if speaking_now and not WAKE_ALLOW_BARGE_IN:
            return False

        return True

    def _interrupt_current_speech_worker(self):
        old_worker = self.worker if self.worker is not None and self.worker.isRunning() else None
        if old_worker is not None:
            old_worker.requestInterruption()

        stop_speaking()
        self.is_ai_speaking = False

        if old_worker is not None:
            if not old_worker.wait(5000):
                print("[GUI] Timed out waiting for speaking worker to stop")
                return False

        return True

    def _start_active_session(self):
        if not ENABLE_ACTIVE_SESSION:
            return
        self._active_session = True
        self._last_active_session_time = time.time()
        self._session_timer.start(ACTIVE_SESSION_TIMEOUT_SECONDS * 1000)
        if self.wake_listener:
            self.wake_listener.set_active_session(True)
        self._set_status("Active session")
        print("[Session] active session started")

    def _mark_wake_command_activity(self):
        if not self._active_session and ACTIVE_SESSION_STARTS_AFTER_WAKE:
            self._start_active_session()
        elif self._active_session:
            self._last_active_session_time = time.time()
            self._session_timer.start(ACTIVE_SESSION_TIMEOUT_SECONDS * 1000)

    def _end_active_session(self, reason="manual"):
        if not self._active_session:
            return
        self._active_session = False
        self._last_active_session_time = None
        self._session_timer.stop()
        if self.wake_listener:
            self.wake_listener.set_active_session(False)
        print(f"[Session] active session ended ({reason})")
        self._add_message("System", "Active session ended.")
        if not self.is_ai_speaking and not is_speaking():
            self._set_status("Passive listening")

    def _on_session_ended(self):
        if self.is_ai_speaking or is_speaking():
            print("[Wake] TTS interrupted: yes")
            if not self._interrupt_current_speech_worker():
                self._set_status("Stopping speech")
                return
        else:
            print("[Wake] TTS interrupted: no")

        self._end_active_session("end phrase")
        self._resume_wake_listener()

    def _on_session_timeout(self):
        if not self._active_session:
            return
        if self._last_active_session_time:
            elapsed = time.time() - self._last_active_session_time
            if elapsed < ACTIVE_SESSION_TIMEOUT_SECONDS:
                remaining = int(ACTIVE_SESSION_TIMEOUT_SECONDS - elapsed) + 1
                self._session_timer.start(remaining * 1000)
                return
        print("[Session] active session timed out")
        self._end_active_session("timeout")
        if not self.is_recording and self.worker is None:
            self._resume_wake_listener()

    def _show_from_tray(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def _cleanup_runtime(self):
        if self.wake_listener is not None and self.wake_listener.isRunning():
            self.wake_listener.stop()
            self.wake_listener.wait(5000)

        if self.is_ai_speaking or is_speaking():
            stop_speaking()
            self.is_ai_speaking = False

        if self.is_recording:
            self.recorder.stop()
            self.is_recording = False

        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            stop_speaking()
            self.worker.wait(5000)

    def _quit_app(self):
        self._really_quitting = True
        self._cleanup_runtime()
        self.tray_icon.hide()
        QApplication.instance().quit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_message_bubble_widths()

    def closeEvent(self, event):
        if self._really_quitting:
            self._cleanup_runtime()
            event.accept()
            return

        event.ignore()
        self.hide()
