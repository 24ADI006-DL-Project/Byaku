from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from speech.audio_utils import audio_data_to_pcm, normalize_recognized_text


class SpeechInputError(Exception):
    """Base exception for expected speech input failures."""


class SpeechTimeoutError(SpeechInputError):
    """No speech began before the configured listening timeout."""


class NoSpeechDetectedError(SpeechInputError):
    """No recognizable words were found in the captured audio."""


class SpeechRecognitionError(SpeechInputError):
    """The backend could not decode the captured audio."""


class MicrophoneUnavailableError(SpeechInputError):
    """The configured microphone could not be opened or read."""


class SpeechBackend(Protocol):
    backend_name: str

    def listen(self, timeout: float, phrase_time_limit: float) -> str | None:
        """Capture a phrase and return its recognized text."""


class VoskSpeechBackend:
    """Offline Vosk speech-to-text backend using SpeechRecognition capture."""

    backend_name = "vosk"
    SAMPLE_RATE = 16_000

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        device_index: int | None = None,
    ) -> None:
        try:
            import speech_recognition as sr
            import vosk
        except ImportError as exc:
            raise RuntimeError("Install SpeechRecognition, PyAudio, and vosk to enable speech input") from exc

        configured_path = model_path or os.environ.get("VOSK_MODEL_PATH")
        self.model_path = Path(configured_path) if configured_path else (
            Path.cwd() / "models" / "vosk-model-small-en-us-0.15"
        )
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Vosk model not found at {self.model_path}; set VOSK_MODEL_PATH to a downloaded model folder"
            )

        try:
            self._model = vosk.Model(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load Vosk model: {exc}") from exc

        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._device_index = device_index

    def listen(self, timeout: float, phrase_time_limit: float) -> str | None:
        try:
            with self._sr.Microphone(device_index=self._device_index) as source:
                audio_data = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
        except self._sr.WaitTimeoutError as exc:
            raise SpeechTimeoutError("No speech detected before timeout") from exc
        except (OSError, AttributeError) as exc:
            raise MicrophoneUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise SpeechRecognitionError(str(exc)) from exc

        try:
            from vosk import KaldiRecognizer

            recognizer = KaldiRecognizer(self._model, self.SAMPLE_RATE)
            pcm_data = audio_data_to_pcm(audio_data, self.SAMPLE_RATE, 2)
            recognizer.AcceptWaveform(pcm_data)
            result = json.loads(recognizer.FinalResult())
        except Exception as exc:
            raise SpeechRecognitionError(str(exc)) from exc

        transcript = normalize_recognized_text(result.get("text"))
        if transcript is None:
            raise NoSpeechDetectedError("No recognizable speech in captured audio")
        return transcript


class SpeechInput:
    """Capture one microphone phrase and return recognized text or None.

    By default, uses the offline Vosk backend. Provide a Vosk model directory
    through ``model_path`` or the ``VOSK_MODEL_PATH`` environment variable.
    Inject any object implementing ``listen(timeout, phrase_time_limit)`` to
    replace the speech-to-text backend.
    """

    def __init__(
        self,
        backend: SpeechBackend | Any | None = None,
        timeout: float = 5.0,
        phrase_time_limit: float = 8.0,
        model_path: str | os.PathLike[str] | None = None,
        device_index: int | None = None,
        backend_factory: Callable[..., SpeechBackend] | None = None,
    ) -> None:
        if timeout <= 0 or phrase_time_limit <= 0:
            raise ValueError("timeout and phrase_time_limit must be positive")

        self.timeout = float(timeout)
        self.phrase_time_limit = float(phrase_time_limit)
        self.backend = backend
        self.backend_name: str | None = None
        self.initialization_error: str | None = None
        self.last_error: str | None = None
        self.last_status = "ready"

        if self.backend is None:
            try:
                factory = backend_factory or VoskSpeechBackend
                self.backend = factory(model_path=model_path, device_index=device_index)
            except Exception as exc:
                self.initialization_error = str(exc)
                self.last_status = "backend_unavailable"

        if self.backend is not None:
            self.backend_name = getattr(self.backend, "backend_name", type(self.backend).__name__)

    def listen(self) -> str | None:
        """Capture and transcribe one phrase; return None when unavailable."""
        self.last_error = None
        if self.backend is None:
            self.last_status = "backend_unavailable"
            return None

        try:
            result = self.backend.listen(
                timeout=self.timeout,
                phrase_time_limit=self.phrase_time_limit,
            )
        except SpeechTimeoutError as exc:
            return self._failure("timeout", exc)
        except NoSpeechDetectedError as exc:
            return self._failure("no_speech", exc)
        except MicrophoneUnavailableError as exc:
            return self._failure("microphone_error", exc)
        except SpeechRecognitionError as exc:
            return self._failure("unrecognized", exc)
        except TimeoutError as exc:
            return self._failure("timeout", exc)
        except OSError as exc:
            return self._failure("microphone_error", exc)
        except Exception as exc:
            return self._failure("recognition_error", exc)

        transcript = normalize_recognized_text(result)
        if transcript is None:
            self.last_status = "no_speech"
            return None

        self.last_status = "recognized"
        return transcript

    def _failure(self, status: str, error: Exception) -> None:
        self.last_status = status
        self.last_error = str(error) or type(error).__name__
        return None