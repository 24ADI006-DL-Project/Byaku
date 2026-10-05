from __future__ import annotations

import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import Any


class TextToSpeech:
    """Synchronous, replaceable text-to-speech interface.

    The default backend is pyttsx3, which uses the operating system's local
    speech engine. Calls are serialized, so speech requests never overlap.
    """

    def __init__(
        self,
        engine: Any = None,
        rate: int = 175,
        volume: float = 1.0,
        repeat_cooldown: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isfinite(repeat_cooldown) or repeat_cooldown < 0:
            raise ValueError("repeat_cooldown must be finite and non-negative")

        self.engine = engine
        self.backend_name: str | None = None
        self.initialization_error: str | None = None
        self.repeat_cooldown = float(repeat_cooldown)
        self._clock = clock
        self._lock = threading.Lock()
        self._last_text: str | None = None
        self._last_spoken_at: float | None = None

        if self.engine is None:
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
                self.backend_name = "pyttsx3"
            except Exception as exc:
                self.initialization_error = str(exc)
                return
        else:
            self.backend_name = getattr(self.engine, "backend_name", type(self.engine).__name__)

        self._configure_engine("rate", rate)
        self._configure_engine("volume", volume)

    def speak(self, text: Any) -> bool:
        """Speak non-empty text; return False when ignored or playback fails."""
        if not isinstance(text, str) or not text.strip() or self.engine is None:
            return False

        clean_text = text.strip()
        repeat_key = clean_text.casefold()
        with self._lock:
            now = self._clock()
            if (
                self._last_text == repeat_key
                and self._last_spoken_at is not None
                and now - self._last_spoken_at < self.repeat_cooldown
            ):
                return False

            try:
                self.engine.say(clean_text)
                self.engine.runAndWait()
            except Exception:
                try:
                    self.engine.stop()
                except Exception:
                    pass
                return False

            self._last_text = repeat_key
            self._last_spoken_at = now
            return True

    def _configure_engine(self, name: str, value: Any) -> None:
        set_property = getattr(self.engine, "setProperty", None)
        if callable(set_property):
            try:
                set_property(name, value)
            except Exception:
                pass