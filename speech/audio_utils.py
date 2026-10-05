from __future__ import annotations

import re
from typing import Any


def audio_data_to_pcm(audio_data: Any, sample_rate: int = 16_000, sample_width: int = 2) -> bytes:
    """Convert SpeechRecognition AudioData to mono PCM for the Vosk model."""
    if sample_rate <= 0 or sample_width <= 0:
        raise ValueError("sample_rate and sample_width must be positive")
    get_raw_data = getattr(audio_data, "get_raw_data", None)
    if not callable(get_raw_data):
        raise TypeError("audio_data must provide get_raw_data()")
    return get_raw_data(convert_rate=sample_rate, convert_width=sample_width)


def normalize_recognized_text(text: Any) -> str | None:
    """Trim and case-normalize a speech transcript without interpreting it."""
    if not isinstance(text, str):
        return None
    normalized = re.sub(r"[^\w\s'-]", "", text.casefold())
    normalized = " ".join(normalized.split())
    return normalized or None