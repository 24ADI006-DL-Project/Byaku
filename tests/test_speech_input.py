from speech.speech_input import (
    MicrophoneUnavailableError,
    NoSpeechDetectedError,
    SpeechInput,
    SpeechRecognitionError,
    SpeechTimeoutError,
)


class FakeBackend:
    backend_name = "fake"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def listen(self, timeout, phrase_time_limit):
        self.calls.append((timeout, phrase_time_limit))
        if self.error is not None:
            raise self.error
        return self.result


def test_returns_normalized_recognized_text():
    backend = FakeBackend("  Where is my bottle?  ")
    speech_input = SpeechInput(backend=backend)

    assert speech_input.listen() == "where is my bottle"
    assert speech_input.last_status == "recognized"


def test_silence_returns_none():
    speech_input = SpeechInput(backend=FakeBackend(error=NoSpeechDetectedError()))

    assert speech_input.listen() is None
    assert speech_input.last_status == "no_speech"


def test_unrecognized_speech_returns_none():
    speech_input = SpeechInput(backend=FakeBackend(error=SpeechRecognitionError()))

    assert speech_input.listen() is None
    assert speech_input.last_status == "unrecognized"


def test_timeout_returns_none():
    speech_input = SpeechInput(backend=FakeBackend(error=SpeechTimeoutError()))

    assert speech_input.listen() is None
    assert speech_input.last_status == "timeout"


def test_microphone_error_returns_none():
    speech_input = SpeechInput(backend=FakeBackend(error=MicrophoneUnavailableError()))

    assert speech_input.listen() is None
    assert speech_input.last_status == "microphone_error"


def test_empty_recognition_result_returns_none():
    speech_input = SpeechInput(backend=FakeBackend(result="  \n "))

    assert speech_input.listen() is None
    assert speech_input.last_status == "no_speech"


def test_timeout_settings_are_forwarded_to_backend():
    backend = FakeBackend("where is the chair")
    speech_input = SpeechInput(backend=backend, timeout=3.5, phrase_time_limit=6.0)

    assert speech_input.listen() == "where is the chair"
    assert backend.calls == [(3.5, 6.0)]


def test_unexpected_backend_error_is_handled():
    speech_input = SpeechInput(backend=FakeBackend(error=RuntimeError("backend failed")))

    assert speech_input.listen() is None
    assert speech_input.last_status == "recognition_error"