import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

from speech.tts import TextToSpeech


class FakeEngine:
    def __init__(self):
        self.messages = []
        self.properties = {}
        self.fail_on = None
        self.active_calls = 0
        self.max_active_calls = 0

    def setProperty(self, name, value):
        self.properties[name] = value

    def say(self, text):
        if self.fail_on == "say":
            raise RuntimeError("backend say failed")
        self.messages.append(text)

    def runAndWait(self):
        if self.fail_on == "run":
            raise RuntimeError("backend playback failed")
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(0.01)
        self.active_calls -= 1

    def stop(self):
        pass


def test_initializes_local_backend(monkeypatch):
    engine = FakeEngine()
    backend = types.ModuleType("pyttsx3")
    backend.init = lambda: engine
    monkeypatch.setitem(sys.modules, "pyttsx3", backend)

    tts = TextToSpeech()

    assert tts.engine is engine
    assert tts.backend_name == "pyttsx3"
    assert engine.properties["rate"] == 175
    assert engine.properties["volume"] == 1.0


def test_accepts_text_through_replaceable_backend():
    engine = FakeEngine()
    tts = TextToSpeech(engine=engine)

    assert tts.speak("  Your bottle is close and on your left.  ") is True
    assert engine.messages == ["Your bottle is close and on your left."]


def test_empty_text_is_ignored():
    engine = FakeEngine()
    tts = TextToSpeech(engine=engine)

    assert tts.speak("") is False
    assert tts.speak("   ") is False
    assert tts.speak(None) is False
    assert engine.messages == []


def test_repeated_text_is_suppressed_until_cooldown():
    engine = FakeEngine()
    now = [10.0]
    tts = TextToSpeech(engine=engine, repeat_cooldown=2.0, clock=lambda: now[0])

    assert tts.speak("Bottle on your left.") is True
    assert tts.speak(" Bottle on your left. ") is False
    now[0] += 2.1
    assert tts.speak("Bottle on your left.") is True
    assert engine.messages == ["Bottle on your left.", "Bottle on your left."]


def test_multiple_different_requests_are_spoken_in_order():
    engine = FakeEngine()
    tts = TextToSpeech(engine=engine)

    assert tts.speak("A chair is ahead.") is True
    assert tts.speak("A person is nearby.") is True

    assert engine.messages == ["A chair is ahead.", "A person is nearby."]


def test_concurrent_requests_do_not_overlap():
    engine = FakeEngine()
    tts = TextToSpeech(engine=engine)
    messages = ["First message.", "Second message.", "Third message."]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(tts.speak, messages))

    assert results == [True, True, True]
    assert engine.max_active_calls == 1
    assert set(engine.messages) == set(messages)


def test_backend_errors_are_reported_without_raising():
    engine = FakeEngine()
    tts = TextToSpeech(engine=engine)
    engine.fail_on = "say"

    assert tts.speak("Hello.") is False

    engine.fail_on = "run"
    assert tts.speak("Another message.") is False


def test_initialization_error_disables_speech_gracefully(monkeypatch):
    backend = types.ModuleType("pyttsx3")

    def fail_initialization():
        raise RuntimeError("no speech engine available")

    backend.init = fail_initialization
    monkeypatch.setitem(sys.modules, "pyttsx3", backend)
    tts = TextToSpeech()

    assert tts.engine is None
    assert tts.speak("Hello.") is False
    assert tts.initialization_error is not None