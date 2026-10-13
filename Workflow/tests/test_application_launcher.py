from pathlib import Path

from Backend import run_byaku as start_live_session


def test_launcher_resolves_repository_root():
    assert start_live_session.WORKSPACE_ROOT == Path(__file__).resolve().parents[2]


def test_initial_frame_timeout_is_configurable_and_bounded(monkeypatch):
    env_name = start_live_session.INITIAL_FRAME_TIMEOUT_ENV
    monkeypatch.delenv(env_name, raising=False)
    assert start_live_session._initial_frame_timeout_seconds() == 180

    monkeypatch.setenv(env_name, "45")
    assert start_live_session._initial_frame_timeout_seconds() == 45

    monkeypatch.setenv(env_name, "9999")
    assert start_live_session._initial_frame_timeout_seconds() == 300

    monkeypatch.setenv(env_name, "0")
    assert start_live_session._initial_frame_timeout_seconds() == 1


def test_launcher_uses_configured_timeout_for_initial_frame_receive(monkeypatch):
    monkeypatch.setenv(start_live_session.INITIAL_FRAME_TIMEOUT_ENV, "75")

    class FakeWebSocket:
        timeout = None

        def recv(self, *, timeout):
            self.timeout = timeout
            return "frame_result"

    websocket = FakeWebSocket()

    assert start_live_session._receive_initial_frame(websocket) == "frame_result"
    assert websocket.timeout == 75
