from pathlib import Path
from types import SimpleNamespace

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


def test_launcher_reports_active_model_metrics_on_held_out_test_split(tmp_path, capsys):
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "sample.jpg").write_bytes(b"test image fixture")
    (label_dir / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text("path: .\ntest: images/test\n", encoding="utf-8")

    class FakeModel:
        names = {0: "person"}
        validation_args = None

        def val(self, **kwargs):
            self.validation_args = kwargs
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(B)": 0.75,
                    "metrics/recall(B)": 0.60,
                    "metrics/mAP50(B)": 0.50,
                    "metrics/mAP50-95(B)": 0.35,
                }
            )

    model = FakeModel()
    weights = tmp_path / "active.pt"
    weights.write_bytes(b"checkpoint fixture")

    start_live_session._print_model_accuracy(model, weights, dataset_root=tmp_path)

    output = capsys.readouterr().out
    assert model.validation_args["split"] == "test"
    assert model.validation_args["device"] == "cpu"
    assert "test=1 images, 1 boxes, 1 supported classes" in output
    assert "Precision=75.00%" in output
    assert "Recall=60.00%" in output
    assert "mAP@0.5=50.00%" in output
    assert "mAP@0.5:0.95=35.00%" in output


def test_launcher_explicitly_reports_unavailable_accuracy_data(tmp_path, capsys):
    start_live_session._print_model_accuracy(
        SimpleNamespace(names={0: "person"}),
        tmp_path / "active.pt",
        dataset_root=tmp_path,
    )

    output = capsys.readouterr()
    assert "[Model Accuracy] UNAVAILABLE: FileNotFoundError" in output.err
