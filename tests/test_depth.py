import numpy as np
import pytest
import torch

import depth.depth_estimator as depth_module


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    loaded_model = None

    @classmethod
    def from_pretrained(cls, model_name):
        cls.loaded_model = model_name
        return cls()

    def __call__(self, images, return_tensors):
        return FakeBatch(pixel_values=torch.zeros((1, 3, 8, 8)))


class FakeModel:
    loaded_model = None
    selected_device = None

    @classmethod
    def from_pretrained(cls, model_name):
        cls.loaded_model = model_name
        return cls()

    def to(self, device):
        self.selected_device = str(device)
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        return type("Output", (), {"predicted_depth": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)})()


def patch_model_loading(monkeypatch):
    monkeypatch.setattr(depth_module, "AutoImageProcessor", FakeProcessor)
    monkeypatch.setattr(depth_module, "AutoModelForDepthEstimation", FakeModel)
    FakeProcessor.loaded_model = None
    FakeModel.loaded_model = None
    FakeModel.selected_device = None


def test_estimator_initializes_pretrained_model(monkeypatch):
    patch_model_loading(monkeypatch)

    estimator = depth_module.DepthEstimator(model_name="test/depth-model")

    assert estimator.model is not None
    assert estimator.processor is not None
    assert FakeProcessor.loaded_model == "test/depth-model"
    assert FakeModel.loaded_model == "test/depth-model"


def test_estimate_depth_returns_float_map_matching_frame_size(monkeypatch):
    patch_model_loading(monkeypatch)
    estimator = depth_module.DepthEstimator()
    frame = np.zeros((120, 200, 3), dtype=np.uint8)

    depth_map = estimator.estimate_depth(frame)

    assert depth_map.shape == (120, 200)
    assert depth_map.dtype == np.float32
    assert np.isfinite(depth_map).all()
    assert float(depth_map.max()) > float(depth_map.min())


def test_estimator_handles_different_image_sizes(monkeypatch):
    patch_model_loading(monkeypatch)
    estimator = depth_module.DepthEstimator()

    assert estimator.estimate_depth(np.zeros((48, 96, 3), dtype=np.uint8)).shape == (48, 96)
    assert estimator.estimate_depth(np.zeros((75, 51, 3), dtype=np.uint8)).shape == (75, 51)


def test_normalize_depth_map_returns_values_between_zero_and_one(monkeypatch):
    patch_model_loading(monkeypatch)
    estimator = depth_module.DepthEstimator()

    normalized = estimator.normalize_depth_map(np.array([[2.0, 4.0]], dtype=np.float32))

    assert normalized.dtype == np.float32
    assert normalized.tolist() == [[0.0, 1.0]]


def test_depth_value_at_pixel_and_bounds(monkeypatch):
    patch_model_loading(monkeypatch)
    estimator = depth_module.DepthEstimator()
    depth_map = np.array([[1.25, 2.5], [3.75, 5.0]], dtype=np.float32)

    assert estimator.get_depth_at_pixel(depth_map, 1, 0) == pytest.approx(2.5)
    assert estimator.get_depth_at_pixel(depth_map, 2, 0) is None
    assert estimator.get_depth_at_pixel(depth_map, -1, 0) is None


def test_invalid_or_empty_frames_return_none(monkeypatch):
    patch_model_loading(monkeypatch)
    estimator = depth_module.DepthEstimator()

    assert estimator.estimate_depth(None) is None
    assert estimator.estimate_depth(np.empty((0, 0, 3), dtype=np.uint8)) is None
    assert estimator.estimate_depth(np.zeros((24, 24), dtype=np.uint8)) is None


def test_device_selection_uses_cuda_when_available(monkeypatch):
    patch_model_loading(monkeypatch)
    monkeypatch.setattr(depth_module.torch.cuda, "is_available", lambda: True)

    estimator = depth_module.DepthEstimator()

    assert str(estimator.device) == "cuda"
    assert estimator.model.selected_device == "cuda"


def test_device_selection_falls_back_to_cpu(monkeypatch):
    patch_model_loading(monkeypatch)
    monkeypatch.setattr(depth_module.torch.cuda, "is_available", lambda: False)

    estimator = depth_module.DepthEstimator()

    assert str(estimator.device) == "cpu"
    assert estimator.model.selected_device == "cpu"
