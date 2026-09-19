from io import BytesIO

import pytest
import torch
from PIL import Image

from plant_disease import errors
from plant_disease.data.class_map import ClassInfo
from plant_disease.model import InferenceModel, _select_device


def _png_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(120, 200, 80))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def patched_model(monkeypatch, tmp_path):
    """Construct InferenceModel without real weights or class file."""
    monkeypatch.setattr(InferenceModel, "_load_weights", lambda self, p: None)
    monkeypatch.setattr(
        "plant_disease.model.load_class_map",
        lambda _p: [ClassInfo("番茄", "患病", "一般", "番茄早疫病")],
    )
    return InferenceModel(weights_path=tmp_path / "fake.pth", classes_txt=tmp_path / "fake.txt")


def test_predict_returns_expected_keys(patched_model):
    out = patched_model.predict(_png_bytes())
    assert set(out.keys()) == {
        "class_id",
        "probability",
        "plant_class",
        "health_status",
        "disease_name",
        "disease_degree",
    }
    assert isinstance(out["class_id"], int)
    assert 0.0 <= out["probability"] <= 1.0


def test_predict_uses_class_map_fallback_when_idx_out_of_range(patched_model):
    out = patched_model.predict(_png_bytes())
    # mapping has 1 entry but the model has 61 outputs; if argmax > 0
    # we still must return a placeholder rather than crash.
    assert isinstance(out["plant_class"], str) and out["plant_class"]


def test_predict_invalid_image_raises_inference_error(patched_model):
    with pytest.raises(errors.InferenceError):
        patched_model.predict(b"not an image")


def test_select_device_cpu_when_no_cuda_no_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)
    assert _select_device().type == "cpu"


def test_load_class_info_falls_back_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(InferenceModel, "_load_weights", lambda self, p: None)
    m = InferenceModel(weights_path=tmp_path / "fake.pth", classes_txt=tmp_path / "missing.txt")
    assert m.class_info == []


def test_missing_weights_path_raises_inference_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plant_disease.model.load_class_map",
        lambda _p: [],
    )
    with pytest.raises(errors.InferenceError):
        InferenceModel(
            weights_path=tmp_path / "does-not-exist.pth", classes_txt=tmp_path / "classes.txt"
        )
