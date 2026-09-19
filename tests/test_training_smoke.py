import torch

from plant_disease.training.train import build_model


def test_build_model_v2_correct_output_dim():
    model = build_model(num_classes=61)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 61)
