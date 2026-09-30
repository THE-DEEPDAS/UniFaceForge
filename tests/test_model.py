import torch
from src.unifaceforge.model import DCTBandBlock, PhaseAmplitudeAttention


def test_frequency_modules_are_real_and_shape_preserving():
    x = torch.randn(2, 8, 8, 8)
    assert DCTBandBlock(8)(x).shape == x.shape
    y = PhaseAmplitudeAttention(8)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
