"""Tests for the RND (Random Network Distillation) module."""

from __future__ import annotations

import numpy as np
import torch

from golds.training.rnd import RNDNetwork, RunningMeanStd


class TestRNDNetwork:
    """Tests for RNDNetwork."""

    def test_forward_shape(self) -> None:
        net = RNDNetwork()
        x = torch.randn(4, 1, 84, 84)
        out = net(x)
        assert out.shape == (4, 512)

    def test_target_and_predictor_differ(self) -> None:
        target = RNDNetwork()
        predictor = RNDNetwork()
        x = torch.randn(4, 1, 84, 84)
        with torch.no_grad():
            target_out = target(x)
            predictor_out = predictor(x)
        assert not torch.allclose(target_out, predictor_out)


class TestRunningMeanStd:
    """Tests for RunningMeanStd."""

    def test_tracks_mean_and_var(self) -> None:
        rms = RunningMeanStd(shape=(3,))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        rms.update(data)
        np.testing.assert_allclose(rms.mean, [4.0, 5.0, 6.0], atol=1e-2)
        expected_var = np.var(data, axis=0)
        np.testing.assert_allclose(rms.var, expected_var, atol=1e-1)
