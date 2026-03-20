"""Tests for learning rate and clip range schedules."""

import pytest

from golds.training.schedules import cosine_schedule, linear_schedule

# ---- Linear schedule ----


def test_linear_schedule_start():
    schedule = linear_schedule(1.0)
    assert schedule(1.0) == pytest.approx(1.0)


def test_linear_schedule_end():
    schedule = linear_schedule(1.0)
    assert schedule(0.0) == pytest.approx(0.0)


def test_linear_schedule_middle():
    schedule = linear_schedule(1.0)
    assert schedule(0.5) == pytest.approx(0.5)


# ---- Cosine schedule ----


def test_cosine_schedule_start():
    schedule = cosine_schedule(1.0)
    assert schedule(1.0) == pytest.approx(1.0)


def test_cosine_schedule_end():
    schedule = cosine_schedule(1.0)
    assert schedule(0.0) == pytest.approx(0.0, abs=1e-10)


def test_cosine_schedule_middle():
    schedule = cosine_schedule(1.0)
    assert schedule(0.5) == pytest.approx(0.5, abs=0.01)


# ---- Monotonicity ----


def test_linear_schedule_monotonic():
    schedule = linear_schedule(2.5e-4)
    values = [schedule(p) for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1]


def test_cosine_schedule_monotonic():
    schedule = cosine_schedule(2.5e-4)
    values = [schedule(p) for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1]
