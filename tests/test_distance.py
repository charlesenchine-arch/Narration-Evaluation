"""测试：距离指标数学正确性（无外部依赖，纯 numpy/scipy）。"""
import numpy as np
import pytest

from narrative_evaluator.metrics.distance import (
    mmd_rbf,
    wasserstein_1d,
    wasserstein_mean,
    frechet_distance,
    human_coverage,
    machine_only_mass,
)


def test_mmd_same_vs_far():
    rng = np.random.RandomState(0)
    A = rng.randn(50, 8)
    B_same = rng.randn(50, 8)
    B_far = rng.randn(50, 8) + 5.0
    assert mmd_rbf(A, B_far) > mmd_rbf(A, B_same)


def test_mmd_same_approx_zero():
    rng = np.random.RandomState(1)
    A = rng.randn(100, 4)
    B = A + rng.randn(100, 4) * 1e-4
    assert mmd_rbf(A, B) < 1e-3


def test_wasserstein_1d_shift():
    a = np.random.RandomState(0).randn(2000)
    b = a + 3.0
    assert abs(wasserstein_1d(a, b) - 3.0) < 0.2


def test_wasserstein_mean_positive():
    rng = np.random.RandomState(2)
    X = rng.randn(30, 5)
    Y = rng.randn(30, 5) + 2.0
    w = wasserstein_mean(X, Y)
    assert w > 0


def test_frechet_same_zero():
    rng = np.random.RandomState(3)
    A = rng.randn(60, 6)
    assert frechet_distance(A, A[:40]) < 1.0


def test_frechet_far_large():
    rng = np.random.RandomState(4)
    A = rng.randn(60, 6)
    B = rng.randn(60, 6) + 10.0
    assert frechet_distance(A, B) > 10.0


def test_coverage_direction():
    rng = np.random.RandomState(5)
    H = rng.randn(30, 8)
    G_close = H + rng.randn(30, 8) * 0.01
    G_far = rng.randn(30, 8) + 10.0
    assert human_coverage(H, G_close, radius=0.5) > human_coverage(H, G_far, radius=0.5)
    assert machine_only_mass(H, G_close, radius=0.5) < machine_only_mass(H, G_far, radius=0.5)
