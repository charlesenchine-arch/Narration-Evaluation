"""分布距离指标：MMD / Wasserstein / Fréchet / 覆盖率。

输入是两组文本的表示向量（numpy 数组，shape = [n, d]）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """RBF 核最大均值差异（MMD²）。X/Y 形状 [n, d]。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.nan
    if sigma is None:
        # 用中位数距离启发式选核带宽
        XY = np.vstack([X, Y])
        dists = _pairwise_distances(XY)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        sigma = med
    gamma = 1.0 / (2.0 * sigma**2)
    k_xx = _rbf_kernel(X, X, gamma)
    k_yy = _rbf_kernel(Y, Y, gamma)
    k_xy = _rbf_kernel(X, Y, gamma)
    m, n = X.shape[0], Y.shape[0]
    return float(k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2.0 * k_xy.sum() / (m * n))


def _pairwise_distances(A: np.ndarray) -> np.ndarray:
    """A 形状 [n, d] → 上三角距离矩阵扁平向量（不含对角）。"""
    n = A.shape[0]
    if n <= 1:
        return np.array([])
    diff = A[:, None, :] - A[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    iu = np.triu_indices(n, k=1)
    return np.sqrt(np.maximum(d2[iu], 0.0))


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.exp(-gamma * d2)


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """一维分布的一阶 Wasserstein 距离（scipy）。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


def wasserstein_mean(X: np.ndarray, Y: np.ndarray) -> float:
    """多维度量下的近似 Wasserstein：对各维度分别算 1D Wasserstein 后取均值。

    严格的多维 Wasserstein（最优传输）计算代价高，这里采用维度级近似，
    足以反映两组表示分布的整体移动。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] == 0 or Y.shape[0] == 0 or X.shape[1] != Y.shape[1]:
        return np.nan
    d = X.shape[1]
    vals = []
    for i in range(d):
        vals.append(wasserstein_distance(X[:, i], Y[:, i]))
    return float(np.mean(vals))


def frechet_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet 距离（FID 式）：均值差异 + 协方差差异。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return np.nan
    mu_x, mu_y = X.mean(0), Y.mean(0)
    sig_x = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    sig_y = np.cov(Y, rowvar=False) + eps * np.eye(Y.shape[1])
    diff = mu_x - mu_y
    # Tr(X + Y - 2*sqrt(sqrt(X) Y sqrt(X))) via eig of sym
    covmean = _sqrtm(sig_x @ sig_y)
    return float(diff @ diff + np.trace(sig_x + sig_y - 2.0 * covmean))


def _sqrtm(A: np.ndarray) -> np.ndarray:
    """对称矩阵的平方根（特征分解法，稳定）。"""
    w, v = np.linalg.eigh((A + A.T) / 2.0)
    w = np.maximum(w, 0.0)
    return (v * np.sqrt(w)) @ v.T


def human_coverage(X: np.ndarray, Y: np.ndarray, radius: float | None = None) -> float:
    """Human coverage：人类文本表示空间被机器文本覆盖的比例。

    以每个人类样本为中心，半径内是否存在机器样本的比例。
    X=人类，Y=机器。radius 默认取人类样本对之间的中位距离。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.nan
    if radius is None:
        dxx = _pairwise_distances(X)
        radius = np.median(dxx) if dxx.size else 1.0
    # 人类样本到最近机器样本的距离
    dy = _nearest_distances(X, Y)
    return float(np.mean(dy <= radius))


def machine_only_mass(X: np.ndarray, Y: np.ndarray, radius: float | None = None) -> float:
    """Machine-only mass：机器文本落在人类空间未覆盖区域的比例。

    机器样本到最近人类样本距离大于 radius 的比例。
    X=人类，Y=机器。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.nan
    if radius is None:
        dxx = _pairwise_distances(X)
        radius = np.median(dxx) if dxx.size else 1.0
    dx = _nearest_distances(Y, X)
    return float(np.mean(dx > radius))


def _nearest_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A 中每点到 B 中最近点的欧氏距离（分块向量化）。"""
    n = A.shape[0]
    out = np.empty(n)
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = A[start:end]
        diff = block[:, None, :] - B[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        out[start:end] = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
    return out
