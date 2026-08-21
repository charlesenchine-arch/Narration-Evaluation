"""A/B/C/D/F public grades backed by a continuous 0--100 utility.

The default cut-points are protocol defaults, not claims about objective
literary quality.  They must be calibrated on held-out human judgments before
reporting a final benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GradeBand:
    code: str
    minimum: float
    label: str
    interpretation: str


DEFAULT_GRADE_SCALE: tuple[GradeBand, ...] = (
    GradeBand("A", 92.0, "顶级", "对高质量锚点保持稳定优势，核心环节几乎无明显短板"),
    GradeBand("B", 80.0, "优秀", "整体成熟，有清晰优势，局部问题不损害主要体验"),
    GradeBand("C", 65.0, "合格", "叙事目标成立，但存在多处可感知缺陷"),
    GradeBand("D", 50.0, "较弱", "部分成立，需要较大修改才能形成稳定阅读体验"),
    GradeBand("F", 0.0, "未达标", "核心叙事目标未成立，需结构性修改"),
)


def validate_grade_scale(scale: Sequence[GradeBand]) -> None:
    if not scale:
        raise ValueError("grade scale must not be empty")
    codes = [band.code for band in scale]
    if len(codes) != len(set(codes)):
        raise ValueError("grade codes must be unique")
    minimums = [float(band.minimum) for band in scale]
    if any(not 0.0 <= value <= 100.0 for value in minimums):
        raise ValueError("grade minimums must be in [0, 100]")
    if minimums != sorted(minimums, reverse=True):
        raise ValueError("grade scale must be ordered from highest to lowest")
    if minimums[-1] != 0.0:
        raise ValueError("lowest grade band must begin at 0")


def grade_for_score(
    score: float,
    scale: Sequence[GradeBand] = DEFAULT_GRADE_SCALE,
) -> GradeBand:
    """Map a calibrated utility score in [0, 100] to an A/B/C/D/F band."""
    validate_grade_scale(scale)
    value = float(score)
    if not 0.0 <= value <= 100.0:
        raise ValueError("score must be in [0, 100]")
    for band in scale:
        if value >= band.minimum:
            return band
    raise AssertionError("validated grade scale must cover zero")


def grade_codes(scale: Iterable[GradeBand] = DEFAULT_GRADE_SCALE) -> tuple[str, ...]:
    return tuple(band.code for band in scale)
