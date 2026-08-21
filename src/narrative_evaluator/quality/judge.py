"""Structured rubric and parsing utilities for LLM-as-a-Judge baselines."""
from __future__ import annotations

import json
from typing import Any

from .grades import grade_for_score
from .schemas import PREFERENCE_VALUES, QUALITY_DIMENSIONS


DIMENSION_GUIDE = {
    "plot_causality": "情节与因果：事件推进、铺垫、转折和结果是否成立",
    "character": "人物塑造：动机、行为、关系和变化是否可信且一致",
    "language_style": "语言表现：表达、声音、清晰度和文体控制",
    "atmosphere_impact": "氛围与感染力：情绪、场景和阅读投入",
    "pacing": "节奏：信息密度、场景展开和收束是否合适",
    "narrative_structure_time": "叙事结构与时间：顺序、视角、时距和时间组织",
    "originality": "新颖性：构思、意象和处理方式是否避免套板",
}


def build_pairwise_judge_prompt(text_a: str, text_b: str) -> str:
    dimensions = "\n".join(f"- {key}: {value}" for key, value in DIMENSION_GUIDE.items())
    return f"""你是一名中文叙事编辑。请盲评下面两篇文本的整体叙事质量，不要猜测作者来源。

先形成内部连续分数（0-100）与 A/B/C/D/F 等级，再判断哪篇整体更好。维度仅作为判断依据，
不能机械等权平均；总体判断优先。必须引用文本中的具体内容，禁止只写通用套话。

ABCDF 标准：A=92—100顶级；B=80—91.99优秀；C=65—79.99合格；
D=50—64.99较弱；F=0—49.99未达标。
偏好强度：2=A明显更好，1=A稍好，0=相当，-1=B稍好，-2=B明显更好。

内部判断维度：
{dimensions}

文本 A：
{text_a}

文本 B：
{text_b}

只输出 JSON：
{{
  "preference": -2|-1|0|1|2,
  "score_a": 0-100,
  "score_b": 0-100,
  "grade_a": "A"|"B"|"C"|"D"|"F",
  "grade_b": "A"|"B"|"C"|"D"|"F",
  "primary_dimensions": ["维度键"],
  "evidence_a": "A中的具体证据",
  "evidence_b": "B中的具体证据",
  "critique": "不超过120字的比较评价"
}}"""


def parse_pairwise_judge_response(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response does not contain a JSON object")
    obj = json.loads(text[start:end + 1])
    preference = int(obj["preference"])
    if preference not in PREFERENCE_VALUES:
        raise ValueError("invalid preference")
    score_a, score_b = float(obj["score_a"]), float(obj["score_b"])
    expected_a = grade_for_score(score_a).code
    expected_b = grade_for_score(score_b).code
    if obj.get("grade_a") != expected_a or obj.get("grade_b") != expected_b:
        raise ValueError("grade does not match score thresholds")
    dimensions = list(obj.get("primary_dimensions") or [])
    unknown = set(dimensions) - set(QUALITY_DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown dimensions: {sorted(unknown)}")
    if preference > 0 and score_a <= score_b:
        raise ValueError("A preference conflicts with scores")
    if preference < 0 and score_b <= score_a:
        raise ValueError("B preference conflicts with scores")
    if preference == 0 and abs(score_a - score_b) > 8:
        raise ValueError("tie conflicts with score difference")
    return {
        "preference": preference,
        "score_a": score_a,
        "score_b": score_b,
        "grade_a": expected_a,
        "grade_b": expected_b,
        "primary_dimensions": dimensions,
        "evidence_a": str(obj.get("evidence_a", "")).strip(),
        "evidence_b": str(obj.get("evidence_b", "")).strip(),
        "critique": str(obj.get("critique", "")).strip(),
    }
