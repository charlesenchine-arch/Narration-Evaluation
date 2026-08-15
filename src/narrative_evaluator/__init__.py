"""类人叙事评估器（Human-like Narrative Evaluator）。

对中文叙事文本输出与人类审美对齐的分数，供生成算法作 reward 信号。
核心实现基于 trial.txt 的三视图分布距离框架：判别距离 + 表示分布距离 + 可解释属性距离。
"""

__version__ = "0.1.0"
