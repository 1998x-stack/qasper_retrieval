"""
评估模块初始化文件
Evaluation module initialization file

该模块提供检索和文本生成的评估功能
This module provides evaluation functionality for retrieval and text generation
"""

from .evaluator import MetricsCalculator, RetrievalEvaluator

__all__ = [
    'MetricsCalculator',
    'RetrievalEvaluator'
]

__version__ = "1.0.0"
__author__ = "QASPER Retrieval Team"