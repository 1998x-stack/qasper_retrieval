"""
数据模块初始化文件
Data module initialization file

该模块提供数据加载和预处理功能
This module provides data loading and preprocessing functionality
"""

from .dataset_loader import QASPERDatasetLoader
from .preprocessor import TextPreprocessor, QASPERPreprocessor

__all__ = [
    'QASPERDatasetLoader',
    'TextPreprocessor',
    'QASPERPreprocessor'
]

__version__ = "1.0.0"
__author__ = "QASPER Retrieval Team"