"""
QASPER检索评估系统源代码模块
QASPER Retrieval Evaluation System Source Code Module

该模块包含数据处理、检索、评估等核心功能
This module contains core functionality for data processing, retrieval, and evaluation
"""

__version__ = "1.0.0"
__author__ = "QASPER Retrieval Team"
__description__ = "QASPER Retrieval Evaluation System - A comprehensive system for evaluating information retrieval methods on the QASPER dataset"

# 导入主要组件 / Import main components
from .data import QASPERDatasetLoader, QASPERPreprocessor, TextPreprocessor
from .retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever
from .evaluation import MetricsCalculator, RetrievalEvaluator
from .utils import get_logger, setup_logging, Timer, MemoryMonitor

__all__ = [
    # 数据处理 / Data processing
    'QASPERDatasetLoader',
    'QASPERPreprocessor', 
    'TextPreprocessor',
    
    # 检索方法 / Retrieval methods
    'BM25Retriever',
    'EmbeddingRetriever',
    'HybridRetriever',
    
    # 评估工具 / Evaluation tools
    'MetricsCalculator',
    'RetrievalEvaluator',
    
    # 工具函数 / Utility functions
    'get_logger',
    'setup_logging',
    'Timer',
    'MemoryMonitor'
]

def get_version() -> str:
    """
    获取版本号
    Get version number
    
    Returns:
        版本号字符串 / Version number string
    """
    return __version__

def get_system_info() -> dict:
    """
    获取系统信息
    Get system information
    
    Returns:
        系统信息字典 / System information dictionary
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__
    }