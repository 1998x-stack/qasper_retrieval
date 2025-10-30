"""
工具模块初始化文件
Utils module initialization file

该模块提供日志管理和通用工具函数
This module provides logging management and common utility functions
"""

from .logger import get_logger, setup_logging, LoggerManager
from .common import (
    Timer,
    MemoryMonitor,
    set_random_seed,
    ensure_dir,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    load_yaml,
    get_file_hash,
    normalize_scores,
    get_device,
    chunk_list,
    suppress_stdout,
    validate_config
)

__all__ = [
    # 日志相关 / Logging related
    'get_logger',
    'setup_logging',
    'LoggerManager',
    
    # 通用工具 / Common utilities
    'Timer',
    'MemoryMonitor',
    'set_random_seed',
    'ensure_dir',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'load_yaml',
    'get_file_hash',
    'normalize_scores',
    'get_device',
    'chunk_list',
    'suppress_stdout',
    'validate_config'
]

__version__ = "1.0.0"
__author__ = "QASPER Retrieval Team"