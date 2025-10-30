"""
通用工具函数模块
Common utilities module

提供常用的辅助功能和工具函数
Provides common utility functions and helper methods
"""

import os
import json
import pickle
import hashlib
import time
import psutil
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import yaml

from .logger import get_logger

logger = get_logger(__name__)


class Timer:
    """
    计时器类
    Timer class
    
    用于测量代码执行时间
    Used for measuring code execution time
    """
    
    def __init__(self, name: str = "Timer") -> None:
        """
        初始化计时器
        Initialize timer
        
        Args:
            name: 计时器名称 / Timer name
        """
        self.name = name
        self.start_time = None
        self.elapsed_time = None
        
    def start(self) -> 'Timer':
        """
        开始计时
        Start timing
        
        Returns:
            计时器实例 / Timer instance
        """
        self.start_time = time.time()
        logger.info(f"{self.name} 开始计时 / {self.name} started timing")
        return self
        
    def stop(self) -> float:
        """
        停止计时
        Stop timing
        
        Returns:
            经过的时间（秒）/ Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("计时器未启动 / Timer not started")
            
        self.elapsed_time = time.time() - self.start_time
        logger.info(f"{self.name} 计时结束，耗时: {self.elapsed_time:.2f}秒 / "
                   f"{self.name} timing finished, elapsed: {self.elapsed_time:.2f}s")
        return self.elapsed_time
        
    def __enter__(self) -> 'Timer':
        """
        上下文管理器入口
        Context manager entry
        """
        return self.start()
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        上下文管理器出口
        Context manager exit
        """
        self.stop()


class MemoryMonitor:
    """
    内存监控器类
    Memory monitor class
    
    用于监控系统内存使用情况
    Used for monitoring system memory usage
    """
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        获取当前内存使用情况
        Get current memory usage
        
        Returns:
            内存使用信息字典 / Memory usage information dictionary
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': process.memory_percent(),
            'system_memory_total_gb': system_memory.total / 1024 / 1024 / 1024,
            'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024,
            'system_memory_percent': system_memory.percent
        }
    
    @staticmethod
    def log_memory_usage(prefix: str = "") -> None:
        """
        记录内存使用情况到日志
        Log memory usage to logger
        
        Args:
            prefix: 日志前缀 / Log prefix
        """
        memory_info = MemoryMonitor.get_memory_usage()
        logger.info(f"{prefix}内存使用情况 / {prefix}Memory usage: "
                   f"进程 {memory_info['process_memory_mb']:.1f}MB "
                   f"({memory_info['process_memory_percent']:.1f}%), "
                   f"系统 {memory_info['system_memory_percent']:.1f}%")


def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现
    Set random seed for reproducibility
    
    Args:
        seed: 随机种子值 / Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"随机种子设置为: {seed} / Random seed set to: {seed}")


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，如不存在则创建
    Ensure directory exists, create if not exists
    
    Args:
        dir_path: 目录路径 / Directory path
        
    Returns:
        目录路径对象 / Directory path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    保存数据为JSON文件
    Save data as JSON file
    
    Args:
        data: 要保存的数据 / Data to save
        file_path: 文件路径 / File path
        indent: JSON缩进 / JSON indentation
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"JSON文件已保存: {file_path} / JSON file saved: {file_path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    加载JSON文件
    Load JSON file
    
    Args:
        file_path: 文件路径 / File path
        
    Returns:
        加载的数据 / Loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path} / File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"JSON文件已加载: {file_path} / JSON file loaded: {file_path}")
    return data


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    保存数据为Pickle文件
    Save data as Pickle file
    
    Args:
        data: 要保存的数据 / Data to save
        file_path: 文件路径 / File path
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Pickle文件已保存: {file_path} / Pickle file saved: {file_path}")


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    加载Pickle文件
    Load Pickle file
    
    Args:
        file_path: 文件路径 / File path
        
    Returns:
        加载的数据 / Loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path} / File not found: {file_path}")
        
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Pickle文件已加载: {file_path} / Pickle file loaded: {file_path}")
    return data


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载YAML配置文件
    Load YAML configuration file
    
    Args:
        file_path: 文件路径 / File path
        
    Returns:
        配置字典 / Configuration dictionary
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path} / Config file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"YAML配置文件已加载: {file_path} / YAML config file loaded: {file_path}")
    return config


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    计算文件哈希值
    Calculate file hash
    
    Args:
        file_path: 文件路径 / File path
        algorithm: 哈希算法 / Hash algorithm
        
    Returns:
        文件哈希值 / File hash value
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path} / File not found: {file_path}")
        
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
            
    return hash_obj.hexdigest()


def normalize_scores(scores: List[float], method: str = 'min_max') -> List[float]:
    """
    标准化分数
    Normalize scores
    
    Args:
        scores: 分数列表 / Score list
        method: 标准化方法 / Normalization method
        
    Returns:
        标准化后的分数 / Normalized scores
    """
    scores = np.array(scores)
    
    if method == 'min_max':
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return [1.0] * len(scores)
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    elif method == 'z_score':
        mean_score = scores.mean()
        std_score = scores.std()
        if std_score == 0:
            return [0.0] * len(scores)
        return ((scores - mean_score) / std_score).tolist()
    
    elif method == 'softmax':
        exp_scores = np.exp(scores - scores.max())
        return (exp_scores / exp_scores.sum()).tolist()
    
    else:
        raise ValueError(f"不支持的标准化方法: {method} / Unsupported normalization method: {method}")


def get_device() -> torch.device:
    """
    获取可用的计算设备
    Get available computing device
    
    Returns:
        PyTorch设备对象 / PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU设备: {torch.cuda.get_device_name()} / Using GPU device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU设备 / Using CPU device")
    
    return device


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    Chunk list into smaller pieces
    
    Args:
        lst: 输入列表 / Input list
        chunk_size: 块大小 / Chunk size
        
    Returns:
        分块后的列表 / Chunked list
    """
    if chunk_size <= 0:
        raise ValueError("块大小必须大于0 / Chunk size must be greater than 0")
        
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


@contextmanager
def suppress_stdout():
    """
    上下文管理器：抑制标准输出
    Context manager to suppress stdout
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    验证配置文件完整性
    Validate configuration completeness
    
    Args:
        config: 配置字典 / Configuration dictionary
        required_keys: 必需的键列表 / Required keys list
        
    Raises:
        ValueError: 如果缺少必需的配置项 / If required configuration is missing
    """
    missing_keys = []
    for key in required_keys:
        if '.' in key:
            # 支持嵌套键检查 / Support nested key checking
            keys = key.split('.')
            current = config
            for k in keys:
                if k not in current:
                    missing_keys.append(key)
                    break
                current = current[k]
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"缺少必需的配置项: {missing_keys} / Missing required configuration: {missing_keys}")


if __name__ == "__main__":
    # 测试工具函数 / Test utility functions
    
    # 测试计时器 / Test timer
    with Timer("测试任务 / Test task"):
        time.sleep(1)
    
    # 测试内存监控 / Test memory monitor
    MemoryMonitor.log_memory_usage("测试 / Test ")
    
    # 测试随机种子 / Test random seed
    set_random_seed(42)
    
    # 测试设备获取 / Test device detection
    device = get_device()
    print(f"设备: {device} / Device: {device}")