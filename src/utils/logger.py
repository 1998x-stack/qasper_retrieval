"""
日志工具模块
Logging utilities module

提供统一的日志管理功能，支持文件和控制台输出
Provides unified logging functionality with file and console output support
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import yaml


class LoggerManager:
    """
    日志管理器类
    Logger manager class
    
    负责配置和管理整个应用的日志系统
    Responsible for configuring and managing the application's logging system
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        初始化日志管理器
        Initialize logger manager
        
        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        self.config_path = config_path
        self.is_configured = False
        
    def setup_logger(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        设置日志配置
        Setup logger configuration
        
        Args:
            config: 日志配置字典 / Logging configuration dictionary
        """
        if self.is_configured:
            return
            
        # 移除默认handler / Remove default handler
        logger.remove()
        
        # 加载配置 / Load configuration
        if config is None:
            config = self._load_config()
            
        log_config = config.get('logging', {})
        
        # 设置日志级别 / Set log level
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', 
                                   "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")
        
        # 添加控制台输出 / Add console output
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 添加文件输出 / Add file output
        log_file = log_config.get('file_path', './logs/qasper_retrieval.log')
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format=log_format,
            rotation=log_config.get('rotation', '100 MB'),
            retention=log_config.get('retention', '30 days'),
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        self.is_configured = True
        logger.info("日志系统初始化完成 / Logger system initialized successfully")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        Load configuration file
        
        Returns:
            配置字典 / Configuration dictionary
        """
        if self.config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e} / Failed to load config file: {e}")
            return {}
            
    def get_logger(self, name: str = __name__) -> logger:
        """
        获取日志记录器
        Get logger instance
        
        Args:
            name: 日志记录器名称 / Logger name
            
        Returns:
            日志记录器实例 / Logger instance
        """
        if not self.is_configured:
            self.setup_logger()
        return logger.bind(name=name)


# 全局日志管理器实例 / Global logger manager instance
_logger_manager = LoggerManager()


def get_logger(name: str = __name__) -> logger:
    """
    获取日志记录器的便捷函数
    Convenience function to get logger instance
    
    Args:
        name: 日志记录器名称 / Logger name
        
    Returns:
        日志记录器实例 / Logger instance
    """
    return _logger_manager.get_logger(name)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    设置日志系统的便捷函数
    Convenience function to setup logging system
    
    Args:
        config: 日志配置字典 / Logging configuration dictionary
    """
    _logger_manager.setup_logger(config)


if __name__ == "__main__":
    # 测试日志功能 / Test logging functionality
    setup_logging()
    test_logger = get_logger("test")
    test_logger.info("这是一条测试日志 / This is a test log message")
    test_logger.warning("这是一条警告日志 / This is a warning log message")
    test_logger.error("这是一条错误日志 / This is an error log message")