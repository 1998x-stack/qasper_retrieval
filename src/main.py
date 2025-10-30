"""
QASPER检索评估系统主程序
QASPER Retrieval Evaluation System Main Program

主程序入口，集成所有模块并提供命令行接口
Main program entry point, integrates all modules and provides command line interface
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径 / Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    get_logger, setup_logging, Timer, MemoryMonitor, 
    set_random_seed, load_yaml, validate_config
)
from src.data import QASPERDatasetLoader, QASPERPreprocessor
from src.retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever
from src.evaluation import RetrievalEvaluator

# 设置日志 / Setup logging
setup_logging()
logger = get_logger(__name__)


class QASPERRetrievalSystem:
    """
    QASPER检索评估系统主类
    QASPER Retrieval Evaluation System Main Class
    
    集成数据加载、预处理、检索和评估功能
    Integrates data loading, preprocessing, retrieval and evaluation functionality
    """
    
    def __init__(self, config_path: str) -> None:
        """
        初始化QASPER检索评估系统
        Initialize QASPER Retrieval Evaluation System
        
        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()
        
        # 设置随机种子 / Set random seed
        random_seed = self.config.get('experiment', {}).get('random_seed', 42)
        set_random_seed(random_seed)
        
        # 初始化组件 / Initialize components
        self.dataset_loader = QASPERDatasetLoader(self.config)
        self.preprocessor = QASPERPreprocessor(self.config)
        self.evaluator = RetrievalEvaluator(self.config)
        
        # 检索器 / Retrievers
        self.bm25_retriever = BM25Retriever(self.config)
        self.embedding_retriever = EmbeddingRetriever(self.config)
        self.hybrid_retriever = HybridRetriever(self.config)
        
        # 数据存储 / Data storage
        self.raw_dataset = None
        self.processed_dataset = None
        self.preprocessed_dataset = None
        
        logger.info("QASPER检索评估系统初始化完成 / QASPER Retrieval Evaluation System initialized")
        logger.info(f"配置文件: {self.config_path} / Configuration file: {self.config_path}")
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        加载和验证配置文件
        Load and validate configuration file
        
        Returns:
            配置字典 / Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path} / "
                                   f"Configuration file not found: {self.config_path}")
        
        config = load_yaml(self.config_path)
        
        # 验证必需的配置项 / Validate required configuration items
        required_keys = [
            'dataset.name',
            'dataset.cache_dir',
            'models.embedding_model',
            'evaluation.metrics'
        ]
        
        validate_config(config, required_keys)
        
        logger.info("配置文件加载和验证完成 / Configuration file loaded and validated")
        return config
    
    def download_and_preprocess_data(self, force_download: bool = False,
                                   force_preprocess: bool = False) -> None:
        """
        下载和预处理数据
        Download and preprocess data
        
        Args:
            force_download: 是否强制重新下载 / Whether to force re-download
            force_preprocess: 是否强制重新预处理 / Whether to force re-preprocess
        """
        logger.info("=== 开始数据下载和预处理 / Starting data download and preprocessing ===")
        
        with Timer("数据下载和预处理 / Data download and preprocessing"):
            # 下载数据集 / Download dataset
            logger.info("步骤1: 下载QASPER数据集 / Step 1: Download QASPER dataset")
            self.raw_dataset = self.dataset_loader.download_dataset(force_download)
            
            # 提取文本数据 / Extract text data
            logger.info("步骤2: 提取文本数据 / Step 2: Extract text data")
            
            processed_data_path = self.dataset_loader.cache_dir / "processed_dataset.json"
            if processed_data_path.exists() and not force_download:
                logger.info("发现已处理的数据，正在加载 / Found processed data, loading...")
                self.processed_dataset = self.dataset_loader.load_processed_data()
            else:
                self.processed_dataset = self.dataset_loader.extract_text_data()
                self.dataset_loader.save_processed_data()
            
            # 预处理数据 / Preprocess data
            logger.info("步骤3: 预处理文本数据 / Step 3: Preprocess text data")
            
            preprocessed_data_path = self.preprocessor.cache_dir / "preprocessed_dataset.json"
            if preprocessed_data_path.exists() and not force_preprocess:
                logger.info("发现已预处理的数据，正在加载 / Found preprocessed data, loading...")
                self.preprocessed_dataset = self.preprocessor.load_preprocessed_data()
            else:
                self.preprocessed_dataset = self.preprocessor.preprocess_dataset(self.processed_dataset)
                self.preprocessor.save_preprocessed_data(self.preprocessed_dataset)
            
            # 打印统计信息 / Print statistics
            self.dataset_loader.print_statistics()
            self.preprocessor.print_statistics(self.preprocessed_dataset)
        
        MemoryMonitor.log_memory_usage("数据预处理完成后 / After data preprocessing ")
        logger.info("数据下载和预处理完成 / Data download and preprocessing completed")
    
    def build_indices(self, methods: List[str] = None) -> None:
        """
        构建检索索引
        Build retrieval indices
        
        Args:
            methods: 要构建的方法列表 / List of methods to build
        """
        if self.preprocessed_dataset is None:
            raise ValueError("预处理数据未加载，请先运行download_and_preprocess_data / "
                           "Preprocessed data not loaded, please run download_and_preprocess_data first")
        
        if methods is None:
            methods = ['bm25', 'embedding', 'hybrid']
        
        logger.info(f"=== 开始构建检索索引，方法: {methods} / Starting to build retrieval indices, methods: {methods} ===")
        
        with Timer("检索索引构建 / Retrieval indices building"):
            for method in methods:
                logger.info(f"构建{method.upper()}索引 / Building {method.upper()} index")
                
                if method == 'bm25':
                    self._build_bm25_index()
                elif method == 'embedding':
                    self._build_embedding_index()
                elif method == 'hybrid':
                    self._build_hybrid_index()
                else:
                    logger.warning(f"未知的检索方法: {method} / Unknown retrieval method: {method}")
                
                MemoryMonitor.log_memory_usage(f"{method}索引构建后 / After {method} index building ")
        
        logger.info("检索索引构建完成 / Retrieval indices building completed")
    
    def _build_bm25_index(self) -> None:
        """构建BM25索引 / Build BM25 index"""
        try:
            # 尝试加载已存在的索引 / Try to load existing index
            self.bm25_retriever.load_index()
            logger.info("BM25索引已从缓存加载 / BM25 index loaded from cache")
        except FileNotFoundError:
            # 构建新索引 / Build new index
            self.bm25_retriever.build_index(self.preprocessed_dataset)
            self.bm25_retriever.save_index()
            logger.info("BM25索引构建并保存完成 / BM25 index built and saved")
    
    def _build_embedding_index(self) -> None:
        """构建Embedding索引 / Build Embedding index"""
        try:
            # 尝试加载已存在的索引 / Try to load existing index
            self.embedding_retriever.load_index()
            logger.info("Embedding索引已从缓存加载 / Embedding index loaded from cache")
        except FileNotFoundError:
            # 构建新索引 / Build new index
            self.embedding_retriever.build_index(self.preprocessed_dataset)
            self.embedding_retriever.save_index()
            logger.info("Embedding索引构建并保存完成 / Embedding index built and saved")
    
    def _build_hybrid_index(self) -> None:
        """构建混合索引 / Build Hybrid index"""
        try:
            # 尝试加载已存在的索引 / Try to load existing index
            self.hybrid_retriever.load_index()
            logger.info("混合索引已从缓存加载 / Hybrid index loaded from cache")
        except FileNotFoundError:
            # 构建新索引 / Build new index
            self.hybrid_retriever.build_index(self.preprocessed_dataset)
            self.hybrid_retriever.save_index()
            logger.info("混合索引构建并保存完成 / Hybrid index built and saved")
    
    def evaluate_methods(self, methods: List[str] = None,
                        test_split: str = 'validation',
                        max_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        评估检索方法
        Evaluate retrieval methods
        
        Args:
            methods: 要评估的方法列表 / List of methods to evaluate
            test_split: 测试数据分割 / Test data split
            max_queries: 最大查询数量 / Maximum number of queries
            
        Returns:
            评估结果 / Evaluation results
        """
        if self.preprocessed_dataset is None:
            raise ValueError("预处理数据未加载 / Preprocessed data not loaded")
        
        if methods is None:
            methods = ['bm25', 'embedding', 'hybrid']
        
        logger.info(f"=== 开始评估检索方法，方法: {methods} / Starting to evaluate retrieval methods, methods: {methods} ===")
        
        # 准备评估数据 / Prepare evaluation data
        eval_data = self._prepare_evaluation_data(test_split, max_queries)
        
        evaluation_results = []
        
        with Timer("检索方法评估 / Retrieval methods evaluation"):
            for method in methods:
                logger.info(f"评估{method.upper()}方法 / Evaluating {method.upper()} method")
                
                try:
                    if method == 'bm25':
                        result = self._evaluate_bm25(eval_data)
                    elif method == 'embedding':
                        result = self._evaluate_embedding(eval_data)
                    elif method == 'hybrid':
                        result = self._evaluate_hybrid(eval_data)
                    else:
                        logger.warning(f"未知的评估方法: {method} / Unknown evaluation method: {method}")
                        continue
                    
                    evaluation_results.append(result)
                    
                    # 打印评估摘要 / Print evaluation summary
                    self.evaluator.print_evaluation_summary(result)
                    
                    # 保存单独的评估结果 / Save individual evaluation results
                    self.evaluator.save_evaluation_results(result)
                    
                except Exception as e:
                    logger.error(f"{method}方法评估失败: {e} / {method} method evaluation failed: {e}")
                    continue
        
        # 比较所有方法 / Compare all methods
        if len(evaluation_results) > 1:
            logger.info("比较所有评估方法 / Comparing all evaluation methods")
            comparison_results = self.evaluator.compare_methods(evaluation_results)
            self._print_comparison_summary(comparison_results)
        
        logger.info("检索方法评估完成 / Retrieval methods evaluation completed")
        return {
            'individual_results': evaluation_results,
            'comparison_results': comparison_results if len(evaluation_results) > 1 else None
        }
    
    def _prepare_evaluation_data(self, test_split: str, 
                               max_queries: Optional[int]) -> Dict[str, Any]:
        """
        准备评估数据
        Prepare evaluation data
        
        Args:
            test_split: 测试数据分割 / Test data split
            max_queries: 最大查询数量 / Maximum number of queries
            
        Returns:
            评估数据 / Evaluation data
        """
        logger.info(f"准备评估数据，分割: {test_split} / Preparing evaluation data, split: {test_split}")
        
        # 获取测试数据 / Get test data
        if test_split not in self.processed_dataset:
            raise ValueError(f"测试分割'{test_split}'不存在 / Test split '{test_split}' does not exist")
        
        test_documents = self.processed_dataset[test_split]
        
        # 提取查询和真实答案 / Extract queries and ground truth
        queries = []
        ground_truth = []
        passage_texts = {}
        
        query_count = 0
        for doc in test_documents:
            # 构建段落文本字典 / Build passage texts dictionary
            for passage in doc['passages']:
                passage_id = len(passage_texts)
                passage_texts[passage_id] = passage['text']
            
            # 提取问答对 / Extract QA pairs
            for qa_pair in doc['qa_pairs']:
                if max_queries and query_count >= max_queries:
                    break
                
                question = qa_pair['question']
                queries.append(question)
                
                # 简化处理：使用所有段落作为潜在相关段落 / Simplified: use all passages as potentially relevant
                # 在实际应用中，这里应该有更精确的相关性标注 / In practice, more precise relevance annotations should be used
                relevant_passage_ids = list(range(len(doc['passages'])))
                ground_truth.append(relevant_passage_ids)
                
                query_count += 1
            
            if max_queries and query_count >= max_queries:
                break
        
        logger.info(f"准备了{len(queries)}个查询用于评估 / Prepared {len(queries)} queries for evaluation")
        
        return {
            'queries': queries,
            'ground_truth': ground_truth,
            'passage_texts': passage_texts
        }
    
    def _evaluate_bm25(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估BM25方法 / Evaluate BM25 method"""
        queries = eval_data['queries']
        ground_truth = eval_data['ground_truth']
        passage_texts = eval_data['passage_texts']
        
        # 执行批量检索 / Perform batch retrieval
        top_k = max(self.evaluator.top_k_values)
        retrieval_results = self.bm25_retriever.batch_search(queries, top_k=top_k)
        
        # 评估结果 / Evaluate results
        return self.evaluator.evaluate_retrieval_results(
            queries, retrieval_results, ground_truth, passage_texts, "BM25"
        )
    
    def _evaluate_embedding(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估Embedding方法 / Evaluate Embedding method"""
        queries = eval_data['queries']
        ground_truth = eval_data['ground_truth']
        passage_texts = eval_data['passage_texts']
        
        # 执行批量检索 / Perform batch retrieval
        top_k = max(self.evaluator.top_k_values)
        retrieval_results = self.embedding_retriever.batch_search(queries, top_k=top_k)
        
        # 评估结果 / Evaluate results
        return self.evaluator.evaluate_retrieval_results(
            queries, retrieval_results, ground_truth, passage_texts, "Embedding"
        )
    
    def _evaluate_hybrid(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估混合方法 / Evaluate Hybrid method"""
        queries = eval_data['queries']
        ground_truth = eval_data['ground_truth']
        passage_texts = eval_data['passage_texts']
        
        # 执行批量检索 / Perform batch retrieval
        top_k = max(self.evaluator.top_k_values)
        retrieval_results = self.hybrid_retriever.batch_search(queries, top_k=top_k)
        
        # 评估结果 / Evaluate results
        return self.evaluator.evaluate_retrieval_results(
            queries, retrieval_results, ground_truth, passage_texts, "Hybrid"
        )
    
    def _print_comparison_summary(self, comparison_results: Dict[str, Any]) -> None:
        """
        打印比较结果摘要
        Print comparison results summary
        
        Args:
            comparison_results: 比较结果 / Comparison results
        """
        logger.info("=== 方法比较结果摘要 / Methods Comparison Summary ===")
        
        methods = comparison_results['methods']
        logger.info(f"比较的方法: {', '.join(methods)} / Compared methods: {', '.join(methods)}")
        
        # 打印最佳方法 / Print best methods
        best_methods = comparison_results['best_methods']
        logger.info("\n各指标最佳方法 / Best methods for each metric:")
        for metric, info in best_methods.items():
            logger.info(f"  {metric}: {info['method']} ({info['value']:.4f})")
        
        # 打印平均排名 / Print average rankings
        avg_rankings = comparison_results['average_rankings']
        logger.info("\n平均排名 / Average rankings:")
        sorted_rankings = sorted(avg_rankings.items(), key=lambda x: x[1])
        for i, (method, ranking) in enumerate(sorted_rankings, 1):
            logger.info(f"  {i}. {method}: {ranking:.2f}")
    
    def run_full_pipeline(self, methods: List[str] = None,
                         force_download: bool = False,
                         force_preprocess: bool = False,
                         force_rebuild_index: bool = False,
                         test_split: str = 'validation',
                         max_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        运行完整的处理流水线
        Run complete processing pipeline
        
        Args:
            methods: 要评估的方法列表 / List of methods to evaluate
            force_download: 是否强制重新下载 / Whether to force re-download
            force_preprocess: 是否强制重新预处理 / Whether to force re-preprocess
            force_rebuild_index: 是否强制重建索引 / Whether to force rebuild index
            test_split: 测试数据分割 / Test data split
            max_queries: 最大查询数量 / Maximum number of queries
            
        Returns:
            完整的评估结果 / Complete evaluation results
        """
        logger.info("=== 开始运行完整的QASPER检索评估流水线 / Starting complete QASPER retrieval evaluation pipeline ===")
        
        start_time = time.time()
        
        with Timer("完整流水线 / Complete pipeline"):
            # 步骤1: 数据下载和预处理 / Step 1: Data download and preprocessing
            self.download_and_preprocess_data(force_download, force_preprocess)
            
            # 步骤2: 构建索引 / Step 2: Build indices
            if force_rebuild_index:
                # 删除现有索引文件 / Delete existing index files
                cache_dir = Path(self.config['dataset']['cache_dir'])
                for index_path in cache_dir.glob("*index*"):
                    if index_path.is_file():
                        index_path.unlink()
                    elif index_path.is_dir():
                        import shutil
                        shutil.rmtree(index_path)
                logger.info("已删除现有索引文件 / Existing index files deleted")
            
            self.build_indices(methods)
            
            # 步骤3: 评估方法 / Step 3: Evaluate methods
            evaluation_results = self.evaluate_methods(methods, test_split, max_queries)
        
        total_time = time.time() - start_time
        logger.info(f"完整流水线运行完成，总耗时: {total_time:.2f}秒 / "
                   f"Complete pipeline finished, total time: {total_time:.2f}s")
        
        # 最终内存使用报告 / Final memory usage report
        MemoryMonitor.log_memory_usage("流水线完成后 / After pipeline completion ")
        
        return evaluation_results


def create_argument_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    Create command line argument parser
    
    Returns:
        参数解析器 / Argument parser
    """
    parser = argparse.ArgumentParser(
        description="QASPER检索评估系统 / QASPER Retrieval Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法 / Example usage:
  python main.py --method all                    # 评估所有方法 / Evaluate all methods
  python main.py --method bm25 embedding         # 评估BM25和Embedding / Evaluate BM25 and Embedding
  python main.py --force-download                # 强制重新下载数据 / Force re-download data
  python main.py --max-queries 100               # 限制查询数量 / Limit query count
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='配置文件路径 / Configuration file path'
    )
    
    parser.add_argument(
        '--method', '-m',
        nargs='+',
        choices=['bm25', 'embedding', 'hybrid', 'all'],
        default=['all'],
        help='要评估的检索方法 / Retrieval methods to evaluate'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='强制重新下载数据集 / Force re-download dataset'
    )
    
    parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help='强制重新预处理数据 / Force re-preprocess data'
    )
    
    parser.add_argument(
        '--force-rebuild-index',
        action='store_true',
        help='强制重建所有索引 / Force rebuild all indices'
    )
    
    parser.add_argument(
        '--test-split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='用于评估的数据分割 / Data split for evaluation'
    )
    
    parser.add_argument(
        '--max-queries',
        type=int,
        default=None,
        help='最大查询数量（用于快速测试） / Maximum number of queries (for quick testing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式 / Verbose output mode'
    )
    
    return parser


def main():
    """主函数 / Main function"""
    # 解析命令行参数 / Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # 处理方法参数 / Process method argument
        if 'all' in args.method:
            methods = ['bm25', 'embedding', 'hybrid']
        else:
            methods = args.method
        
        logger.info(f"启动QASPER检索评估系统 / Starting QASPER Retrieval Evaluation System")
        logger.info(f"配置文件: {args.config} / Configuration file: {args.config}")
        logger.info(f"评估方法: {methods} / Evaluation methods: {methods}")
        logger.info(f"测试分割: {args.test_split} / Test split: {args.test_split}")
        if args.max_queries:
            logger.info(f"最大查询数: {args.max_queries} / Maximum queries: {args.max_queries}")
        
        # 创建系统实例 / Create system instance
        system = QASPERRetrievalSystem(args.config)
        
        # 运行完整流水线 / Run complete pipeline
        results = system.run_full_pipeline(
            methods=methods,
            force_download=args.force_download,
            force_preprocess=args.force_preprocess,
            force_rebuild_index=args.force_rebuild_index,
            test_split=args.test_split,
            max_queries=args.max_queries
        )
        
        logger.info("QASPER检索评估系统运行完成 / QASPER Retrieval Evaluation System completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断执行 / User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"系统运行失败: {e} / System execution failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()} / Detailed error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)