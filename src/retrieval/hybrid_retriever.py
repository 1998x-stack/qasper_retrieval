"""
混合检索器模块
Hybrid retriever module

结合BM25和embedding检索方法，实现混合检索
Combines BM25 and embedding retrieval methods for hybrid retrieval
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from collections import defaultdict

from ..utils import get_logger, Timer, normalize_scores, save_json, load_json
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever

logger = get_logger(__name__)


class HybridRetriever:
    """
    混合检索器类
    Hybrid retriever class
    
    结合BM25和embedding检索方法，提供混合检索功能
    Combines BM25 and embedding retrieval methods for hybrid retrieval
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化混合检索器
        Initialize hybrid retriever
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.hybrid_config = config.get('hybrid', {})
        self.cache_dir = Path(config['dataset']['cache_dir'])
        
        # 权重配置 / Weight configuration
        self.bm25_weight = self.hybrid_config.get('bm25_weight', 0.5)
        self.embedding_weight = self.hybrid_config.get('embedding_weight', 0.5)
        self.normalization_method = self.hybrid_config.get('normalization', 'min_max')
        
        # 验证权重 / Validate weights
        total_weight = self.bm25_weight + self.embedding_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"权重之和不等于1.0: {total_weight}，将进行归一化 / "
                          f"Weight sum not equal to 1.0: {total_weight}, will normalize")
            self.bm25_weight /= total_weight
            self.embedding_weight /= total_weight
        
        # 初始化检索器 / Initialize retrievers
        self.bm25_retriever = BM25Retriever(config)
        self.embedding_retriever = EmbeddingRetriever(config)
        
        # 索引状态 / Index status
        self.index_built = False
        
        logger.info("混合检索器初始化完成 / Hybrid retriever initialized")
        logger.info(f"权重配置 - BM25: {self.bm25_weight:.3f}, Embedding: {self.embedding_weight:.3f}")
        logger.info(f"标准化方法: {self.normalization_method} / Normalization method: {self.normalization_method}")
    
    def build_index(self, preprocessed_dataset: Dict[str, Any]) -> None:
        """
        构建混合索引
        Build hybrid index
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
        """
        logger.info("开始构建混合检索索引 / Starting to build hybrid retrieval index")
        
        with Timer("混合索引构建 / Hybrid index building"):
            # 构建BM25索引 / Build BM25 index
            logger.info("构建BM25索引 / Building BM25 index")
            self.bm25_retriever.build_index(preprocessed_dataset)
            
            # 构建embedding索引 / Build embedding index
            logger.info("构建Embedding索引 / Building Embedding index")
            self.embedding_retriever.build_index(preprocessed_dataset)
            
            self.index_built = True
        
        logger.info("混合检索索引构建完成 / Hybrid retrieval index built successfully")
        self._log_index_statistics()
    
    def _log_index_statistics(self) -> None:
        """
        记录索引统计信息
        Log index statistics
        """
        logger.info("=== 混合检索索引统计信息 / Hybrid Retrieval Index Statistics ===")
        logger.info(f"BM25索引状态: {'已构建' if self.bm25_retriever.bm25_model else '未构建'} / "
                   f"BM25 index status: {'Built' if self.bm25_retriever.bm25_model else 'Not built'}")
        logger.info(f"Embedding索引状态: {'已构建' if self.embedding_retriever.index_built else '未构建'} / "
                   f"Embedding index status: {'Built' if self.embedding_retriever.index_built else 'Not built'}")
        logger.info(f"权重配置 - BM25: {self.bm25_weight:.3f}, Embedding: {self.embedding_weight:.3f}")
    
    def search(self, query: str, top_k: int = 10,
              bm25_top_k: Optional[int] = None,
              embedding_top_k: Optional[int] = None,
              min_score: float = 0.0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        执行混合检索
        Perform hybrid retrieval
        
        Args:
            query: 查询文本 / Query text
            top_k: 最终返回的前k个结果 / Final top k results
            bm25_top_k: BM25检索的前k个结果 / Top k results for BM25
            embedding_top_k: Embedding检索的前k个结果 / Top k results for embedding
            min_score: 最小分数阈值 / Minimum score threshold
            
        Returns:
            混合检索结果列表 / Hybrid retrieval results list
            格式: [(passage_id, score, metadata), ...]
        """
        if not self.index_built:
            raise ValueError("混合索引未构建，请先调用build_index / "
                           "Hybrid index not built, please call build_index first")
        
        # 设置默认的检索数量 / Set default retrieval counts
        if bm25_top_k is None:
            bm25_top_k = min(top_k * 2, 100)  # 获取更多候选结果 / Get more candidate results
        if embedding_top_k is None:
            embedding_top_k = min(top_k * 2, 100)
        
        # 执行BM25检索 / Perform BM25 retrieval
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k, min_score=0.0)
        
        # 执行embedding检索 / Perform embedding retrieval
        embedding_results = self.embedding_retriever.search(query, top_k=embedding_top_k, min_score=0.0)
        
        # 融合结果 / Fuse results
        fused_results = self._fuse_results(bm25_results, embedding_results, query)
        
        # 应用最小分数阈值并限制结果数量 / Apply minimum score threshold and limit results
        final_results = []
        for passage_id, score, metadata in fused_results[:top_k]:
            if score >= min_score:
                final_results.append((passage_id, score, metadata))
            else:
                break
        
        logger.debug(f"混合检索完成，查询: '{query}'，返回{len(final_results)}个结果 / "
                    f"Hybrid retrieval completed, query: '{query}', returned {len(final_results)} results")
        
        return final_results
    
    def _fuse_results(self, bm25_results: List[Tuple[int, float, Dict[str, Any]]],
                     embedding_results: List[Tuple[int, float, Dict[str, Any]]],
                     query: str) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        融合BM25和embedding检索结果
        Fuse BM25 and embedding retrieval results
        
        Args:
            bm25_results: BM25检索结果 / BM25 retrieval results
            embedding_results: Embedding检索结果 / Embedding retrieval results
            query: 查询文本 / Query text
            
        Returns:
            融合后的结果 / Fused results
        """
        # 收集所有段落ID和分数 / Collect all passage IDs and scores
        passage_scores = defaultdict(lambda: {'bm25': 0.0, 'embedding': 0.0, 'metadata': None})
        
        # 处理BM25结果 / Process BM25 results
        bm25_scores = [score for _, score, _ in bm25_results]
        if bm25_scores:
            normalized_bm25_scores = normalize_scores(bm25_scores, self.normalization_method)
            for i, (passage_id, score, metadata) in enumerate(bm25_results):
                passage_scores[passage_id]['bm25'] = normalized_bm25_scores[i]
                passage_scores[passage_id]['metadata'] = metadata
        
        # 处理embedding结果 / Process embedding results
        embedding_scores = [score for _, score, _ in embedding_results]
        if embedding_scores:
            normalized_embedding_scores = normalize_scores(embedding_scores, self.normalization_method)
            for i, (passage_id, score, metadata) in enumerate(embedding_results):
                passage_scores[passage_id]['embedding'] = normalized_embedding_scores[i]
                if passage_scores[passage_id]['metadata'] is None:
                    passage_scores[passage_id]['metadata'] = metadata
        
        # 计算混合分数 / Calculate hybrid scores
        fused_results = []
        for passage_id, scores in passage_scores.items():
            hybrid_score = (self.bm25_weight * scores['bm25'] + 
                           self.embedding_weight * scores['embedding'])
            
            # 更新元数据 / Update metadata
            metadata = scores['metadata'].copy() if scores['metadata'] else {}
            metadata.update({
                'bm25_score': scores['bm25'],
                'embedding_score': scores['embedding'],
                'hybrid_score': hybrid_score,
                'bm25_weight': self.bm25_weight,
                'embedding_weight': self.embedding_weight
            })
            
            fused_results.append((passage_id, hybrid_score, metadata))
        
        # 按混合分数排序 / Sort by hybrid score
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新排名 / Update ranking
        for i, (passage_id, score, metadata) in enumerate(fused_results):
            metadata['rank'] = i + 1
        
        return fused_results
    
    def batch_search(self, queries: List[str], top_k: int = 10,
                    bm25_top_k: Optional[int] = None,
                    embedding_top_k: Optional[int] = None,
                    min_score: float = 0.0) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        """
        批量混合检索
        Batch hybrid retrieval
        
        Args:
            queries: 查询列表 / Query list
            top_k: 最终返回的前k个结果 / Final top k results
            bm25_top_k: BM25检索的前k个结果 / Top k results for BM25
            embedding_top_k: Embedding检索的前k个结果 / Top k results for embedding
            min_score: 最小分数阈值 / Minimum score threshold
            
        Returns:
            批量混合检索结果 / Batch hybrid retrieval results
        """
        logger.info(f"开始批量混合检索，查询数量: {len(queries)} / "
                   f"Starting batch hybrid retrieval, number of queries: {len(queries)}")
        
        with Timer("批量混合检索 / Batch hybrid retrieval"):
            # 设置默认的检索数量 / Set default retrieval counts
            if bm25_top_k is None:
                bm25_top_k = min(top_k * 2, 100)
            if embedding_top_k is None:
                embedding_top_k = min(top_k * 2, 100)
            
            # 批量BM25检索 / Batch BM25 retrieval
            logger.info("执行批量BM25检索 / Performing batch BM25 retrieval")
            bm25_batch_results = self.bm25_retriever.batch_search(
                queries, top_k=bm25_top_k, min_score=0.0
            )
            
            # 批量embedding检索 / Batch embedding retrieval
            logger.info("执行批量Embedding检索 / Performing batch embedding retrieval")
            embedding_batch_results = self.embedding_retriever.batch_search(
                queries, top_k=embedding_top_k, min_score=0.0
            )
            
            # 融合所有结果 / Fuse all results
            logger.info("融合检索结果 / Fusing retrieval results")
            fused_batch_results = []
            for i, (query, bm25_results, embedding_results) in enumerate(
                zip(queries, bm25_batch_results, embedding_batch_results)
            ):
                fused_results = self._fuse_results(bm25_results, embedding_results, query)
                
                # 应用最小分数阈值并限制结果数量 / Apply minimum score threshold and limit results
                final_results = []
                for passage_id, score, metadata in fused_results[:top_k]:
                    if score >= min_score:
                        final_results.append((passage_id, score, metadata))
                    else:
                        break
                
                fused_batch_results.append(final_results)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已融合 {i + 1}/{len(queries)} 个查询结果 / "
                               f"Fused {i + 1}/{len(queries)} query results")
        
        logger.info("批量混合检索完成 / Batch hybrid retrieval completed")
        return fused_batch_results
    
    def analyze_retrieval_overlap(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        分析BM25和embedding检索结果的重叠情况
        Analyze overlap between BM25 and embedding retrieval results
        
        Args:
            query: 查询文本 / Query text
            top_k: 分析的前k个结果 / Top k results to analyze
            
        Returns:
            重叠分析结果 / Overlap analysis results
        """
        if not self.index_built:
            raise ValueError("混合索引未构建 / Hybrid index not built")
        
        # 获取检索结果 / Get retrieval results
        bm25_results = self.bm25_retriever.search(query, top_k=top_k, min_score=0.0)
        embedding_results = self.embedding_retriever.search(query, top_k=top_k, min_score=0.0)
        
        # 提取段落ID / Extract passage IDs
        bm25_passage_ids = set(passage_id for passage_id, _, _ in bm25_results)
        embedding_passage_ids = set(passage_id for passage_id, _, _ in embedding_results)
        
        # 计算重叠 / Calculate overlap
        overlap_ids = bm25_passage_ids & embedding_passage_ids
        bm25_only_ids = bm25_passage_ids - embedding_passage_ids
        embedding_only_ids = embedding_passage_ids - bm25_passage_ids
        
        # 计算统计信息 / Calculate statistics
        analysis = {
            'query': query,
            'top_k': top_k,
            'bm25_count': len(bm25_passage_ids),
            'embedding_count': len(embedding_passage_ids),
            'overlap_count': len(overlap_ids),
            'bm25_only_count': len(bm25_only_ids),
            'embedding_only_count': len(embedding_only_ids),
            'overlap_ratio': len(overlap_ids) / max(len(bm25_passage_ids | embedding_passage_ids), 1),
            'jaccard_similarity': len(overlap_ids) / len(bm25_passage_ids | embedding_passage_ids) if (bm25_passage_ids | embedding_passage_ids) else 0,
            'overlap_passage_ids': list(overlap_ids),
            'bm25_only_passage_ids': list(bm25_only_ids),
            'embedding_only_passage_ids': list(embedding_only_ids)
        }
        
        return analysis
    
    def optimize_weights(self, evaluation_queries: List[str],
                        ground_truth: List[List[int]],
                        weight_candidates: List[Tuple[float, float]] = None) -> Tuple[float, float, float]:
        """
        优化混合检索权重
        Optimize hybrid retrieval weights
        
        Args:
            evaluation_queries: 评估查询列表 / Evaluation query list
            ground_truth: 真实相关段落ID列表 / Ground truth relevant passage ID list
            weight_candidates: 权重候选列表 / Weight candidate list
            
        Returns:
            最优权重和评估分数 / Optimal weights and evaluation score
            格式: (best_bm25_weight, best_embedding_weight, best_score)
        """
        if weight_candidates is None:
            weight_candidates = [
                (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
                (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)
            ]
        
        logger.info(f"开始权重优化，候选权重数量: {len(weight_candidates)} / "
                   f"Starting weight optimization, candidate count: {len(weight_candidates)}")
        
        best_score = -1.0
        best_weights = (0.5, 0.5)
        
        original_bm25_weight = self.bm25_weight
        original_embedding_weight = self.embedding_weight
        
        try:
            for bm25_weight, embedding_weight in weight_candidates:
                # 设置临时权重 / Set temporary weights
                self.bm25_weight = bm25_weight
                self.embedding_weight = embedding_weight
                
                # 执行检索 / Perform retrieval
                results = self.batch_search(evaluation_queries, top_k=10)
                
                # 计算评估分数（这里使用简单的精确率@10） / Calculate evaluation score (using precision@10)
                total_precision = 0.0
                for query_results, gt_passage_ids in zip(results, ground_truth):
                    retrieved_ids = set(passage_id for passage_id, _, _ in query_results[:10])
                    relevant_ids = set(gt_passage_ids)
                    
                    if retrieved_ids:
                        precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
                        total_precision += precision
                
                avg_precision = total_precision / len(evaluation_queries) if evaluation_queries else 0.0
                
                logger.info(f"权重 ({bm25_weight:.1f}, {embedding_weight:.1f}), 精确率@10: {avg_precision:.4f} / "
                           f"Weights ({bm25_weight:.1f}, {embedding_weight:.1f}), Precision@10: {avg_precision:.4f}")
                
                if avg_precision > best_score:
                    best_score = avg_precision
                    best_weights = (bm25_weight, embedding_weight)
        
        finally:
            # 恢复原始权重 / Restore original weights
            self.bm25_weight = original_bm25_weight
            self.embedding_weight = original_embedding_weight
        
        logger.info(f"权重优化完成，最优权重: ({best_weights[0]:.3f}, {best_weights[1]:.3f}), "
                   f"最优分数: {best_score:.4f} / "
                   f"Weight optimization completed, optimal weights: ({best_weights[0]:.3f}, {best_weights[1]:.3f}), "
                   f"optimal score: {best_score:.4f}")
        
        return best_weights[0], best_weights[1], best_score
    
    def save_index(self, index_path: Optional[str] = None) -> str:
        """
        保存混合索引
        Save hybrid index
        
        Args:
            index_path: 索引保存路径 / Index save path
            
        Returns:
            保存路径 / Save path
        """
        if not self.index_built:
            raise ValueError("没有混合索引可保存 / No hybrid index to save")
        
        if index_path is None:
            index_dir = self.cache_dir / "hybrid_index"
        else:
            index_dir = Path(index_path)
        
        logger.info(f"保存混合索引到: {index_dir} / Saving hybrid index to: {index_dir}")
        
        with Timer("混合索引保存 / Hybrid index saving"):
            # 保存BM25索引 / Save BM25 index
            bm25_path = self.bm25_retriever.save_index(index_dir / "bm25_index.pkl")
            
            # 保存embedding索引 / Save embedding index
            embedding_path = self.embedding_retriever.save_index(index_dir / "embedding_index")
            
            # 保存混合检索器配置 / Save hybrid retriever configuration
            hybrid_config = {
                'bm25_weight': self.bm25_weight,
                'embedding_weight': self.embedding_weight,
                'normalization_method': self.normalization_method,
                'index_built': self.index_built
            }
            
            config_path = index_dir / "hybrid_config.json"
            save_json(hybrid_config, config_path)
        
        logger.info("混合索引保存完成 / Hybrid index saved successfully")
        return str(index_dir)
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        加载混合索引
        Load hybrid index
        
        Args:
            index_path: 索引文件路径 / Index file path
        """
        if index_path is None:
            index_dir = self.cache_dir / "hybrid_index"
        else:
            index_dir = Path(index_path)
        
        if not index_dir.exists():
            raise FileNotFoundError(f"混合索引目录不存在: {index_dir} / "
                                   f"Hybrid index directory not found: {index_dir}")
        
        logger.info(f"加载混合索引从: {index_dir} / Loading hybrid index from: {index_dir}")
        
        with Timer("混合索引加载 / Hybrid index loading"):
            # 加载BM25索引 / Load BM25 index
            bm25_path = index_dir / "bm25_index.pkl"
            self.bm25_retriever.load_index(bm25_path)
            
            # 加载embedding索引 / Load embedding index
            embedding_path = index_dir / "embedding_index"
            self.embedding_retriever.load_index(embedding_path)
            
            # 加载混合检索器配置 / Load hybrid retriever configuration
            config_path = index_dir / "hybrid_config.json"
            if config_path.exists():
                hybrid_config = load_json(config_path)
                self.bm25_weight = hybrid_config.get('bm25_weight', self.bm25_weight)
                self.embedding_weight = hybrid_config.get('embedding_weight', self.embedding_weight)
                self.normalization_method = hybrid_config.get('normalization_method', self.normalization_method)
                self.index_built = hybrid_config.get('index_built', True)
            else:
                self.index_built = True
        
        logger.info("混合索引加载完成 / Hybrid index loaded successfully")
        self._log_index_statistics()
    
    def get_passage_by_id(self, passage_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID获取段落信息
        Get passage information by ID
        
        Args:
            passage_id: 段落ID / Passage ID
            
        Returns:
            段落信息 / Passage information
        """
        # 优先从BM25检索器获取 / Prefer getting from BM25 retriever
        passage_info = self.bm25_retriever.get_passage_by_id(passage_id)
        if passage_info is None:
            passage_info = self.embedding_retriever.get_passage_by_id(passage_id)
        return passage_info


if __name__ == "__main__":
    # 测试混合检索器 / Test hybrid retriever
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建检索器 / Create retriever
    hybrid_retriever = HybridRetriever(config)
    
    # 示例数据 / Example data
    example_preprocessed_data = {
        'corpus': {
            'passages': [
                'machine learning deep neural networks',
                'natural language processing transformers',
                'computer vision convolutional networks'
            ],
            'passage_ids': [0, 1, 2],
            'document_ids': ['doc1', 'doc1', 'doc2']
        },
        'passages': {
            0: {
                'paper_id': 'doc1',
                'section_name': 'Introduction',
                'section_idx': 0,
                'paragraph_idx': 0,
                'original_text': 'Machine learning with deep neural networks...',
                'cleaned_text': 'machine learning deep neural networks',
                'tokens': ['machine', 'learning', 'deep', 'neural', 'networks'],
                'length': 41
            },
            1: {
                'paper_id': 'doc1',
                'section_name': 'Methods',
                'section_idx': 1,
                'paragraph_idx': 0,
                'original_text': 'Natural language processing with transformers...',
                'cleaned_text': 'natural language processing transformers',
                'tokens': ['natural', 'language', 'processing', 'transformers'],
                'length': 39
            },
            2: {
                'paper_id': 'doc2',
                'section_name': 'Introduction',
                'section_idx': 0,
                'paragraph_idx': 0,
                'original_text': 'Computer vision using convolutional networks...',
                'cleaned_text': 'computer vision convolutional networks',
                'tokens': ['computer', 'vision', 'convolutional', 'networks'],
                'length': 38
            }
        }
    }
    
    # 构建索引 / Build index
    hybrid_retriever.build_index(example_preprocessed_data)
    
    # 测试检索 / Test retrieval
    results = hybrid_retriever.search("machine learning", top_k=2)
    logger.info(f"混合检索结果: {len(results)} / Hybrid retrieval results: {len(results)}")
    
    # 分析重叠情况 / Analyze overlap
    overlap_analysis = hybrid_retriever.analyze_retrieval_overlap("machine learning")
    logger.info(f"重叠分析: 重叠率={overlap_analysis['overlap_ratio']:.3f} / "
               f"Overlap analysis: overlap_ratio={overlap_analysis['overlap_ratio']:.3f}")