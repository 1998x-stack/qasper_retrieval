"""
BM25检索器模块
BM25 retriever module

实现基于BM25算法的文本检索功能
Implements text retrieval using BM25 algorithm
"""

import math
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from ..utils import get_logger, Timer, ensure_dir, save_pickle, load_pickle
from ..data import TextPreprocessor

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25检索器类
    BM25 retriever class
    
    实现基于BM25算法的文档检索功能
    Implements document retrieval using BM25 algorithm
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化BM25检索器
        Initialize BM25 retriever
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.bm25_config = config.get('bm25', {})
        self.cache_dir = Path(config['dataset']['cache_dir'])
        
        # BM25参数 / BM25 parameters
        self.k1 = self.bm25_config.get('k1', 1.2)
        self.b = self.bm25_config.get('b', 0.75)
        self.epsilon = self.bm25_config.get('epsilon', 0.25)
        self.tokenizer_type = self.bm25_config.get('tokenizer', 'nltk')
        
        # 初始化文本预处理器 / Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(config)
        
        # BM25模型和索引 / BM25 model and index
        self.bm25_model: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self.passage_ids: List[int] = []
        self.document_ids: List[str] = []
        self.passage_metadata: Dict[int, Dict[str, Any]] = {}
        
        # 统计信息 / Statistics
        self.vocab_size = 0
        self.avg_doc_length = 0.0
        self.total_docs = 0
        
        logger.info("BM25检索器初始化完成 / BM25 retriever initialized")
        logger.info(f"BM25参数 - k1: {self.k1}, b: {self.b}, epsilon: {self.epsilon}")
    
    def build_index(self, preprocessed_dataset: Dict[str, Any]) -> None:
        """
        构建BM25索引
        Build BM25 index
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
        """
        logger.info("开始构建BM25索引 / Starting to build BM25 index")
        
        with Timer("BM25索引构建 / BM25 index building"):
            # 提取语料库 / Extract corpus
            corpus_data = preprocessed_dataset['corpus']
            passages_data = preprocessed_dataset['passages']
            
            # 构建tokenized corpus / Build tokenized corpus
            self.corpus = []
            self.passage_ids = []
            self.document_ids = []
            self.passage_metadata = {}
            
            for i, passage_id in enumerate(corpus_data['passage_ids']):
                # 获取段落数据 / Get passage data
                passage_data = passages_data[passage_id]
                
                # 根据配置选择处理方式 / Choose processing method based on config
                if 'tokens' in passage_data and passage_data['tokens']:
                    tokens = passage_data['tokens']
                else:
                    # 如果没有预处理的tokens，重新处理 / If no preprocessed tokens, reprocess
                    tokens = self.text_preprocessor.process_text_pipeline(
                        passage_data['cleaned_text'],
                        tokenization_method=self.tokenizer_type
                    )
                
                self.corpus.append(tokens)
                self.passage_ids.append(passage_id)
                self.document_ids.append(corpus_data['document_ids'][i])
                
                # 保存段落元数据 / Save passage metadata
                self.passage_metadata[passage_id] = {
                    'paper_id': passage_data['paper_id'],
                    'section_name': passage_data['section_name'],
                    'section_idx': passage_data['section_idx'],
                    'paragraph_idx': passage_data['paragraph_idx'],
                    'original_text': passage_data['original_text'],
                    'cleaned_text': passage_data['cleaned_text'],
                    'length': passage_data['length']
                }
            
            # 构建BM25模型 / Build BM25 model
            logger.info(f"构建BM25模型，语料库大小: {len(self.corpus)} / "
                       f"Building BM25 model, corpus size: {len(self.corpus)}")
            
            self.bm25_model = BM25Okapi(
                self.corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon
            )
            
            # 计算统计信息 / Calculate statistics
            self._compute_statistics()
            
            logger.info("BM25索引构建完成 / BM25 index built successfully")
            self._log_index_statistics()
    
    def _compute_statistics(self) -> None:
        """
        计算索引统计信息
        Compute index statistics
        """
        if not self.corpus:
            return
        
        self.total_docs = len(self.corpus)
        
        # 计算词汇表大小 / Calculate vocabulary size
        vocab = set()
        total_length = 0
        
        for doc in self.corpus:
            vocab.update(doc)
            total_length += len(doc)
        
        self.vocab_size = len(vocab)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0.0
    
    def _log_index_statistics(self) -> None:
        """
        记录索引统计信息
        Log index statistics
        """
        logger.info("=== BM25索引统计信息 / BM25 Index Statistics ===")
        logger.info(f"文档总数: {self.total_docs} / Total documents: {self.total_docs}")
        logger.info(f"词汇表大小: {self.vocab_size} / Vocabulary size: {self.vocab_size}")
        logger.info(f"平均文档长度: {self.avg_doc_length:.2f} / Average document length: {self.avg_doc_length:.2f}")
        logger.info(f"BM25参数 - k1: {self.k1}, b: {self.b}, epsilon: {self.epsilon}")
    
    def search(self, query: str, top_k: int = 10, 
              min_score: float = 0.0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        执行BM25检索
        Perform BM25 retrieval
        
        Args:
            query: 查询文本 / Query text
            top_k: 返回前k个结果 / Return top k results
            min_score: 最小分数阈值 / Minimum score threshold
            
        Returns:
            检索结果列表 / Retrieval results list
            格式: [(passage_id, score, metadata), ...]
        """
        if self.bm25_model is None:
            raise ValueError("BM25索引未构建，请先调用build_index / "
                           "BM25 index not built, please call build_index first")
        
        # 预处理查询 / Preprocess query
        query_tokens = self.text_preprocessor.process_text_pipeline(
            query,
            tokenization_method=self.tokenizer_type
        )
        
        if not query_tokens:
            logger.warning(f"查询预处理后为空: {query} / Query is empty after preprocessing: {query}")
            return []
        
        # 执行BM25检索 / Perform BM25 retrieval
        scores = self.bm25_model.get_scores(query_tokens)
        
        # 获取排序后的结果 / Get sorted results
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for i in sorted_indices[:top_k]:
            score = float(scores[i])
            if score < min_score:
                break
                
            passage_id = self.passage_ids[i]
            metadata = self.passage_metadata[passage_id].copy()
            metadata['document_id'] = self.document_ids[i]
            metadata['rank'] = len(results) + 1
            
            results.append((passage_id, score, metadata))
        
        logger.debug(f"BM25检索完成，查询: '{query}'，返回{len(results)}个结果 / "
                    f"BM25 retrieval completed, query: '{query}', returned {len(results)} results")
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10,
                    min_score: float = 0.0) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        """
        批量检索
        Batch retrieval
        
        Args:
            queries: 查询列表 / Query list
            top_k: 返回前k个结果 / Return top k results
            min_score: 最小分数阈值 / Minimum score threshold
            
        Returns:
            批量检索结果 / Batch retrieval results
        """
        logger.info(f"开始批量BM25检索，查询数量: {len(queries)} / "
                   f"Starting batch BM25 retrieval, number of queries: {len(queries)}")
        
        results = []
        with Timer("批量BM25检索 / Batch BM25 retrieval"):
            for i, query in enumerate(queries):
                query_results = self.search(query, top_k, min_score)
                results.append(query_results)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(queries)} 个查询 / "
                               f"Processed {i + 1}/{len(queries)} queries")
        
        logger.info(f"批量BM25检索完成 / Batch BM25 retrieval completed")
        return results
    
    def get_document_frequency(self, term: str) -> int:
        """
        获取词汇的文档频率
        Get document frequency of a term
        
        Args:
            term: 词汇 / Term
            
        Returns:
            文档频率 / Document frequency
        """
        if self.bm25_model is None:
            return 0
        
        term_tokens = self.text_preprocessor.process_text_pipeline(
            term,
            tokenization_method=self.tokenizer_type
        )
        
        if not term_tokens:
            return 0
        
        token = term_tokens[0]  # 取第一个token
        return self.bm25_model.doc_freqs.get(token, 0)
    
    def get_term_scores(self, query: str) -> Dict[str, List[float]]:
        """
        获取查询中每个词汇的分数分布
        Get score distribution for each term in query
        
        Args:
            query: 查询文本 / Query text
            
        Returns:
            词汇分数字典 / Term scores dictionary
        """
        if self.bm25_model is None:
            raise ValueError("BM25索引未构建 / BM25 index not built")
        
        query_tokens = self.text_preprocessor.process_text_pipeline(
            query,
            tokenization_method=self.tokenizer_type
        )
        
        term_scores = {}
        for token in set(query_tokens):
            scores = []
            for doc in self.corpus:
                # 计算单个词汇的BM25分数 / Calculate BM25 score for single term
                tf = doc.count(token)
                if tf > 0:
                    idf = self.bm25_model.idf.get(token, 0)
                    doc_len = len(doc)
                    score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
                    scores.append(score)
                else:
                    scores.append(0.0)
            term_scores[token] = scores
        
        return term_scores
    
    def save_index(self, index_path: Optional[str] = None) -> str:
        """
        保存BM25索引
        Save BM25 index
        
        Args:
            index_path: 索引保存路径 / Index save path
            
        Returns:
            保存路径 / Save path
        """
        if self.bm25_model is None:
            raise ValueError("没有BM25索引可保存 / No BM25 index to save")
        
        if index_path is None:
            index_path = self.cache_dir / "bm25_index.pkl"
        else:
            index_path = Path(index_path)
        
        ensure_dir(index_path.parent)
        
        logger.info(f"保存BM25索引到: {index_path} / Saving BM25 index to: {index_path}")
        
        index_data = {
            'bm25_model': self.bm25_model,
            'corpus': self.corpus,
            'passage_ids': self.passage_ids,
            'document_ids': self.document_ids,
            'passage_metadata': self.passage_metadata,
            'config': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon,
                'tokenizer_type': self.tokenizer_type
            },
            'statistics': {
                'vocab_size': self.vocab_size,
                'avg_doc_length': self.avg_doc_length,
                'total_docs': self.total_docs
            }
        }
        
        with Timer("BM25索引保存 / BM25 index saving"):
            save_pickle(index_data, index_path)
        
        logger.info("BM25索引保存完成 / BM25 index saved successfully")
        return str(index_path)
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        加载BM25索引
        Load BM25 index
        
        Args:
            index_path: 索引文件路径 / Index file path
        """
        if index_path is None:
            index_path = self.cache_dir / "bm25_index.pkl"
        else:
            index_path = Path(index_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"BM25索引文件不存在: {index_path} / "
                                   f"BM25 index file not found: {index_path}")
        
        logger.info(f"加载BM25索引从: {index_path} / Loading BM25 index from: {index_path}")
        
        with Timer("BM25索引加载 / BM25 index loading"):
            index_data = load_pickle(index_path)
        
        # 恢复索引数据 / Restore index data
        self.bm25_model = index_data['bm25_model']
        self.corpus = index_data['corpus']
        self.passage_ids = index_data['passage_ids']
        self.document_ids = index_data['document_ids']
        self.passage_metadata = index_data['passage_metadata']
        
        # 恢复配置 / Restore configuration
        config = index_data['config']
        self.k1 = config['k1']
        self.b = config['b']
        self.epsilon = config['epsilon']
        self.tokenizer_type = config['tokenizer_type']
        
        # 恢复统计信息 / Restore statistics
        stats = index_data['statistics']
        self.vocab_size = stats['vocab_size']
        self.avg_doc_length = stats['avg_doc_length']
        self.total_docs = stats['total_docs']
        
        logger.info("BM25索引加载完成 / BM25 index loaded successfully")
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
        return self.passage_metadata.get(passage_id)
    
    def analyze_query_coverage(self, query: str) -> Dict[str, Any]:
        """
        分析查询在语料库中的覆盖情况
        Analyze query coverage in corpus
        
        Args:
            query: 查询文本 / Query text
            
        Returns:
            覆盖分析结果 / Coverage analysis results
        """
        if self.bm25_model is None:
            raise ValueError("BM25索引未构建 / BM25 index not built")
        
        query_tokens = self.text_preprocessor.process_text_pipeline(
            query,
            tokenization_method=self.tokenizer_type
        )
        
        analysis = {
            'query_tokens': query_tokens,
            'total_tokens': len(query_tokens),
            'unique_tokens': len(set(query_tokens)),
            'token_coverage': {},
            'covered_tokens': 0,
            'coverage_ratio': 0.0
        }
        
        for token in set(query_tokens):
            df = self.bm25_model.doc_freqs.get(token, 0)
            analysis['token_coverage'][token] = {
                'document_frequency': df,
                'coverage_ratio': df / self.total_docs if self.total_docs > 0 else 0.0,
                'in_vocabulary': token in self.bm25_model.doc_freqs
            }
            
            if df > 0:
                analysis['covered_tokens'] += 1
        
        if analysis['unique_tokens'] > 0:
            analysis['coverage_ratio'] = analysis['covered_tokens'] / analysis['unique_tokens']
        
        return analysis


if __name__ == "__main__":
    # 测试BM25检索器 / Test BM25 retriever
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建检索器 / Create retriever
    retriever = BM25Retriever(config)
    
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
    retriever.build_index(example_preprocessed_data)
    
    # 测试检索 / Test retrieval
    results = retriever.search("machine learning", top_k=2)
    logger.info(f"检索结果: {results} / Retrieval results: {results}")
    
    # 分析查询覆盖 / Analyze query coverage
    coverage = retriever.analyze_query_coverage("machine learning")
    logger.info(f"查询覆盖分析: {coverage} / Query coverage analysis: {coverage}")