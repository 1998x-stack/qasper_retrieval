"""
Embedding检索器模块
Embedding retriever module

实现基于向量相似度的文本检索功能
Implements text retrieval using vector similarity
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import faiss
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

from ..utils import get_logger, Timer, MemoryMonitor, ensure_dir, save_pickle, load_pickle, get_device, chunk_list

logger = get_logger(__name__)


class TextDataset(Dataset):
    """
    文本数据集类
    Text dataset class
    
    用于批量处理文本编码
    Used for batch text encoding
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512) -> None:
        """
        初始化文本数据集
        Initialize text dataset
        
        Args:
            texts: 文本列表 / Text list
            tokenizer: 分词器 / Tokenizer
            max_length: 最大长度 / Maximum length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}


class EmbeddingRetriever:
    """
    Embedding检索器类
    Embedding retriever class
    
    实现基于向量相似度的文档检索功能
    Implements document retrieval using vector similarity
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化Embedding检索器
        Initialize Embedding retriever
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('models', {})
        self.faiss_config = config.get('faiss', {})
        self.cache_dir = Path(config['dataset']['cache_dir'])
        
        # 模型配置 / Model configuration
        self.model_name = self.model_config.get('embedding_model', 'moka-ai/m3e-base')
        self.embedding_dim = self.model_config.get('embedding_dim', 768)
        self.max_seq_length = self.model_config.get('max_seq_length', 512)
        self.batch_size = self.model_config.get('batch_size', 32)
        self.device = get_device()
        
        # FAISS配置 / FAISS configuration
        self.index_type = self.faiss_config.get('index_type', 'IndexFlatIP')
        self.nlist = self.faiss_config.get('nlist', 100)
        self.nprobe = self.faiss_config.get('nprobe', 10)
        
        # 初始化模型和分词器 / Initialize model and tokenizer
        self.model: Optional[Union[AutoModel, SentenceTransformer]] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model()
        
        # 向量索引和数据 / Vector index and data
        self.faiss_index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.passage_ids: List[int] = []
        self.document_ids: List[str] = []
        self.passage_metadata: Dict[int, Dict[str, Any]] = {}
        self.texts: List[str] = []
        
        # 统计信息 / Statistics
        self.total_docs = 0
        self.index_built = False
        
        logger.info("Embedding检索器初始化完成 / Embedding retriever initialized")
        logger.info(f"模型: {self.model_name}, 设备: {self.device}")
    
    def _load_model(self) -> None:
        """
        加载预训练模型
        Load pretrained model
        """
        logger.info(f"加载embedding模型: {self.model_name} / Loading embedding model: {self.model_name}")
        
        try:
            # 尝试使用SentenceTransformer / Try using SentenceTransformer
            if 'sentence-transformers' in self.model_name or 'm3e' in self.model_name:
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                self.use_sentence_transformer = True
                logger.info("使用SentenceTransformer模型 / Using SentenceTransformer model")
            else:
                # 使用AutoModel / Use AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
                self.use_sentence_transformer = False
                logger.info("使用AutoModel模型 / Using AutoModel model")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e} / Model loading failed: {e}")
            raise
        
        MemoryMonitor.log_memory_usage("模型加载后 / After model loading ")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码文本为向量
        Encode texts to vectors
        
        Args:
            texts: 文本列表 / Text list
            show_progress: 是否显示进度 / Whether to show progress
            
        Returns:
            文本向量矩阵 / Text vector matrix
        """
        if not texts:
            return np.array([])
        
        logger.info(f"开始编码{len(texts)}个文本 / Starting to encode {len(texts)} texts")
        
        with Timer("文本编码 / Text encoding"):
            if self.use_sentence_transformer:
                # 使用SentenceTransformer编码 / Use SentenceTransformer encoding
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                # 使用AutoModel编码 / Use AutoModel encoding
                embeddings = self._encode_with_automodel(texts, show_progress)
        
        logger.info(f"文本编码完成，向量形状: {embeddings.shape} / "
                   f"Text encoding completed, vector shape: {embeddings.shape}")
        
        MemoryMonitor.log_memory_usage("文本编码后 / After text encoding ")
        return embeddings
    
    def _encode_with_automodel(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        使用AutoModel编码文本
        Encode texts using AutoModel
        
        Args:
            texts: 文本列表 / Text list
            show_progress: 是否显示进度 / Whether to show progress
            
        Returns:
            文本向量矩阵 / Text vector matrix
        """
        # 创建数据集和数据加载器 / Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # 避免多进程问题 / Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        embeddings = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # 移动到设备 / Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 获取模型输出 / Get model output
                outputs = self.model(**batch)
                
                # 池化操作 / Pooling operation
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    # 使用CLS token或平均池化 / Use CLS token or mean pooling
                    last_hidden_state = outputs.last_hidden_state
                    attention_mask = batch['attention_mask']
                    
                    # 平均池化 / Mean pooling
                    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
                    batch_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # 标准化 / Normalization
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
                if show_progress and (i + 1) % 10 == 0:
                    logger.info(f"已处理 {(i + 1) * self.batch_size}/{len(texts)} 个文本 / "
                               f"Processed {(i + 1) * self.batch_size}/{len(texts)} texts")
        
        return np.vstack(embeddings)
    
    def build_index(self, preprocessed_dataset: Dict[str, Any]) -> None:
        """
        构建向量索引
        Build vector index
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
        """
        logger.info("开始构建向量索引 / Starting to build vector index")
        
        with Timer("向量索引构建 / Vector index building"):
            # 提取文本数据 / Extract text data
            corpus_data = preprocessed_dataset['corpus']
            passages_data = preprocessed_dataset['passages']
            
            # 准备文本和元数据 / Prepare texts and metadata
            self.texts = []
            self.passage_ids = []
            self.document_ids = []
            self.passage_metadata = {}
            
            for i, passage_id in enumerate(corpus_data['passage_ids']):
                passage_data = passages_data[passage_id]
                
                # 使用清洗后的文本 / Use cleaned text
                text = passage_data['cleaned_text']
                self.texts.append(text)
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
            
            # 编码文本 / Encode texts
            logger.info(f"编码{len(self.texts)}个段落 / Encoding {len(self.texts)} passages")
            self.embeddings = self.encode_texts(self.texts)
            
            # 构建FAISS索引 / Build FAISS index
            self._build_faiss_index()
            
            self.total_docs = len(self.texts)
            self.index_built = True
            
            logger.info("向量索引构建完成 / Vector index built successfully")
            self._log_index_statistics()
    
    def _build_faiss_index(self) -> None:
        """
        构建FAISS索引
        Build FAISS index
        """
        logger.info(f"构建FAISS索引，类型: {self.index_type} / Building FAISS index, type: {self.index_type}")
        
        # 检查向量维度 / Check vector dimension
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("没有向量数据可构建索引 / No vector data to build index")
        
        vector_dim = self.embeddings.shape[1]
        
        # 创建FAISS索引 / Create FAISS index
        if self.index_type == 'IndexFlatL2':
            self.faiss_index = faiss.IndexFlatL2(vector_dim)
        elif self.index_type == 'IndexFlatIP':
            self.faiss_index = faiss.IndexFlatIP(vector_dim)
        elif self.index_type == 'IndexIVFFlat':
            quantizer = faiss.IndexFlatL2(vector_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, vector_dim, self.nlist)
            # 训练索引 / Train index
            logger.info("训练IVF索引 / Training IVF index")
            self.faiss_index.train(self.embeddings.astype(np.float32))
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type} / Unsupported index type: {self.index_type}")
        
        # 添加向量到索引 / Add vectors to index
        logger.info(f"添加{len(self.embeddings)}个向量到索引 / Adding {len(self.embeddings)} vectors to index")
        self.faiss_index.add(self.embeddings.astype(np.float32))
        
        # 设置搜索参数 / Set search parameters
        if self.index_type == 'IndexIVFFlat':
            self.faiss_index.nprobe = self.nprobe
        
        logger.info(f"FAISS索引构建完成，包含{self.faiss_index.ntotal}个向量 / "
                   f"FAISS index built, contains {self.faiss_index.ntotal} vectors")
    
    def _log_index_statistics(self) -> None:
        """
        记录索引统计信息
        Log index statistics
        """
        logger.info("=== Embedding索引统计信息 / Embedding Index Statistics ===")
        logger.info(f"文档总数: {self.total_docs} / Total documents: {self.total_docs}")
        logger.info(f"向量维度: {self.embedding_dim} / Vector dimension: {self.embedding_dim}")
        logger.info(f"索引类型: {self.index_type} / Index type: {self.index_type}")
        logger.info(f"模型: {self.model_name} / Model: {self.model_name}")
        if self.faiss_index:
            logger.info(f"FAISS索引大小: {self.faiss_index.ntotal} / FAISS index size: {self.faiss_index.ntotal}")
    
    def search(self, query: str, top_k: int = 10,
              min_score: float = 0.0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        执行向量检索
        Perform vector retrieval
        
        Args:
            query: 查询文本 / Query text
            top_k: 返回前k个结果 / Return top k results
            min_score: 最小分数阈值 / Minimum score threshold
            
        Returns:
            检索结果列表 / Retrieval results list
            格式: [(passage_id, score, metadata), ...]
        """
        if not self.index_built or self.faiss_index is None:
            raise ValueError("向量索引未构建，请先调用build_index / "
                           "Vector index not built, please call build_index first")
        
        # 编码查询 / Encode query
        query_embedding = self.encode_texts([query])
        if len(query_embedding) == 0:
            logger.warning(f"查询编码失败: {query} / Query encoding failed: {query}")
            return []
        
        # 执行检索 / Perform retrieval
        scores, indices = self.faiss_index.search(
            query_embedding.astype(np.float32),
            min(top_k, self.faiss_index.ntotal)
        )
        
        # 处理结果 / Process results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS返回-1表示无效结果 / FAISS returns -1 for invalid results
                break
                
            score = float(score)
            if score < min_score:
                break
            
            passage_id = self.passage_ids[idx]
            metadata = self.passage_metadata[passage_id].copy()
            metadata['document_id'] = self.document_ids[idx]
            metadata['rank'] = i + 1
            
            results.append((passage_id, score, metadata))
        
        logger.debug(f"向量检索完成，查询: '{query}'，返回{len(results)}个结果 / "
                    f"Vector retrieval completed, query: '{query}', returned {len(results)} results")
        
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
        logger.info(f"开始批量向量检索，查询数量: {len(queries)} / "
                   f"Starting batch vector retrieval, number of queries: {len(queries)}")
        
        with Timer("批量向量检索 / Batch vector retrieval"):
            # 批量编码查询 / Batch encode queries
            query_embeddings = self.encode_texts(queries, show_progress=True)
            
            # 批量检索 / Batch retrieval
            all_scores, all_indices = self.faiss_index.search(
                query_embeddings.astype(np.float32),
                min(top_k, self.faiss_index.ntotal)
            )
            
            # 处理结果 / Process results
            results = []
            for i, (scores, indices) in enumerate(zip(all_scores, all_indices)):
                query_results = []
                for j, (score, idx) in enumerate(zip(scores, indices)):
                    if idx == -1:
                        break
                        
                    score = float(score)
                    if score < min_score:
                        break
                    
                    passage_id = self.passage_ids[idx]
                    metadata = self.passage_metadata[passage_id].copy()
                    metadata['document_id'] = self.document_ids[idx]
                    metadata['rank'] = j + 1
                    
                    query_results.append((passage_id, score, metadata))
                
                results.append(query_results)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(queries)} 个查询 / "
                               f"Processed {i + 1}/{len(queries)} queries")
        
        logger.info("批量向量检索完成 / Batch vector retrieval completed")
        return results
    
    def compute_similarity_matrix(self, queries: List[str], 
                                 passage_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        计算查询与段落的相似度矩阵
        Compute similarity matrix between queries and passages
        
        Args:
            queries: 查询列表 / Query list
            passage_indices: 段落索引列表，None表示所有段落 / Passage indices, None for all passages
            
        Returns:
            相似度矩阵 / Similarity matrix
        """
        # 编码查询 / Encode queries
        query_embeddings = self.encode_texts(queries)
        
        # 获取段落向量 / Get passage vectors
        if passage_indices is None:
            passage_embeddings = self.embeddings
        else:
            passage_embeddings = self.embeddings[passage_indices]
        
        # 计算相似度 / Compute similarity
        similarity_matrix = np.dot(query_embeddings, passage_embeddings.T)
        
        return similarity_matrix
    
    def save_index(self, index_path: Optional[str] = None) -> str:
        """
        保存向量索引
        Save vector index
        
        Args:
            index_path: 索引保存路径 / Index save path
            
        Returns:
            保存路径 / Save path
        """
        if not self.index_built or self.faiss_index is None:
            raise ValueError("没有向量索引可保存 / No vector index to save")
        
        if index_path is None:
            index_dir = self.cache_dir / "embedding_index"
            ensure_dir(index_dir)
        else:
            index_dir = Path(index_path)
            ensure_dir(index_dir)
        
        logger.info(f"保存向量索引到: {index_dir} / Saving vector index to: {index_dir}")
        
        with Timer("向量索引保存 / Vector index saving"):
            # 保存FAISS索引 / Save FAISS index
            faiss_path = index_dir / "faiss_index.bin"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            # 保存元数据 / Save metadata
            metadata = {
                'embeddings': self.embeddings,
                'passage_ids': self.passage_ids,
                'document_ids': self.document_ids,
                'passage_metadata': self.passage_metadata,
                'texts': self.texts,
                'config': {
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim,
                    'max_seq_length': self.max_seq_length,
                    'index_type': self.index_type,
                    'nlist': self.nlist,
                    'nprobe': self.nprobe
                },
                'statistics': {
                    'total_docs': self.total_docs,
                    'index_built': self.index_built
                }
            }
            
            metadata_path = index_dir / "metadata.pkl"
            save_pickle(metadata, metadata_path)
        
        logger.info("向量索引保存完成 / Vector index saved successfully")
        return str(index_dir)
    
    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        加载向量索引
        Load vector index
        
        Args:
            index_path: 索引文件路径 / Index file path
        """
        if index_path is None:
            index_dir = self.cache_dir / "embedding_index"
        else:
            index_dir = Path(index_path)
        
        if not index_dir.exists():
            raise FileNotFoundError(f"向量索引目录不存在: {index_dir} / "
                                   f"Vector index directory not found: {index_dir}")
        
        logger.info(f"加载向量索引从: {index_dir} / Loading vector index from: {index_dir}")
        
        with Timer("向量索引加载 / Vector index loading"):
            # 加载FAISS索引 / Load FAISS index
            faiss_path = index_dir / "faiss_index.bin"
            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS索引文件不存在: {faiss_path}")
            
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # 加载元数据 / Load metadata
            metadata_path = index_dir / "metadata.pkl"
            if not metadata_path.exists():
                raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
            
            metadata = load_pickle(metadata_path)
            
            # 恢复数据 / Restore data
            self.embeddings = metadata['embeddings']
            self.passage_ids = metadata['passage_ids']
            self.document_ids = metadata['document_ids']
            self.passage_metadata = metadata['passage_metadata']
            self.texts = metadata['texts']
            
            # 恢复配置 / Restore configuration
            config = metadata['config']
            if config['model_name'] != self.model_name:
                logger.warning(f"模型名称不匹配: 当前{self.model_name}, 索引{config['model_name']} / "
                              f"Model name mismatch: current {self.model_name}, index {config['model_name']}")
            
            # 恢复统计信息 / Restore statistics
            stats = metadata['statistics']
            self.total_docs = stats['total_docs']
            self.index_built = stats['index_built']
            
            # 设置搜索参数 / Set search parameters
            if self.index_type == 'IndexIVFFlat' and hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = self.nprobe
        
        logger.info("向量索引加载完成 / Vector index loaded successfully")
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
    
    def get_passage_embedding(self, passage_id: int) -> Optional[np.ndarray]:
        """
        获取段落的向量表示
        Get vector representation of passage
        
        Args:
            passage_id: 段落ID / Passage ID
            
        Returns:
            段落向量 / Passage vector
        """
        try:
            idx = self.passage_ids.index(passage_id)
            return self.embeddings[idx]
        except ValueError:
            return None


if __name__ == "__main__":
    # 测试Embedding检索器 / Test Embedding retriever
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建检索器 / Create retriever
    retriever = EmbeddingRetriever(config)
    
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
                'length': 41
            },
            1: {
                'paper_id': 'doc1',
                'section_name': 'Methods',
                'section_idx': 1,
                'paragraph_idx': 0,
                'original_text': 'Natural language processing with transformers...',
                'cleaned_text': 'natural language processing transformers',
                'length': 39
            },
            2: {
                'paper_id': 'doc2',
                'section_name': 'Introduction',
                'section_idx': 0,
                'paragraph_idx': 0,
                'original_text': 'Computer vision using convolutional networks...',
                'cleaned_text': 'computer vision convolutional networks',
                'length': 38
            }
        }
    }
    
    # 构建索引 / Build index
    retriever.build_index(example_preprocessed_data)
    
    # 测试检索 / Test retrieval
    results = retriever.search("machine learning", top_k=2)
    logger.info(f"检索结果: {results} / Retrieval results: {results}")
    
    # 计算相似度矩阵 / Compute similarity matrix
    similarity_matrix = retriever.compute_similarity_matrix(["machine learning", "computer vision"])
    logger.info(f"相似度矩阵形状: {similarity_matrix.shape} / Similarity matrix shape: {similarity_matrix.shape}")