"""
数据集加载器模块
Dataset loader module

负责下载和加载QASPER数据集
Responsible for downloading and loading QASPER dataset
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from datasets import load_dataset, Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..utils import get_logger, Timer, MemoryMonitor, ensure_dir, save_json, load_json

logger = get_logger(__name__)


class QASPERDatasetLoader:
    """
    QASPER数据集加载器类
    QASPER dataset loader class
    
    负责下载、缓存和加载QASPER数据集
    Responsible for downloading, caching and loading QASPER dataset
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化数据集加载器
        Initialize dataset loader
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.dataset_config = config['dataset']
        self.cache_dir = Path(self.dataset_config['cache_dir'])
        self.dataset_name = self.dataset_config['name']
        self.max_samples = self.dataset_config.get('max_samples')
        self.splits = self.dataset_config.get('splits', ['train', 'validation', 'test'])
        
        # 确保缓存目录存在 / Ensure cache directory exists
        ensure_dir(self.cache_dir)
        
        # 数据集存储 / Dataset storage
        self.raw_dataset: Optional[DatasetDict] = None
        self.processed_dataset: Optional[Dict[str, List[Dict[str, Any]]]] = None
        
        logger.info(f"QASPER数据集加载器初始化完成 / QASPER dataset loader initialized")
        logger.info(f"缓存目录: {self.cache_dir} / Cache directory: {self.cache_dir}")
        
    def download_dataset(self, force_download: bool = False) -> DatasetDict:
        """
        下载QASPER数据集
        Download QASPER dataset
        
        Args:
            force_download: 是否强制重新下载 / Whether to force re-download
            
        Returns:
            数据集字典 / Dataset dictionary
        """
        logger.info(f"开始下载QASPER数据集 / Starting to download QASPER dataset")
        
        # 检查是否已存在缓存 / Check if cache exists
        cache_path = self.cache_dir / "raw_dataset"
        if cache_path.exists() and not force_download:
            logger.info(f"发现缓存数据集，正在加载 / Found cached dataset, loading...")
            try:
                self.raw_dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir)
                )
                logger.info(f"从缓存加载数据集成功 / Successfully loaded dataset from cache")
                return self.raw_dataset
            except Exception as e:
                logger.warning(f"从缓存加载失败，重新下载: {e} / Failed to load from cache, re-downloading: {e}")
        
        # 下载数据集 / Download dataset
        try:
            with Timer("数据集下载 / Dataset download"):
                self.raw_dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir)
                )
            
            logger.info(f"数据集下载完成 / Dataset download completed")
            self._log_dataset_info()
            
            return self.raw_dataset
            
        except Exception as e:
            logger.error(f"数据集下载失败: {e} / Dataset download failed: {e}")
            raise
    
    def _log_dataset_info(self) -> None:
        """
        记录数据集信息
        Log dataset information
        """
        if self.raw_dataset is None:
            return
            
        logger.info("数据集信息 / Dataset information:")
        for split_name, split_dataset in self.raw_dataset.items():
            if split_name in self.splits:
                logger.info(f"  {split_name}: {len(split_dataset)} 样本 / {len(split_dataset)} samples")
                
        # 记录内存使用 / Log memory usage
        MemoryMonitor.log_memory_usage("数据集加载后 / After dataset loading ")
    
    def extract_text_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        从原始数据集中提取文本数据
        Extract text data from raw dataset
        
        Returns:
            提取的文本数据 / Extracted text data
        """
        if self.raw_dataset is None:
            raise ValueError("数据集未加载，请先调用download_dataset / Dataset not loaded, please call download_dataset first")
        
        logger.info("开始提取文本数据 / Starting to extract text data")
        
        extracted_data = {}
        
        for split_name in self.splits:
            if split_name not in self.raw_dataset:
                logger.warning(f"数据集中不存在分割: {split_name} / Split not found in dataset: {split_name}")
                continue
                
            split_data = []
            split_dataset = self.raw_dataset[split_name]
            
            # 限制样本数量（如果设置了） / Limit sample count (if set)
            num_samples = len(split_dataset)
            if self.max_samples is not None and self.max_samples > 0:
                num_samples = min(num_samples, self.max_samples)
                logger.info(f"{split_name}分割限制样本数量为: {num_samples} / "
                           f"{split_name} split limited to {num_samples} samples")
            
            with Timer(f"{split_name}分割文本提取 / {split_name} split text extraction"):
                for i in range(num_samples):
                    sample = split_dataset[i]
                    extracted_sample = self._extract_sample_data(sample, i)
                    split_data.append(extracted_sample)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"已处理 {i + 1}/{num_samples} 个样本 / "
                                   f"Processed {i + 1}/{num_samples} samples")
            
            extracted_data[split_name] = split_data
            logger.info(f"{split_name}分割提取完成，共{len(split_data)}个样本 / "
                       f"{split_name} split extraction completed, {len(split_data)} samples total")
        
        self.processed_dataset = extracted_data
        return extracted_data
    
    def _extract_sample_data(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """
        提取单个样本的数据
        Extract data from a single sample
        
        Args:
            sample: 原始样本数据 / Raw sample data
            sample_idx: 样本索引 / Sample index
            
        Returns:
            提取的样本数据 / Extracted sample data
        """
        try:
            # 提取基本信息 / Extract basic information
            paper_id = sample.get('id', f'unknown_{sample_idx}')
            title = sample.get('title', '')
            abstract = sample.get('abstract', '')
            
            # 提取全文 / Extract full text
            full_text = sample.get('full_text', {})
            paragraphs = full_text.get('paragraphs', [])
            section_names = full_text.get('section_name', [])
            
            # 构建文档文本 / Build document text
            document_text = self._build_document_text(title, abstract, paragraphs, section_names)
            
            # 提取段落 / Extract paragraphs
            passages = self._extract_passages(paragraphs, section_names)
            
            # 提取问答对 / Extract QA pairs
            qas = sample.get('qas', {})
            qa_pairs = self._extract_qa_pairs(qas)
            
            return {
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract,
                'document_text': document_text,
                'passages': passages,
                'qa_pairs': qa_pairs,
                'num_passages': len(passages),
                'num_questions': len(qa_pairs)
            }
            
        except Exception as e:
            logger.error(f"提取样本数据失败，样本索引: {sample_idx}, 错误: {e} / "
                        f"Failed to extract sample data, index: {sample_idx}, error: {e}")
            return {
                'paper_id': f'error_{sample_idx}',
                'title': '',
                'abstract': '',
                'document_text': '',
                'passages': [],
                'qa_pairs': [],
                'num_passages': 0,
                'num_questions': 0
            }
    
    def _build_document_text(self, title: str, abstract: str, 
                           paragraphs: List[List[str]], 
                           section_names: List[str]) -> str:
        """
        构建完整的文档文本
        Build complete document text
        
        Args:
            title: 标题 / Title
            abstract: 摘要 / Abstract
            paragraphs: 段落列表 / Paragraph list
            section_names: 章节名称列表 / Section name list
            
        Returns:
            完整文档文本 / Complete document text
        """
        document_parts = []
        
        # 添加标题 / Add title
        if title:
            document_parts.append(f"Title: {title}")
        
        # 添加摘要 / Add abstract
        if abstract:
            document_parts.append(f"Abstract: {abstract}")
        
        # 添加章节内容 / Add section content
        for i, section_paragraphs in enumerate(paragraphs):
            if i < len(section_names):
                section_name = section_names[i]
                document_parts.append(f"\n{section_name}")
            
            for paragraph in section_paragraphs:
                if paragraph.strip():
                    document_parts.append(paragraph.strip())
        
        return "\n\n".join(document_parts)
    
    def _extract_passages(self, paragraphs: List[List[str]], 
                         section_names: List[str]) -> List[Dict[str, Any]]:
        """
        提取段落作为检索单元
        Extract paragraphs as retrieval units
        
        Args:
            paragraphs: 段落列表 / Paragraph list
            section_names: 章节名称列表 / Section name list
            
        Returns:
            段落信息列表 / Passage information list
        """
        passages = []
        passage_id = 0
        
        for section_idx, section_paragraphs in enumerate(paragraphs):
            section_name = section_names[section_idx] if section_idx < len(section_names) else f"Section_{section_idx}"
            
            for para_idx, paragraph in enumerate(section_paragraphs):
                if paragraph.strip():
                    passages.append({
                        'passage_id': passage_id,
                        'section_name': section_name,
                        'section_idx': section_idx,
                        'paragraph_idx': para_idx,
                        'text': paragraph.strip(),
                        'length': len(paragraph.strip())
                    })
                    passage_id += 1
        
        return passages
    
    def _extract_qa_pairs(self, qas: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取问答对
        Extract QA pairs
        
        Args:
            qas: 问答数据 / QA data
            
        Returns:
            问答对列表 / QA pair list
        """
        qa_pairs = []
        
        questions = qas.get('question', [])
        question_ids = qas.get('question_id', [])
        answers_list = qas.get('answers', [])
        
        for i, question in enumerate(questions):
            question_id = question_ids[i] if i < len(question_ids) else f'q_{i}'
            answers = answers_list[i] if i < len(answers_list) else []
            
            # 处理答案 / Process answers
            processed_answers = []
            for answer_data in answers:
                if isinstance(answer_data, dict):
                    processed_answer = {
                        'answer_text': answer_data.get('answer', {}).get('free_form_answer', ''),
                        'evidence': answer_data.get('answer', {}).get('evidence', []),
                        'extractive_spans': answer_data.get('answer', {}).get('extractive_spans', []),
                        'yes_no': answer_data.get('answer', {}).get('yes_no', None),
                        'unanswerable': answer_data.get('answer', {}).get('unanswerable', False)
                    }
                    processed_answers.append(processed_answer)
            
            qa_pairs.append({
                'question_id': question_id,
                'question': question,
                'answers': processed_answers,
                'num_answers': len(processed_answers)
            })
        
        return qa_pairs
    
    def save_processed_data(self, output_path: Optional[str] = None) -> str:
        """
        保存处理后的数据
        Save processed data
        
        Args:
            output_path: 输出路径 / Output path
            
        Returns:
            保存路径 / Save path
        """
        if self.processed_dataset is None:
            raise ValueError("没有处理后的数据可保存 / No processed data to save")
        
        if output_path is None:
            output_path = self.cache_dir / "processed_dataset.json"
        else:
            output_path = Path(output_path)
        
        logger.info(f"保存处理后的数据到: {output_path} / Saving processed data to: {output_path}")
        
        with Timer("数据保存 / Data saving"):
            save_json(self.processed_dataset, output_path)
        
        logger.info(f"处理后的数据保存完成 / Processed data saved successfully")
        return str(output_path)
    
    def load_processed_data(self, input_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        加载处理后的数据
        Load processed data
        
        Args:
            input_path: 输入路径 / Input path
            
        Returns:
            处理后的数据 / Processed data
        """
        if input_path is None:
            input_path = self.cache_dir / "processed_dataset.json"
        else:
            input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"处理后的数据文件不存在: {input_path} / "
                                   f"Processed data file not found: {input_path}")
        
        logger.info(f"加载处理后的数据从: {input_path} / Loading processed data from: {input_path}")
        
        with Timer("数据加载 / Data loading"):
            self.processed_dataset = load_json(input_path)
        
        logger.info(f"处理后的数据加载完成 / Processed data loaded successfully")
        return self.processed_dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        Get dataset statistics
        
        Returns:
            统计信息字典 / Statistics dictionary
        """
        if self.processed_dataset is None:
            raise ValueError("数据集未处理，请先提取文本数据 / Dataset not processed, please extract text data first")
        
        stats = {
            'splits': {},
            'total_papers': 0,
            'total_passages': 0,
            'total_questions': 0
        }
        
        for split_name, split_data in self.processed_dataset.items():
            split_stats = {
                'num_papers': len(split_data),
                'num_passages': sum(sample['num_passages'] for sample in split_data),
                'num_questions': sum(sample['num_questions'] for sample in split_data),
                'avg_passages_per_paper': 0,
                'avg_questions_per_paper': 0,
                'avg_passage_length': 0
            }
            
            if split_stats['num_papers'] > 0:
                split_stats['avg_passages_per_paper'] = split_stats['num_passages'] / split_stats['num_papers']
                split_stats['avg_questions_per_paper'] = split_stats['num_questions'] / split_stats['num_papers']
            
            # 计算平均段落长度 / Calculate average passage length
            total_length = 0
            total_passages = 0
            for sample in split_data:
                for passage in sample['passages']:
                    total_length += passage['length']
                    total_passages += 1
            
            if total_passages > 0:
                split_stats['avg_passage_length'] = total_length / total_passages
            
            stats['splits'][split_name] = split_stats
            stats['total_papers'] += split_stats['num_papers']
            stats['total_passages'] += split_stats['num_passages']
            stats['total_questions'] += split_stats['num_questions']
        
        return stats
    
    def print_statistics(self) -> None:
        """
        打印数据集统计信息
        Print dataset statistics
        """
        try:
            stats = self.get_statistics()
            
            logger.info("=== QASPER数据集统计信息 / QASPER Dataset Statistics ===")
            logger.info(f"总论文数: {stats['total_papers']} / Total papers: {stats['total_papers']}")
            logger.info(f"总段落数: {stats['total_passages']} / Total passages: {stats['total_passages']}")
            logger.info(f"总问题数: {stats['total_questions']} / Total questions: {stats['total_questions']}")
            
            for split_name, split_stats in stats['splits'].items():
                logger.info(f"\n{split_name.upper()}分割统计 / {split_name.upper()} Split Statistics:")
                logger.info(f"  论文数: {split_stats['num_papers']} / Papers: {split_stats['num_papers']}")
                logger.info(f"  段落数: {split_stats['num_passages']} / Passages: {split_stats['num_passages']}")
                logger.info(f"  问题数: {split_stats['num_questions']} / Questions: {split_stats['num_questions']}")
                logger.info(f"  平均每篇论文段落数: {split_stats['avg_passages_per_paper']:.1f} / "
                           f"Avg passages per paper: {split_stats['avg_passages_per_paper']:.1f}")
                logger.info(f"  平均每篇论文问题数: {split_stats['avg_questions_per_paper']:.1f} / "
                           f"Avg questions per paper: {split_stats['avg_questions_per_paper']:.1f}")
                logger.info(f"  平均段落长度: {split_stats['avg_passage_length']:.1f} / "
                           f"Avg passage length: {split_stats['avg_passage_length']:.1f}")
            
        except Exception as e:
            logger.error(f"打印统计信息失败: {e} / Failed to print statistics: {e}")


if __name__ == "__main__":
    # 测试数据集加载器 / Test dataset loader
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集加载器 / Create dataset loader
    loader = QASPERDatasetLoader(config)
    
    # 下载数据集 / Download dataset
    dataset = loader.download_dataset()
    
    # 提取文本数据 / Extract text data
    processed_data = loader.extract_text_data()
    
    # 保存处理后的数据 / Save processed data
    save_path = loader.save_processed_data()
    
    # 打印统计信息 / Print statistics
    loader.print_statistics()