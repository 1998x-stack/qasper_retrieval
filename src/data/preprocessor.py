"""
数据预处理器模块
Data preprocessor module

负责文本清洗、标准化和预处理
Responsible for text cleaning, normalization and preprocessing
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from transformers import AutoTokenizer

from ..utils import get_logger, Timer, ensure_dir, save_json, load_json

logger = get_logger(__name__)


class TextPreprocessor:
    """
    文本预处理器类
    Text preprocessor class
    
    提供各种文本清洗和预处理功能
    Provides various text cleaning and preprocessing functions
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化文本预处理器
        Initialize text preprocessor
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self._setup_nltk()
        self._setup_spacy()
        self._setup_jieba()
        
        # 创建词干提取器和词形还原器 / Create stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # 停用词 / Stop words
        self.stop_words = set(stopwords.words('english'))
        
        logger.info("文本预处理器初始化完成 / Text preprocessor initialized")
    
    def _setup_nltk(self) -> None:
        """
        设置NLTK资源
        Setup NLTK resources
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            logger.info("下载NLTK资源 / Downloading NLTK resources")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _setup_spacy(self) -> None:
        """
        设置SpaCy模型
        Setup SpaCy model
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy英文模型加载成功 / SpaCy English model loaded successfully")
        except OSError:
            logger.warning("SpaCy英文模型未安装，某些功能可能不可用 / "
                          "SpaCy English model not installed, some features may be unavailable")
            self.nlp = None
    
    def _setup_jieba(self) -> None:
        """
        设置jieba分词器
        Setup jieba tokenizer
        """
        # jieba默认已经可用 / jieba is available by default
        logger.info("jieba分词器设置完成 / jieba tokenizer setup completed")
    
    def clean_text(self, text: str, 
                   remove_html: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_extra_whitespace: bool = True,
                   normalize_unicode: bool = True) -> str:
        """
        清洗文本
        Clean text
        
        Args:
            text: 输入文本 / Input text
            remove_html: 是否移除HTML标签 / Whether to remove HTML tags
            remove_urls: 是否移除URL / Whether to remove URLs
            remove_emails: 是否移除邮箱地址 / Whether to remove email addresses
            remove_extra_whitespace: 是否移除多余空白字符 / Whether to remove extra whitespace
            normalize_unicode: 是否标准化Unicode / Whether to normalize Unicode
            
        Returns:
            清洗后的文本 / Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned_text = text
        
        # Unicode标准化 / Unicode normalization
        if normalize_unicode:
            cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
        
        # 移除HTML标签 / Remove HTML tags
        if remove_html:
            cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        # 移除URL / Remove URLs
        if remove_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            cleaned_text = re.sub(url_pattern, '', cleaned_text)
        
        # 移除邮箱地址 / Remove email addresses
        if remove_emails:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            cleaned_text = re.sub(email_pattern, '', cleaned_text)
        
        # 移除多余空白字符 / Remove extra whitespace
        if remove_extra_whitespace:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def tokenize_text(self, text: str, method: str = 'nltk') -> List[str]:
        """
        文本分词
        Tokenize text
        
        Args:
            text: 输入文本 / Input text
            method: 分词方法 / Tokenization method (nltk, spacy, jieba)
            
        Returns:
            分词结果 / Tokenization result
        """
        if not text:
            return []
        
        if method == 'nltk':
            return word_tokenize(text.lower())
        
        elif method == 'spacy':
            if self.nlp is None:
                logger.warning("SpaCy模型不可用，回退到NLTK / SpaCy model unavailable, falling back to NLTK")
                return word_tokenize(text.lower())
            
            doc = self.nlp(text.lower())
            return [token.text for token in doc if not token.is_space]
        
        elif method == 'jieba':
            return list(jieba.cut(text.lower()))
        
        else:
            raise ValueError(f"不支持的分词方法: {method} / Unsupported tokenization method: {method}")
    
    def remove_stopwords(self, tokens: List[str], 
                        custom_stopwords: Optional[List[str]] = None) -> List[str]:
        """
        移除停用词
        Remove stop words
        
        Args:
            tokens: 词汇列表 / Token list
            custom_stopwords: 自定义停用词 / Custom stop words
            
        Returns:
            移除停用词后的词汇列表 / Token list after removing stop words
        """
        stopwords_set = self.stop_words.copy()
        if custom_stopwords:
            stopwords_set.update(custom_stopwords)
        
        return [token for token in tokens if token not in stopwords_set and len(token) > 1]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        词干提取
        Stem tokens
        
        Args:
            tokens: 词汇列表 / Token list
            
        Returns:
            词干提取后的词汇列表 / Stemmed token list
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        词形还原
        Lemmatize tokens
        
        Args:
            tokens: 词汇列表 / Token list
            
        Returns:
            词形还原后的词汇列表 / Lemmatized token list
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        移除标点符号
        Remove punctuation
        
        Args:
            tokens: 词汇列表 / Token list
            
        Returns:
            移除标点后的词汇列表 / Token list after removing punctuation
        """
        return [token for token in tokens if token not in string.punctuation]
    
    def filter_tokens_by_length(self, tokens: List[str], 
                               min_length: int = 2, 
                               max_length: int = 50) -> List[str]:
        """
        按长度过滤词汇
        Filter tokens by length
        
        Args:
            tokens: 词汇列表 / Token list
            min_length: 最小长度 / Minimum length
            max_length: 最大长度 / Maximum length
            
        Returns:
            过滤后的词汇列表 / Filtered token list
        """
        return [token for token in tokens if min_length <= len(token) <= max_length]
    
    def process_text_pipeline(self, text: str,
                             clean_text: bool = True,
                             tokenize: bool = True,
                             remove_stopwords: bool = True,
                             remove_punctuation: bool = True,
                             lemmatize: bool = True,
                             filter_length: bool = True,
                             tokenization_method: str = 'nltk') -> List[str]:
        """
        文本处理流水线
        Text processing pipeline
        
        Args:
            text: 输入文本 / Input text
            clean_text: 是否清洗文本 / Whether to clean text
            tokenize: 是否分词 / Whether to tokenize
            remove_stopwords: 是否移除停用词 / Whether to remove stop words
            remove_punctuation: 是否移除标点 / Whether to remove punctuation
            lemmatize: 是否词形还原 / Whether to lemmatize
            filter_length: 是否按长度过滤 / Whether to filter by length
            tokenization_method: 分词方法 / Tokenization method
            
        Returns:
            处理后的词汇列表 / Processed token list
        """
        if not text:
            return []
        
        # 文本清洗 / Text cleaning
        if clean_text:
            text = self.clean_text(text)
        
        # 分词 / Tokenization
        if tokenize:
            tokens = self.tokenize_text(text, method=tokenization_method)
        else:
            tokens = [text]
        
        # 移除标点符号 / Remove punctuation
        if remove_punctuation:
            tokens = self.remove_punctuation(tokens)
        
        # 移除停用词 / Remove stop words
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # 词形还原 / Lemmatization
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # 按长度过滤 / Filter by length
        if filter_length:
            tokens = self.filter_tokens_by_length(tokens)
        
        return tokens


class QASPERPreprocessor:
    """
    QASPER数据预处理器类
    QASPER data preprocessor class
    
    专门用于处理QASPER数据集的预处理
    Specialized for preprocessing QASPER dataset
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化QASPER预处理器
        Initialize QASPER preprocessor
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.text_preprocessor = TextPreprocessor(config)
        self.cache_dir = Path(config['dataset']['cache_dir'])
        
        # 预处理配置 / Preprocessing configuration
        self.max_passage_length = config.get('retrieval', {}).get('max_passage_length', 1000)
        self.overlap_threshold = config.get('retrieval', {}).get('overlap_threshold', 0.8)
        
        logger.info("QASPER预处理器初始化完成 / QASPER preprocessor initialized")
    
    def preprocess_dataset(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        预处理整个数据集
        Preprocess entire dataset
        
        Args:
            dataset: 原始数据集 / Raw dataset
            
        Returns:
            预处理后的数据集 / Preprocessed dataset
        """
        logger.info("开始预处理QASPER数据集 / Starting to preprocess QASPER dataset")
        
        preprocessed_dataset = {
            'documents': {},
            'passages': {},
            'questions': {},
            'corpus': {
                'passages': [],
                'passage_ids': [],
                'document_ids': []
            }
        }
        
        passage_id = 0
        question_id = 0
        
        for split_name, split_data in dataset.items():
            logger.info(f"处理{split_name}分割 / Processing {split_name} split")
            
            with Timer(f"{split_name}分割预处理 / {split_name} split preprocessing"):
                for doc_data in split_data:
                    # 预处理文档 / Preprocess document
                    processed_doc = self._preprocess_document(doc_data)
                    preprocessed_dataset['documents'][processed_doc['paper_id']] = processed_doc
                    
                    # 预处理段落 / Preprocess passages
                    for passage in doc_data['passages']:
                        processed_passage = self._preprocess_passage(passage, processed_doc['paper_id'])
                        processed_passage['passage_id'] = passage_id
                        processed_passage['split'] = split_name
                        
                        preprocessed_dataset['passages'][passage_id] = processed_passage
                        
                        # 添加到语料库 / Add to corpus
                        preprocessed_dataset['corpus']['passages'].append(processed_passage['processed_text'])
                        preprocessed_dataset['corpus']['passage_ids'].append(passage_id)
                        preprocessed_dataset['corpus']['document_ids'].append(processed_doc['paper_id'])
                        
                        passage_id += 1
                    
                    # 预处理问题 / Preprocess questions
                    for qa_pair in doc_data['qa_pairs']:
                        processed_question = self._preprocess_question(qa_pair, processed_doc['paper_id'])
                        processed_question['question_id'] = question_id
                        processed_question['split'] = split_name
                        
                        preprocessed_dataset['questions'][question_id] = processed_question
                        question_id += 1
        
        # 添加统计信息 / Add statistics
        preprocessed_dataset['statistics'] = self._compute_statistics(preprocessed_dataset)
        
        logger.info("QASPER数据集预处理完成 / QASPER dataset preprocessing completed")
        return preprocessed_dataset
    
    def _preprocess_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理单个文档
        Preprocess single document
        
        Args:
            doc_data: 文档数据 / Document data
            
        Returns:
            预处理后的文档 / Preprocessed document
        """
        # 清洗标题和摘要 / Clean title and abstract
        cleaned_title = self.text_preprocessor.clean_text(doc_data['title'])
        cleaned_abstract = self.text_preprocessor.clean_text(doc_data['abstract'])
        
        # 处理标题和摘要文本 / Process title and abstract text
        title_tokens = self.text_preprocessor.process_text_pipeline(cleaned_title)
        abstract_tokens = self.text_preprocessor.process_text_pipeline(cleaned_abstract)
        
        return {
            'paper_id': doc_data['paper_id'],
            'title': cleaned_title,
            'abstract': cleaned_abstract,
            'title_tokens': title_tokens,
            'abstract_tokens': abstract_tokens,
            'num_passages': doc_data['num_passages'],
            'num_questions': doc_data['num_questions']
        }
    
    def _preprocess_passage(self, passage: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
        """
        预处理单个段落
        Preprocess single passage
        
        Args:
            passage: 段落数据 / Passage data
            paper_id: 论文ID / Paper ID
            
        Returns:
            预处理后的段落 / Preprocessed passage
        """
        # 清洗文本 / Clean text
        cleaned_text = self.text_preprocessor.clean_text(passage['text'])
        
        # 截断过长的段落 / Truncate overly long passages
        if len(cleaned_text) > self.max_passage_length:
            cleaned_text = cleaned_text[:self.max_passage_length]
            logger.debug(f"段落文本被截断到{self.max_passage_length}字符 / "
                        f"Passage text truncated to {self.max_passage_length} characters")
        
        # 处理文本 / Process text
        processed_tokens = self.text_preprocessor.process_text_pipeline(cleaned_text)
        
        return {
            'paper_id': paper_id,
            'section_name': passage['section_name'],
            'section_idx': passage['section_idx'],
            'paragraph_idx': passage['paragraph_idx'],
            'original_text': passage['text'],
            'cleaned_text': cleaned_text,
            'processed_text': ' '.join(processed_tokens),
            'tokens': processed_tokens,
            'length': len(cleaned_text),
            'num_tokens': len(processed_tokens)
        }
    
    def _preprocess_question(self, qa_pair: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
        """
        预处理单个问题
        Preprocess single question
        
        Args:
            qa_pair: 问答对数据 / QA pair data
            paper_id: 论文ID / Paper ID
            
        Returns:
            预处理后的问题 / Preprocessed question
        """
        # 清洗问题文本 / Clean question text
        cleaned_question = self.text_preprocessor.clean_text(qa_pair['question'])
        
        # 处理问题文本 / Process question text
        question_tokens = self.text_preprocessor.process_text_pipeline(cleaned_question)
        
        # 处理答案 / Process answers
        processed_answers = []
        for answer in qa_pair['answers']:
            if answer['answer_text']:
                cleaned_answer = self.text_preprocessor.clean_text(answer['answer_text'])
                answer_tokens = self.text_preprocessor.process_text_pipeline(cleaned_answer)
                
                processed_answers.append({
                    'original_text': answer['answer_text'],
                    'cleaned_text': cleaned_answer,
                    'processed_text': ' '.join(answer_tokens),
                    'tokens': answer_tokens,
                    'evidence': answer['evidence'],
                    'extractive_spans': answer['extractive_spans'],
                    'yes_no': answer['yes_no'],
                    'unanswerable': answer['unanswerable']
                })
        
        return {
            'paper_id': paper_id,
            'original_question': qa_pair['question'],
            'cleaned_question': cleaned_question,
            'processed_question': ' '.join(question_tokens),
            'question_tokens': question_tokens,
            'answers': processed_answers,
            'num_answers': len(processed_answers)
        }
    
    def _compute_statistics(self, preprocessed_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算预处理后数据集的统计信息
        Compute statistics for preprocessed dataset
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
            
        Returns:
            统计信息 / Statistics
        """
        stats = {
            'num_documents': len(preprocessed_dataset['documents']),
            'num_passages': len(preprocessed_dataset['passages']),
            'num_questions': len(preprocessed_dataset['questions']),
            'corpus_size': len(preprocessed_dataset['corpus']['passages']),
            'avg_passage_length': 0,
            'avg_passage_tokens': 0,
            'avg_question_tokens': 0
        }
        
        # 计算平均段落长度 / Calculate average passage length
        if stats['num_passages'] > 0:
            total_length = sum(p['length'] for p in preprocessed_dataset['passages'].values())
            total_tokens = sum(p['num_tokens'] for p in preprocessed_dataset['passages'].values())
            stats['avg_passage_length'] = total_length / stats['num_passages']
            stats['avg_passage_tokens'] = total_tokens / stats['num_passages']
        
        # 计算平均问题token数 / Calculate average question tokens
        if stats['num_questions'] > 0:
            total_question_tokens = sum(len(q['question_tokens']) for q in preprocessed_dataset['questions'].values())
            stats['avg_question_tokens'] = total_question_tokens / stats['num_questions']
        
        return stats
    
    def save_preprocessed_data(self, preprocessed_dataset: Dict[str, Any], 
                              output_path: Optional[str] = None) -> str:
        """
        保存预处理后的数据
        Save preprocessed data
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
            output_path: 输出路径 / Output path
            
        Returns:
            保存路径 / Save path
        """
        if output_path is None:
            output_path = self.cache_dir / "preprocessed_dataset.json"
        else:
            output_path = Path(output_path)
        
        logger.info(f"保存预处理数据到: {output_path} / Saving preprocessed data to: {output_path}")
        
        with Timer("预处理数据保存 / Preprocessed data saving"):
            save_json(preprocessed_dataset, output_path)
        
        logger.info("预处理数据保存完成 / Preprocessed data saved successfully")
        return str(output_path)
    
    def load_preprocessed_data(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载预处理后的数据
        Load preprocessed data
        
        Args:
            input_path: 输入路径 / Input path
            
        Returns:
            预处理后的数据集 / Preprocessed dataset
        """
        if input_path is None:
            input_path = self.cache_dir / "preprocessed_dataset.json"
        else:
            input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"预处理数据文件不存在: {input_path} / "
                                   f"Preprocessed data file not found: {input_path}")
        
        logger.info(f"加载预处理数据从: {input_path} / Loading preprocessed data from: {input_path}")
        
        with Timer("预处理数据加载 / Preprocessed data loading"):
            preprocessed_dataset = load_json(input_path)
        
        logger.info("预处理数据加载完成 / Preprocessed data loaded successfully")
        return preprocessed_dataset
    
    def print_statistics(self, preprocessed_dataset: Dict[str, Any]) -> None:
        """
        打印预处理统计信息
        Print preprocessing statistics
        
        Args:
            preprocessed_dataset: 预处理后的数据集 / Preprocessed dataset
        """
        stats = preprocessed_dataset.get('statistics', {})
        
        logger.info("=== 预处理统计信息 / Preprocessing Statistics ===")
        logger.info(f"文档数: {stats.get('num_documents', 0)} / Documents: {stats.get('num_documents', 0)}")
        logger.info(f"段落数: {stats.get('num_passages', 0)} / Passages: {stats.get('num_passages', 0)}")
        logger.info(f"问题数: {stats.get('num_questions', 0)} / Questions: {stats.get('num_questions', 0)}")
        logger.info(f"语料库大小: {stats.get('corpus_size', 0)} / Corpus size: {stats.get('corpus_size', 0)}")
        logger.info(f"平均段落长度: {stats.get('avg_passage_length', 0):.1f} / "
                   f"Average passage length: {stats.get('avg_passage_length', 0):.1f}")
        logger.info(f"平均段落token数: {stats.get('avg_passage_tokens', 0):.1f} / "
                   f"Average passage tokens: {stats.get('avg_passage_tokens', 0):.1f}")
        logger.info(f"平均问题token数: {stats.get('avg_question_tokens', 0):.1f} / "
                   f"Average question tokens: {stats.get('avg_question_tokens', 0):.1f}")


if __name__ == "__main__":
    # 测试预处理器 / Test preprocessor
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建预处理器 / Create preprocessor
    preprocessor = QASPERPreprocessor(config)
    
    # 测试文本处理 / Test text processing
    test_text = "This is a test document with some <html>tags</html> and URLs http://example.com"
    cleaned = preprocessor.text_preprocessor.clean_text(test_text)
    tokens = preprocessor.text_preprocessor.process_text_pipeline(cleaned)
    
    logger.info(f"原始文本: {test_text} / Original text: {test_text}")
    logger.info(f"清洗后文本: {cleaned} / Cleaned text: {cleaned}")
    logger.info(f"处理后tokens: {tokens} / Processed tokens: {tokens}")