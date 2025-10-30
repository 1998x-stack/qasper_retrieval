"""
评估器模块
Evaluator module

实现ROUGE、BLEU、METEOR等评估指标
Implements ROUGE, BLEU, METEOR and other evaluation metrics
"""

import os
import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pandas as pd

from ..utils import get_logger, Timer, ensure_dir, save_json, load_json, save_pickle, load_pickle

logger = get_logger(__name__)


class MetricsCalculator:
    """
    评估指标计算器类
    Metrics calculator class
    
    实现各种文本评估指标的计算
    Implements calculation of various text evaluation metrics
    """
    
    def __init__(self) -> None:
        """
        初始化评估指标计算器
        Initialize metrics calculator
        """
        self._setup_nltk()
        
        # 初始化ROUGE scorer / Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # BLEU平滑函数 / BLEU smoothing function
        self.smoothing_function = SmoothingFunction().method1
        
        logger.info("评估指标计算器初始化完成 / Metrics calculator initialized")
    
    def _setup_nltk(self) -> None:
        """
        设置NLTK资源
        Setup NLTK resources
        """
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("下载NLTK WordNet资源 / Downloading NLTK WordNet resources")
            nltk.download('wordnet', quiet=True)
    
    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        Calculate ROUGE scores
        
        Args:
            prediction: 预测文本 / Prediction text
            reference: 参考文本 / Reference text
            
        Returns:
            ROUGE分数字典 / ROUGE scores dictionary
        """
        if not prediction.strip() or not reference.strip():
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e} / ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bleu_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算BLEU分数
        Calculate BLEU scores
        
        Args:
            prediction: 预测文本 / Prediction text
            reference: 参考文本 / Reference text
            
        Returns:
            BLEU分数字典 / BLEU scores dictionary
        """
        if not prediction.strip() or not reference.strip():
            return {'bleu1': 0.0, 'bleu4': 0.0}
        
        try:
            # 分词 / Tokenize
            pred_tokens = prediction.lower().split()
            ref_tokens = [reference.lower().split()]
            
            # 计算BLEU-1和BLEU-4 / Calculate BLEU-1 and BLEU-4
            bleu1 = sentence_bleu(
                ref_tokens, pred_tokens, 
                weights=(1.0, 0, 0, 0),
                smoothing_function=self.smoothing_function
            )
            
            bleu4 = sentence_bleu(
                ref_tokens, pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing_function
            )
            
            return {'bleu1': bleu1, 'bleu4': bleu4}
            
        except Exception as e:
            logger.warning(f"BLEU计算失败: {e} / BLEU calculation failed: {e}")
            return {'bleu1': 0.0, 'bleu4': 0.0}
    
    def calculate_meteor_score(self, prediction: str, reference: str) -> float:
        """
        计算METEOR分数
        Calculate METEOR score
        
        Args:
            prediction: 预测文本 / Prediction text
            reference: 参考文本 / Reference text
            
        Returns:
            METEOR分数 / METEOR score
        """
        if not prediction.strip() or not reference.strip():
            return 0.0
        
        try:
            # 分词 / Tokenize
            pred_tokens = prediction.lower().split()
            ref_tokens = reference.lower().split()
            
            score = meteor_score([ref_tokens], pred_tokens)
            return float(score)
            
        except Exception as e:
            logger.warning(f"METEOR计算失败: {e} / METEOR calculation failed: {e}")
            return 0.0
    
    def calculate_all_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        计算所有评估指标
        Calculate all evaluation metrics
        
        Args:
            prediction: 预测文本 / Prediction text
            reference: 参考文本 / Reference text
            
        Returns:
            所有评估指标字典 / All evaluation metrics dictionary
        """
        metrics = {}
        
        # ROUGE分数 / ROUGE scores
        rouge_scores = self.calculate_rouge_scores(prediction, reference)
        metrics.update(rouge_scores)
        
        # BLEU分数 / BLEU scores
        bleu_scores = self.calculate_bleu_scores(prediction, reference)
        metrics.update(bleu_scores)
        
        # METEOR分数 / METEOR score
        meteor = self.calculate_meteor_score(prediction, reference)
        metrics['meteor'] = meteor
        
        return metrics
    
    def calculate_corpus_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算语料库级别的BLEU分数
        Calculate corpus-level BLEU scores
        
        Args:
            predictions: 预测文本列表 / Prediction text list
            references: 参考文本列表 / Reference text list
            
        Returns:
            语料库BLEU分数 / Corpus BLEU scores
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考文本数量不匹配 / Prediction and reference count mismatch")
        
        # 准备数据 / Prepare data
        pred_tokens = [pred.lower().split() for pred in predictions]
        ref_tokens = [[ref.lower().split()] for ref in references]
        
        try:
            # 计算语料库BLEU / Calculate corpus BLEU
            bleu1 = corpus_bleu(
                ref_tokens, pred_tokens,
                weights=(1.0, 0, 0, 0),
                smoothing_function=self.smoothing_function
            )
            
            bleu4 = corpus_bleu(
                ref_tokens, pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing_function
            )
            
            return {'corpus_bleu1': bleu1, 'corpus_bleu4': bleu4}
            
        except Exception as e:
            logger.warning(f"语料库BLEU计算失败: {e} / Corpus BLEU calculation failed: {e}")
            return {'corpus_bleu1': 0.0, 'corpus_bleu4': 0.0}


class RetrievalEvaluator:
    """
    检索评估器类
    Retrieval evaluator class
    
    专门用于评估信息检索系统的性能
    Specialized for evaluating information retrieval system performance
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化检索评估器
        Initialize retrieval evaluator
        
        Args:
            config: 配置字典 / Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.results_dir = Path(self.eval_config.get('results_dir', './results'))
        
        # 确保结果目录存在 / Ensure results directory exists
        ensure_dir(self.results_dir)
        
        # 评估指标配置 / Evaluation metrics configuration
        self.metrics = self.eval_config.get('metrics', ['rouge', 'bleu', 'meteor'])
        self.rouge_types = self.eval_config.get('rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        self.bleu_types = self.eval_config.get('bleu_types', ['bleu1', 'bleu4'])
        self.top_k_values = self.eval_config.get('top_k', [1, 3, 5, 10])
        
        # 初始化指标计算器 / Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("检索评估器初始化完成 / Retrieval evaluator initialized")
        logger.info(f"评估指标: {self.metrics} / Evaluation metrics: {self.metrics}")
        logger.info(f"评估top-k: {self.top_k_values} / Evaluation top-k: {self.top_k_values}")
    
    def evaluate_retrieval_results(self, 
                                  queries: List[str],
                                  retrieval_results: List[List[Tuple[int, float, Dict[str, Any]]]],
                                  ground_truth: List[List[int]],
                                  passage_texts: Dict[int, str],
                                  method_name: str = "Unknown") -> Dict[str, Any]:
        """
        评估检索结果
        Evaluate retrieval results
        
        Args:
            queries: 查询列表 / Query list
            retrieval_results: 检索结果列表 / Retrieval results list
            ground_truth: 真实相关段落ID列表 / Ground truth relevant passage ID list
            passage_texts: 段落文本字典 / Passage texts dictionary
            method_name: 方法名称 / Method name
            
        Returns:
            评估结果字典 / Evaluation results dictionary
        """
        logger.info(f"开始评估检索结果，方法: {method_name} / "
                   f"Starting retrieval evaluation, method: {method_name}")
        
        if len(queries) != len(retrieval_results) or len(queries) != len(ground_truth):
            raise ValueError("查询、检索结果和真实标签数量不匹配 / "
                           "Query, retrieval results and ground truth count mismatch")
        
        with Timer(f"{method_name}检索评估 / {method_name} retrieval evaluation"):
            # 计算检索指标 / Calculate retrieval metrics
            retrieval_metrics = self._calculate_retrieval_metrics(
                retrieval_results, ground_truth
            )
            
            # 计算文本生成指标 / Calculate text generation metrics
            text_metrics = self._calculate_text_metrics(
                queries, retrieval_results, ground_truth, passage_texts
            )
            
            # 合并结果 / Merge results
            evaluation_results = {
                'method_name': method_name,
                'num_queries': len(queries),
                'retrieval_metrics': retrieval_metrics,
                'text_metrics': text_metrics,
                'query_level_results': []
            }
            
            # 计算查询级别的详细结果 / Calculate query-level detailed results
            for i, (query, results, gt_ids) in enumerate(zip(queries, retrieval_results, ground_truth)):
                query_result = self._evaluate_single_query(
                    query, results, gt_ids, passage_texts, i
                )
                evaluation_results['query_level_results'].append(query_result)
        
        logger.info(f"{method_name}检索评估完成 / {method_name} retrieval evaluation completed")
        return evaluation_results
    
    def _calculate_retrieval_metrics(self, 
                                   retrieval_results: List[List[Tuple[int, float, Dict[str, Any]]]],
                                   ground_truth: List[List[int]]) -> Dict[str, float]:
        """
        计算检索指标
        Calculate retrieval metrics
        
        Args:
            retrieval_results: 检索结果列表 / Retrieval results list
            ground_truth: 真实相关段落ID列表 / Ground truth relevant passage ID list
            
        Returns:
            检索指标字典 / Retrieval metrics dictionary
        """
        metrics = {}
        
        for k in self.top_k_values:
            # 计算精确率、召回率和F1 / Calculate precision, recall and F1
            precisions = []
            recalls = []
            f1_scores = []
            map_scores = []  # Mean Average Precision
            
            for results, gt_ids in zip(retrieval_results, ground_truth):
                # 获取top-k结果 / Get top-k results
                top_k_ids = [passage_id for passage_id, _, _ in results[:k]]
                gt_set = set(gt_ids)
                retrieved_set = set(top_k_ids)
                
                # 计算精确率 / Calculate precision
                if top_k_ids:
                    precision = len(retrieved_set & gt_set) / len(retrieved_set)
                else:
                    precision = 0.0
                precisions.append(precision)
                
                # 计算召回率 / Calculate recall
                if gt_set:
                    recall = len(retrieved_set & gt_set) / len(gt_set)
                else:
                    recall = 0.0
                recalls.append(recall)
                
                # 计算F1分数 / Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                f1_scores.append(f1)
                
                # 计算AP（Average Precision） / Calculate AP (Average Precision)
                ap = self._calculate_average_precision(top_k_ids, gt_set, k)
                map_scores.append(ap)
            
            # 计算平均值 / Calculate averages
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
            metrics[f'f1@{k}'] = np.mean(f1_scores)
            metrics[f'map@{k}'] = np.mean(map_scores)
        
        # 计算MRR（Mean Reciprocal Rank） / Calculate MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for results, gt_ids in zip(retrieval_results, ground_truth):
            gt_set = set(gt_ids)
            reciprocal_rank = 0.0
            
            for rank, (passage_id, _, _) in enumerate(results, 1):
                if passage_id in gt_set:
                    reciprocal_rank = 1.0 / rank
                    break
            
            mrr_scores.append(reciprocal_rank)
        
        metrics['mrr'] = np.mean(mrr_scores)
        
        return metrics
    
    def _calculate_average_precision(self, retrieved_ids: List[int], 
                                   relevant_ids: set, k: int) -> float:
        """
        计算平均精确率
        Calculate Average Precision
        
        Args:
            retrieved_ids: 检索到的ID列表 / Retrieved ID list
            relevant_ids: 相关ID集合 / Relevant ID set
            k: top-k值 / top-k value
            
        Returns:
            平均精确率 / Average precision
        """
        if not relevant_ids:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)
        
        if precisions:
            return np.mean(precisions)
        else:
            return 0.0
    
    def _calculate_text_metrics(self, 
                              queries: List[str],
                              retrieval_results: List[List[Tuple[int, float, Dict[str, Any]]]],
                              ground_truth: List[List[int]],
                              passage_texts: Dict[int, str]) -> Dict[str, float]:
        """
        计算文本生成评估指标
        Calculate text generation evaluation metrics
        
        Args:
            queries: 查询列表 / Query list
            retrieval_results: 检索结果列表 / Retrieval results list
            ground_truth: 真实相关段落ID列表 / Ground truth relevant passage ID list
            passage_texts: 段落文本字典 / Passage texts dictionary
            
        Returns:
            文本评估指标字典 / Text evaluation metrics dictionary
        """
        text_metrics = {}
        
        if 'rouge' not in self.metrics and 'bleu' not in self.metrics and 'meteor' not in self.metrics:
            return text_metrics
        
        # 为每个查询构建预测和参考文本 / Build prediction and reference texts for each query
        all_predictions = []
        all_references = []
        
        for query, results, gt_ids in zip(queries, retrieval_results, ground_truth):
            # 使用top-1检索结果作为预测 / Use top-1 retrieval result as prediction
            if results:
                top_passage_id = results[0][0]
                prediction = passage_texts.get(top_passage_id, "")
            else:
                prediction = ""
            
            # 使用真实相关段落作为参考（取第一个） / Use ground truth relevant passage as reference (take first)
            if gt_ids:
                reference = passage_texts.get(gt_ids[0], "")
            else:
                reference = ""
            
            all_predictions.append(prediction)
            all_references.append(reference)
        
        # 计算各种文本指标 / Calculate various text metrics
        if 'rouge' in self.metrics:
            rouge_scores = []
            for pred, ref in zip(all_predictions, all_references):
                rouge_score = self.metrics_calculator.calculate_rouge_scores(pred, ref)
                rouge_scores.append(rouge_score)
            
            # 计算平均ROUGE分数 / Calculate average ROUGE scores
            for rouge_type in self.rouge_types:
                scores = [score.get(rouge_type, 0.0) for score in rouge_scores]
                text_metrics[rouge_type] = np.mean(scores)
        
        if 'bleu' in self.metrics:
            bleu_scores = []
            for pred, ref in zip(all_predictions, all_references):
                bleu_score = self.metrics_calculator.calculate_bleu_scores(pred, ref)
                bleu_scores.append(bleu_score)
            
            # 计算平均BLEU分数 / Calculate average BLEU scores
            for bleu_type in self.bleu_types:
                scores = [score.get(bleu_type, 0.0) for score in bleu_scores]
                text_metrics[bleu_type] = np.mean(scores)
            
            # 计算语料库级别BLEU / Calculate corpus-level BLEU
            corpus_bleu = self.metrics_calculator.calculate_corpus_bleu(
                all_predictions, all_references
            )
            text_metrics.update(corpus_bleu)
        
        if 'meteor' in self.metrics:
            meteor_scores = []
            for pred, ref in zip(all_predictions, all_references):
                meteor_score = self.metrics_calculator.calculate_meteor_score(pred, ref)
                meteor_scores.append(meteor_score)
            
            text_metrics['meteor'] = np.mean(meteor_scores)
        
        return text_metrics
    
    def _evaluate_single_query(self, 
                              query: str,
                              results: List[Tuple[int, float, Dict[str, Any]]],
                              gt_ids: List[int],
                              passage_texts: Dict[int, str],
                              query_idx: int) -> Dict[str, Any]:
        """
        评估单个查询的结果
        Evaluate results for a single query
        
        Args:
            query: 查询文本 / Query text
            results: 检索结果 / Retrieval results
            gt_ids: 真实相关段落ID / Ground truth relevant passage IDs
            passage_texts: 段落文本字典 / Passage texts dictionary
            query_idx: 查询索引 / Query index
            
        Returns:
            单查询评估结果 / Single query evaluation results
        """
        query_result = {
            'query_idx': query_idx,
            'query': query,
            'num_results': len(results),
            'num_ground_truth': len(gt_ids),
            'ground_truth_ids': gt_ids
        }
        
        # 检索指标 / Retrieval metrics
        gt_set = set(gt_ids)
        for k in self.top_k_values:
            top_k_ids = [passage_id for passage_id, _, _ in results[:k]]
            retrieved_set = set(top_k_ids)
            
            # 精确率、召回率、F1 / Precision, recall, F1
            precision = len(retrieved_set & gt_set) / len(retrieved_set) if retrieved_set else 0.0
            recall = len(retrieved_set & gt_set) / len(gt_set) if gt_set else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            query_result[f'precision@{k}'] = precision
            query_result[f'recall@{k}'] = recall
            query_result[f'f1@{k}'] = f1
        
        # MRR / MRR
        reciprocal_rank = 0.0
        for rank, (passage_id, _, _) in enumerate(results, 1):
            if passage_id in gt_set:
                reciprocal_rank = 1.0 / rank
                break
        query_result['reciprocal_rank'] = reciprocal_rank
        
        # 文本指标（如果有检索结果） / Text metrics (if there are retrieval results)
        if results and gt_ids:
            top_passage_id = results[0][0]
            prediction = passage_texts.get(top_passage_id, "")
            reference = passage_texts.get(gt_ids[0], "")
            
            if prediction and reference:
                text_metrics = self.metrics_calculator.calculate_all_metrics(prediction, reference)
                query_result.update(text_metrics)
        
        return query_result
    
    def compare_methods(self, 
                       evaluation_results: List[Dict[str, Any]],
                       save_results: bool = True) -> Dict[str, Any]:
        """
        比较多种检索方法的性能
        Compare performance of multiple retrieval methods
        
        Args:
            evaluation_results: 多个方法的评估结果列表 / Evaluation results list for multiple methods
            save_results: 是否保存结果 / Whether to save results
            
        Returns:
            比较结果 / Comparison results
        """
        logger.info(f"开始比较{len(evaluation_results)}种检索方法 / "
                   f"Starting comparison of {len(evaluation_results)} retrieval methods")
        
        comparison_results = {
            'methods': [result['method_name'] for result in evaluation_results],
            'summary': {},
            'detailed_comparison': {},
            'statistical_significance': {}
        }
        
        # 收集所有指标 / Collect all metrics
        all_metrics = set()
        for result in evaluation_results:
            all_metrics.update(result['retrieval_metrics'].keys())
            all_metrics.update(result['text_metrics'].keys())
        
        # 创建比较表 / Create comparison table
        comparison_data = []
        for result in evaluation_results:
            row = {'method': result['method_name']}
            row.update(result['retrieval_metrics'])
            row.update(result['text_metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_results['comparison_table'] = comparison_df.to_dict('records')
        
        # 找出最佳方法 / Find best methods
        best_methods = {}
        for metric in all_metrics:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_methods[metric] = {
                    'method': comparison_df.loc[best_idx, 'method'],
                    'value': comparison_df.loc[best_idx, metric]
                }
        
        comparison_results['best_methods'] = best_methods
        
        # 计算平均排名 / Calculate average ranking
        method_rankings = defaultdict(list)
        for metric in all_metrics:
            if metric in comparison_df.columns:
                # 降序排名（分数越高排名越好） / Descending ranking (higher score = better ranking)
                rankings = comparison_df[metric].rank(ascending=False)
                for i, method in enumerate(comparison_df['method']):
                    method_rankings[method].append(rankings.iloc[i])
        
        avg_rankings = {}
        for method, rankings in method_rankings.items():
            avg_rankings[method] = np.mean(rankings)
        
        comparison_results['average_rankings'] = avg_rankings
        
        # 保存结果 / Save results
        if save_results:
            self._save_comparison_results(comparison_results)
        
        logger.info("检索方法比较完成 / Retrieval methods comparison completed")
        return comparison_results
    
    def _save_comparison_results(self, comparison_results: Dict[str, Any]) -> None:
        """
        保存比较结果
        Save comparison results
        
        Args:
            comparison_results: 比较结果 / Comparison results
        """
        # 保存JSON格式 / Save in JSON format
        json_path = self.results_dir / "comparison_results.json"
        save_json(comparison_results, json_path)
        
        # 保存CSV格式的比较表 / Save comparison table in CSV format
        comparison_df = pd.DataFrame(comparison_results['comparison_table'])
        csv_path = self.results_dir / "comparison_table.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        logger.info(f"比较结果已保存到: {self.results_dir} / "
                   f"Comparison results saved to: {self.results_dir}")
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any], 
                              filename: Optional[str] = None) -> str:
        """
        保存评估结果
        Save evaluation results
        
        Args:
            evaluation_results: 评估结果 / Evaluation results
            filename: 文件名 / Filename
            
        Returns:
            保存路径 / Save path
        """
        if filename is None:
            method_name = evaluation_results.get('method_name', 'unknown')
            filename = f"{method_name}_evaluation_results.json"
        
        file_path = self.results_dir / filename
        save_json(evaluation_results, file_path)
        
        logger.info(f"评估结果已保存: {file_path} / Evaluation results saved: {file_path}")
        return str(file_path)
    
    def load_evaluation_results(self, filename: str) -> Dict[str, Any]:
        """
        加载评估结果
        Load evaluation results
        
        Args:
            filename: 文件名 / Filename
            
        Returns:
            评估结果 / Evaluation results
        """
        file_path = self.results_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"评估结果文件不存在: {file_path} / "
                                   f"Evaluation results file not found: {file_path}")
        
        evaluation_results = load_json(file_path)
        logger.info(f"评估结果已加载: {file_path} / Evaluation results loaded: {file_path}")
        return evaluation_results
    
    def print_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> None:
        """
        打印评估结果摘要
        Print evaluation results summary
        
        Args:
            evaluation_results: 评估结果 / Evaluation results
        """
        method_name = evaluation_results.get('method_name', 'Unknown')
        num_queries = evaluation_results.get('num_queries', 0)
        
        logger.info(f"=== {method_name} 评估结果摘要 / {method_name} Evaluation Summary ===")
        logger.info(f"查询数量: {num_queries} / Number of queries: {num_queries}")
        
        # 检索指标 / Retrieval metrics
        retrieval_metrics = evaluation_results.get('retrieval_metrics', {})
        if retrieval_metrics:
            logger.info("\n检索指标 / Retrieval Metrics:")
            for metric, value in retrieval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # 文本指标 / Text metrics
        text_metrics = evaluation_results.get('text_metrics', {})
        if text_metrics:
            logger.info("\n文本指标 / Text Metrics:")
            for metric, value in text_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    # 测试评估器 / Test evaluator
    import yaml
    
    # 加载配置 / Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建评估器 / Create evaluator
    evaluator = RetrievalEvaluator(config)
    
    # 测试指标计算 / Test metrics calculation
    calculator = MetricsCalculator()
    prediction = "machine learning is a subset of artificial intelligence"
    reference = "machine learning is part of artificial intelligence"
    
    metrics = calculator.calculate_all_metrics(prediction, reference)
    logger.info(f"测试指标: {metrics} / Test metrics: {metrics}")
    
    # 测试检索评估 / Test retrieval evaluation
    queries = ["machine learning", "natural language processing"]
    retrieval_results = [
        [(0, 0.9, {}), (1, 0.7, {})],  # 查询1的结果 / Results for query 1
        [(1, 0.8, {}), (2, 0.6, {})]   # 查询2的结果 / Results for query 2
    ]
    ground_truth = [[0], [1, 2]]  # 真实相关段落 / Ground truth relevant passages
    passage_texts = {
        0: "machine learning deep neural networks",
        1: "natural language processing transformers", 
        2: "computer vision convolutional networks"
    }
    
    eval_results = evaluator.evaluate_retrieval_results(
        queries, retrieval_results, ground_truth, passage_texts, "TestMethod"
    )
    
    evaluator.print_evaluation_summary(eval_results)