# QASPER检索评估系统 / QASPER Retrieval Evaluation System
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DAhw8wHFQGeuyZk6yxZLXZfGe1rbezDF?usp=sharing)


> 一个全面的信息检索评估系统，专门用于在QASPER数据集上比较BM25、embedding和混合检索方法。
> A comprehensive information retrieval evaluation system specifically designed for comparing BM25, embedding, and hybrid retrieval methods on the QASPER dataset.

## 📋 目录 / Table of Contents

- [功能特性](#功能特性--features)
- [系统架构](#系统架构--system-architecture)
- [安装指南](#安装指南--installation)
- [快速开始](#快速开始--quick-start)
- [详细使用](#详细使用--detailed-usage)
- [配置说明](#配置说明--configuration)
- [评估指标](#评估指标--evaluation-metrics)
- [性能优化](#性能优化--performance-optimization)
- [贡献指南](#贡献指南--contributing)
- [许可证](#许可证--license)

## 🌟 功能特性 / Features

### 核心功能 / Core Features

- **多种检索方法支持** / **Multiple Retrieval Methods**
  - BM25算法（基于词频-逆文档频率）
  - Embedding检索（基于深度学习语义向量）
  - 混合检索（BM25与embedding的智能融合）

- **完整的评估体系** / **Comprehensive Evaluation System**
  - ROUGE评估指标（ROUGE-1, ROUGE-2, ROUGE-L）
  - BLEU评估指标（BLEU-1, BLEU-4, Corpus BLEU）
  - METEOR评估指标
  - 检索评估指标（Precision@K, Recall@K, F1@K, MAP@K, MRR）

- **工业级代码质量** / **Industrial-Grade Code Quality**
  - 完整的类型注释和文档字符串
  - 鲁棒的错误处理和边界条件检查
  - 内存监控和性能优化
  - 详细的日志记录系统

### 技术特性 / Technical Features

- **高性能向量检索** / **High-Performance Vector Retrieval**
  - FAISS向量索引支持
  - 批量处理优化
  - GPU加速支持

- **灵活的配置系统** / **Flexible Configuration System**
  - YAML配置文件
  - 命令行参数支持
  - 环境变量配置

- **可扩展架构** / **Extensible Architecture**
  - 模块化设计
  - 易于添加新的检索方法
  - 插件式评估指标

## 🏗️ 系统架构 / System Architecture

```
qasper_retrieval/
├── config/                    # 配置文件 / Configuration files
│   ├── __init__.py
│   └── config.yaml           # 主配置文件 / Main config file
├── src/                      # 源代码 / Source code
│   ├── __init__.py
│   ├── data/                 # 数据处理模块 / Data processing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py # 数据集加载器 / Dataset loader
│   │   └── preprocessor.py   # 数据预处理器 / Data preprocessor
│   ├── retrieval/            # 检索模块 / Retrieval methods
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py # BM25检索器 / BM25 retriever
│   │   ├── embedding_retriever.py # Embedding检索器 / Embedding retriever
│   │   └── hybrid_retriever.py    # 混合检索器 / Hybrid retriever
│   ├── evaluation/           # 评估模块 / Evaluation module
│   │   ├── __init__.py
│   │   └── evaluator.py      # 评估器 / Evaluator
│   ├── utils/                # 工具模块 / Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py         # 日志工具 / Logging utilities
│   │   └── common.py         # 通用工具 / Common utilities
│   └── main.py               # 主程序入口 / Main entry point
├── cache/                    # 缓存目录 / Cache directory
├── logs/                     # 日志目录 / Logs directory
├── results/                  # 结果目录 / Results directory
├── requirements.txt          # 依赖包列表 / Dependencies
└── README.md                # 项目说明 / Project documentation
```

## 🚀 安装指南 / Installation

### 环境要求 / Requirements

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速 / Optional, for GPU acceleration)
- 内存：建议16GB+ / Memory: 16GB+ recommended
- 存储：至少10GB可用空间 / Storage: At least 10GB free space

### 安装步骤 / Installation Steps

1. **克隆仓库 / Clone Repository**
```bash
git clone https://github.com/your-username/qasper-retrieval.git
cd qasper-retrieval
```

2. **创建虚拟环境 / Create Virtual Environment**
```bash
# 使用conda / Using conda
conda create -n qasper-retrieval python=3.9
conda activate qasper-retrieval

# 或使用venv / Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **安装依赖 / Install Dependencies**
```bash
# 基础安装 / Basic installation
pip install -r requirements.txt

# 中国用户可使用清华源 / Chinese users can use Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU支持（可选）/ GPU support (optional)
pip install faiss-gpu
```

4. **下载SpaCy模型 / Download SpaCy Model**
```bash
python -m spacy download en_core_web_sm
```

5. **创建必要目录 / Create Necessary Directories**
```bash
mkdir -p cache logs results
```

## 🎯 快速开始 / Quick Start

### 基本使用 / Basic Usage

1. **运行完整评估 / Run Complete Evaluation**
```bash
python src/main.py --method all
```

2. **评估特定方法 / Evaluate Specific Methods**
```bash
# 只评估BM25和Embedding
python src/main.py --method bm25 embedding

# 只评估混合方法
python src/main.py --method hybrid
```

3. **快速测试（限制查询数量）/ Quick Test (Limited Queries)**
```bash
python src/main.py --method all --max-queries 50
```

### 命令行参数 / Command Line Arguments

```bash
python src/main.py [options]

选项 / Options:
  -h, --help              显示帮助信息 / Show help message
  -c, --config CONFIG     配置文件路径 / Configuration file path
  -m, --method METHOD     检索方法 / Retrieval methods [bm25|embedding|hybrid|all]
  --force-download        强制重新下载数据 / Force re-download data
  --force-preprocess      强制重新预处理 / Force re-preprocess
  --force-rebuild-index   强制重建索引 / Force rebuild indices
  --test-split SPLIT      测试数据分割 / Test data split [train|validation|test]
  --max-queries N         最大查询数量 / Maximum number of queries
  -v, --verbose           详细输出 / Verbose output
```

## 📖 详细使用 / Detailed Usage

### 配置文件说明 / Configuration File

编辑 `config/config.yaml` 来自定义系统行为：

```yaml
# 数据集配置
dataset:
  name: "allenai/qasper"
  cache_dir: "./cache"
  max_samples: null  # null表示使用全部数据

# 模型配置
models:
  embedding_model: "moka-ai/m3e-base"
  batch_size: 32
  device: "auto"  # auto, cpu, cuda, cuda:0
  trust_remote_code: false  # 仅在明确需要远程自定义代码时启用

# BM25配置
bm25:
  k1: 1.2
  b: 0.75
  tokenizer: "nltk"

# 混合检索配置
hybrid:
  bm25_weight: 0.5
  embedding_weight: 0.5
  normalization: "min_max"

# 评估配置
evaluation:
  metrics: ["rouge", "bleu", "meteor"]
  top_k: [1, 3, 5, 10]
```

### 编程接口 / Programming Interface

```python
from src import QASPERRetrievalSystem

# 创建系统实例
system = QASPERRetrievalSystem("config/config.yaml")

# 运行完整流水线
results = system.run_full_pipeline(
    methods=['bm25', 'embedding', 'hybrid'],
    max_queries=100
)

# 查看结果
for result in results['individual_results']:
    print(f"方法: {result['method_name']}")
    print(f"Precision@10: {result['retrieval_metrics']['precision@10']:.4f}")
    print(f"ROUGE-L: {result['text_metrics']['rougeL']:.4f}")
```

### 单独使用检索器 / Using Individual Retrievers

```python
from src.retrieval import BM25Retriever, EmbeddingRetriever
from src.utils import load_yaml

# 加载配置
config = load_yaml("config/config.yaml")

# 使用BM25检索器
bm25_retriever = BM25Retriever(config)
bm25_retriever.load_index()  # 加载预构建的索引
results = bm25_retriever.search("machine learning", top_k=5)

# 使用Embedding检索器
embedding_retriever = EmbeddingRetriever(config)
embedding_retriever.load_index()
results = embedding_retriever.search("natural language processing", top_k=5)
```

## 📊 评估指标 / Evaluation Metrics

### 检索评估指标 / Retrieval Metrics

- **Precision@K**: 前K个结果中相关文档的比例
- **Recall@K**: 前K个结果中找到的相关文档占所有相关文档的比例
- **F1@K**: Precision@K和Recall@K的调和平均数
- **MAP@K**: 平均精确率，考虑结果排序质量
- **MRR**: 平均倒数排名，衡量第一个相关结果的位置

### 文本生成评估指标 / Text Generation Metrics

- **ROUGE-1**: 基于单词重叠的评估
- **ROUGE-2**: 基于双词组重叠的评估
- **ROUGE-L**: 基于最长公共子序列的评估
- **BLEU-1/4**: 基于n-gram精确率的评估
- **METEOR**: 考虑词形变化和同义词的评估

## ⚡ 性能优化 / Performance Optimization

### 内存优化 / Memory Optimization

- 使用批量处理减少内存峰值
- 实时内存监控和报告
- 大文件的流式处理

### 计算优化 / Computational Optimization

- GPU加速的向量计算
- 多进程并行处理
- FAISS高效向量检索

### 存储优化 / Storage Optimization

- 压缩的索引存储格式
- 增量式数据处理
- 智能缓存机制

## 🔧 故障排除 / Troubleshooting

### 常见问题 / Common Issues

1. **内存不足 / Out of Memory**
   ```bash
   # 减少批量大小
   # 在config.yaml中设置较小的batch_size
   models:
     batch_size: 16  # 默认32
   ```

2. **CUDA错误 / CUDA Errors**
   ```bash
   # 使用CPU模式
   models:
     device: "cpu"
   ```

3. **下载失败 / Download Failures**
   ```bash
   # 使用代理或镜像源
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 日志调试 / Log Debugging

系统提供详细的日志记录：
- 日志文件位置：`logs/qasper_retrieval.log`
- 日志级别可在配置文件中调整
- 使用 `--verbose` 参数获得更详细的输出

## 📈 实验结果示例 / Example Results

| 方法 / Method | Precision@10 | ROUGE-L | BLEU-4 | MRR |
|---------------|--------------|---------|--------|-----|
| BM25          | 0.7234      | 0.3456  | 0.2134 | 0.8123 |
| Embedding     | 0.7891      | 0.3789  | 0.2456 | 0.8456 |
| Hybrid        | 0.8123      | 0.3912  | 0.2567 | 0.8678 |

## 🤝 贡献指南 / Contributing

我们欢迎社区贡献！请参考以下步骤：

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

### 代码规范 / Code Standards

- 遵循Google Python Style Guide
- 使用类型注释
- 添加详细的文档字符串
- 编写单元测试

## 📄 许可证 / License

本项目采用MIT许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

## 🙏 致谢 / Acknowledgments

- [QASPER数据集](https://allenai.org/data/qasper) - Allen Institute for AI
- [Transformers库](https://huggingface.co/transformers/) - Hugging Face
- [FAISS库](https://github.com/facebookresearch/faiss) - Facebook Research
- [rank-bm25库](https://github.com/dorianbrown/rank_bm25) - Dorian Brown

## 📞 联系方式 / Contact

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-username/qasper-retrieval/issues)
- 邮箱: your-email@example.com

---

**注意 / Note**: 这是一个研究用途的项目，请确保在使用时遵守相关数据集的使用条款。

**Note**: This is a research project. Please ensure compliance with relevant dataset usage terms when using this system.