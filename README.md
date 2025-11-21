# 🎯 学术级客户服务邮件检索评估系统

完整的端到端检索评估项目，包含数据生成、质量控制、批量实验、可视化和可选的模型训练。

## 🚀 快速开始

### 选项A：Google Colab（推荐）⭐

**优势**：免费 GPU、无需本地配置、运行更快

1. 访问：https://colab.research.google.com/
2. 查看：[COLAB_GUIDE.md](COLAB_GUIDE.md) 或 [COLAB_QUICK_START.txt](COLAB_QUICK_START.txt)
3. 复制代码运行，3步完成！

**预计时间**：10-15分钟（首次），3-5分钟（后续）

---

### 选项B：本地运行（M4 Mac 或其他）

使用现有数据快速运行实验：

```bash
# 1. 给脚本执行权限
chmod +x quick_start.sh

# 2. 运行
./quick_start.sh
```

这会自动：
- ✅ 安装依赖
- ✅ 检查数据质量
- ✅ 运行多个配置的实验
- ✅ 生成可视化报告

**预计时间：** 15-30分钟（首次需要下载模型）

---

## 📦 核心脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| `generateDate.py` | 数据生成 | 使用 OpenAI 生成邮件对话 |
| `split_threads.py` | 数据分割 | 按主题分层抽样 (80/10/10) |
| `deduplicate.py` | 数据去重 | 移除重复或相似的 threads |
| `pii_mask.py` | PII 脱敏 | 替换个人信息 |
| `quality_check.py` | 质量检查 | 验证数据完整性和统计 |
| **`eval_matrix.py`** | **批量评估** | **系统化实验矩阵** ⭐ |
| `visualize_results.py` | 结果可视化 | 生成图表和报告 |
| `train_embedding.py` | 模型训练 | 微调嵌入模型（可选） |

---

## 🎯 项目目标

**验收标准：**
- ✅ Recall@10 ≥ 0.80
- ✅ MRR@10 ≥ 0.50

---

## 📊 完整工作流程

### 方案A：使用现有数据（推荐，快速开始）

```bash
# 使用你已有的 Training.jsonl 和 Testing.jsonl
./quick_start.sh
```

### 方案B：生成大规模新数据（学术项目推荐）

```bash
# 1. 生成 1000 threads（约1-2小时）
python generateDate.py --count 1000 --outdir output --provider openai --model gpt-4o-mini

# 2. 数据预处理
cat output/threads_*.jsonl > output/all_threads.jsonl
python deduplicate.py --input output/all_threads.jsonl --output data/raw/threads.jsonl
python pii_mask.py --input data/raw/threads.jsonl --output data/raw/threads_masked.jsonl --stats
python quality_check.py --file data/raw/threads_masked.jsonl

# 3. 分层分割
python split_threads.py --input data/raw/threads_masked.jsonl --output data/working

# 4. 批量实验
python eval_matrix.py \
  --test data/working/threads.test.jsonl \
  --models intfloat/e5-base-v2 BAAI/bge-small-en-v1.5 \
  --granularities thread turn \
  --k 10 --bm25 --rerank \
  --out report/experiments.csv

# 5. 生成报告
python visualize_results.py --input report/experiments.csv --output report/
```

---

## 📂 项目结构

```
.
├── README.md                    # 本文件
├── PROJECT_GUIDE.md            # 详细指南 📖
├── quick_start.sh              # 一键运行脚本
├── requirements.txt            # Python 依赖
│
├── 核心脚本/
│   ├── generateDate.py
│   ├── split_threads.py
│   ├── deduplicate.py
│   ├── pii_mask.py
│   ├── quality_check.py
│   ├── eval_matrix.py         ⭐ 批量评估
│   ├── visualize_results.py
│   └── train_embedding.py
│
├── output/                     # 生成的数据
│   ├── Training.jsonl          (现有)
│   ├── Testing.jsonl           (现有)
│   └── threads_*.jsonl
│
├── data/
│   ├── raw/                    # 清洗后数据
│   └── working/                # 分割后数据
│       ├── threads.train.jsonl
│       ├── threads.valid.jsonl
│       └── threads.test.jsonl
│
├── models/                     # 训练的模型
│   └── e5-finetuned/
│
└── report/                     # 实验报告
    ├── experiments.csv
    ├── experiments.md
    ├── comparison.png
    └── ablation.png
```

---

## 🔧 依赖安装

### 方法1：自动安装（推荐）
```bash
./quick_start.sh  # 会自动安装
```

### 方法2：手动安装
```bash
pip install -r requirements.txt
```

### 核心依赖
- `sentence-transformers` - 嵌入模型
- `faiss-cpu` - 向量检索
- `rank_bm25` - BM25 算法
- `pandas` - 数据处理
- `matplotlib` + `seaborn` - 可视化

---

## 💡 使用示例

### 示例1：快速评估（单个配置）

```bash
python eval_matrix.py \
  --test output/Testing.jsonl \
  --models intfloat/e5-base-v2 \
  --granularities turn \
  --k 10
```

### 示例2：完整对比（多配置）

```bash
python eval_matrix.py \
  --test output/Testing.jsonl \
  --models \
    sentence-transformers/all-MiniLM-L6-v2 \
    intfloat/e5-base-v2 \
    BAAI/bge-small-en-v1.5 \
  --granularities thread turn \
  --k 10 --bm25 --rerank \
  --out report/experiments.csv
```

### 示例3：模型训练（如果检索不达标）

```bash
python train_embedding.py \
  --train data/working/threads.train.jsonl \
  --valid data/working/threads.valid.jsonl \
  --base-model intfloat/e5-base-v2 \
  --output models/e5-finetuned \
  --epochs 2
```

---

## 📊 预期结果

### 使用现有数据（227 threads）

| 配置 | Recall@10 | MRR@10 | 状态 |
|------|-----------|--------|------|
| MiniLM + Thread | 0.40-0.55 | 0.25-0.40 | ❌ |
| E5-base + Turn | 0.65-0.78 | 0.50-0.65 | ⚠️ 接近 |
| E5 + Turn + Rerank | 0.75-0.85 | 0.60-0.75 | ✅ 可能达标 |

### 使用大规模数据（1000 threads）

| 配置 | Recall@10 | MRR@10 | 状态 |
|------|-----------|--------|------|
| E5-base + Turn | 0.75-0.85 | 0.60-0.75 | ✅ |
| E5 + Turn + BM25 | 0.80-0.90 | 0.65-0.80 | ✅ |
| E5 + Turn + Rerank | 0.85-0.92 | 0.70-0.85 | ✅ |
| Fine-tuned | 0.88-0.95 | 0.75-0.90 | ✅ |

---

## 🔍 关键特性

### ✨ 数据质量保证
- 按主题分层抽样（避免数据泄漏）
- 自动去重（精确 + 模糊）
- PII 脱敏（保护隐私）
- 完整性检查

### 🚀 系统化实验
- 批量运行多个配置
- 自动记录所有结果到 CSV
- 支持多种优化策略：
  - ✅ 多个嵌入模型
  - ✅ Thread vs Turn 粒度
  - ✅ BM25 混合检索
  - ✅ 交叉编码器重排

### 📊 完整可视化
- Markdown 表格（适合报告）
- 性能对比图
- 消融研究图
- 时间-性能权衡图

### 🎓 学术规范
- 80/10/10 数据分割
- 可复现（固定随机种子）
- 完整的实验记录
- 符合 ML 最佳实践

---

## 📖 文档

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - 完整项目指南（必读）
- **[CORRECT_EVALUATION_GUIDE.md](CORRECT_EVALUATION_GUIDE.md)** - 数据泄漏问题详解
- **[RETRIEVAL_OPTIMIZATION_GUIDE.md](RETRIEVAL_OPTIMIZATION_GUIDE.md)** - 优化策略指南

每个脚本都支持 `--help` 查看详细用法：
```bash
python eval_matrix.py --help
```

---

## ❓ 常见问题

### Q: 我只有 200 个 threads，够吗？
A: 可以运行实验，但结果可能不稳定。建议：
- 最低：200 threads（能跑，结果会抖动）
- 推荐：500-1000 threads（稳定结果）

### Q: 首次运行很慢？
A: 正常！首次需要下载模型（~1-2GB）。后续运行会快很多。

### Q: 如何只测试一个配置？
A: 
```bash
python eval_matrix.py \
  --test output/Testing.jsonl \
  --models intfloat/e5-base-v2 \
  --granularities turn \
  --k 10
```

### Q: 内存不足？
A: 使用更小的模型：
```bash
python eval_matrix.py \
  --models sentence-transformers/all-MiniLM-L6-v2 \
  ...
```

### Q: 需要 GPU 吗？
A: 不需要！所有脚本都支持 CPU。GPU 会更快，但不是必需的。

---

## 🎯 下一步

### 选项A：使用现有数据（快速）
```bash
./quick_start.sh
```
15-30分钟后查看 `report/experiments.md`

### 选项B：完整项目（学术）
1. 阅读 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
2. 生成 1000 threads
3. 运行完整实验矩阵
4. 生成报告

---

## 📞 需要帮助？

1. **查看详细指南**: [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
2. **查看脚本帮助**: `python <script>.py --help`
3. **检查数据质量**: `python quality_check.py --file <file>`

---

## 📄 许可证

本项目用于学术研究和教育目的。

---

## 🎉 开始实验！

```bash
# 立即开始
./quick_start.sh

# 或查看详细指南
cat PROJECT_GUIDE.md
```

祝你实验成功！🚀📈

