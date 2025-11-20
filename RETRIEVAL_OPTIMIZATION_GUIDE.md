# 🚀 检索优化实战指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备测试数据

确保你有测试数据文件（JSONL格式），例如：
- `Jupiter/data/working/threads.test.jsonl`
- 或者使用生成的数据：`output/threads_openai_*.jsonl`

## 使用方法

### 方案A：一键对比所有配置（推荐）⭐

```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --compare-all
```

**这会自动运行5种配置并输出对比表格：**
1. Baseline (MiniLM + Thread-level)
2. Better Model (E5-base + Thread-level)
3. Turn-level Split (MiniLM + Turn-level)
4. Model + Turn-level (E5-base + Turn-level)
5. Full Optimization (E5-base + Turn-level + Rerank)

### 方案B：单独测试各种配置

#### 配置1：仅换更强模型
```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --model e5-base-v2
```

#### 配置2：使用 Turn-level 切分
```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --turn-level
```

#### 配置3：加重排
```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --rerank
```

#### 配置4：全部优化一起上
```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --model e5-base-v2 \
  --turn-level \
  --rerank
```

## 可选模型

| 模型名称 | 说明 | 速度 | 效果 |
|---------|------|------|------|
| `minilm` | 原版（all-MiniLM-L6-v2） | ⚡⚡⚡ 最快 | ⭐⭐ 一般 |
| `e5-base-v2` | 更强的通用检索模型 | ⚡⚡ 较快 | ⭐⭐⭐⭐ 强 |
| `bge-small` | BGE系列小模型 | ⚡⚡ 较快 | ⭐⭐⭐⭐ 强 |
| `e5-large` | E5最强版本 | ⚡ 慢 | ⭐⭐⭐⭐⭐ 最强 |

## 优化策略说明

### 策略1：换更强的嵌入模型 🔄
**效果**: Recall/MRR 预计提升 **10-20%**  
**成本**: 首次下载模型需要时间，推理速度略慢  
**建议**: 优先尝试 `e5-base-v2`

### 策略2：Turn-level 文档切分 ✂️
**效果**: Recall 预计提升 **15-30%**，MRR 提升 **10-20%**  
**原理**: 细粒度检索，每个邮件回合独立索引  
**成本**: 文档数量增加，索引构建时间增加  
**建议**: 强烈推荐

### 策略3：交叉编码器重排 🎯
**效果**: MRR 预计提升 **20-40%**，Recall 提升 **5-15%**  
**原理**: 在向量检索的Top100基础上，用更精确的模型重新排序  
**成本**: 每个查询需要额外计算，速度降低约2-3倍  
**建议**: 如果前两步还未达标再启用

## 性能目标 🎯

- **Recall@10 ≥ 0.80** (在前10个结果中找到相关文档)
- **MRR@10 ≥ 0.50** (相关文档平均排在前2名)

## 预期效果

| 配置 | 预期 Recall@10 | 预期 MRR@10 |
|------|---------------|-------------|
| Baseline | 0.40 - 0.55 | 0.25 - 0.35 |
| + Better Model | 0.50 - 0.65 | 0.30 - 0.42 |
| + Turn-level | 0.65 - 0.80 | 0.45 - 0.60 |
| + Reranking | 0.75 - 0.90 | 0.55 - 0.75 |

## 下一步：训练（如果需要）

如果以上优化后仍未达标 (Recall@10 < 0.80)，再进行微调训练：

```bash
# 使用训练脚本（需要 embedding.train.jsonl）
python train_embedding.py \
  --train-file Jupiter/data/working/embedding.train.jsonl \
  --base-model intfloat/e5-base-v2 \
  --output-dir ./models/e5-finetuned \
  --epochs 1 \
  --batch-size 16
```

训练完成后，将模型路径传给评估脚本：
```bash
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --model ./models/e5-finetuned \
  --turn-level \
  --rerank
```

## 故障排查

### 问题1: 找不到测试文件
```bash
# 检查文件是否存在
ls -lh output/*.jsonl

# 或使用你生成的文件
python eval_retrieval_enhanced.py --test output/threads_openai_20251110_1541.jsonl --k 10 --compare-all
```

### 问题2: 模型下载慢
```bash
# 使用国内镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 或者先手动下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"
```

### 问题3: 内存不足
```bash
# 使用更小的batch size（修改脚本中的 batch_size=256 → 64）
# 或使用更小的模型（minilm 或 bge-small）
python eval_retrieval_enhanced.py --test <file> --k 10 --model minilm --turn-level
```

## 输出示例

```
================================================================================
📊 FINAL COMPARISON TABLE
================================================================================
Configuration                                      Recall@10    MRR@10       Status        
--------------------------------------------------------------------------------
Baseline (MiniLM + Thread-level)                       0.450      0.280      ❌ Below Target
Better Model (E5-base + Thread-level)                  0.620      0.380      ❌ Below Target
Turn-level Split (MiniLM + Turn-level)                 0.720      0.520      ❌ Below Target
Model + Turn-level (E5-base + Turn-level)              0.820      0.640      ✅ TARGET MET
Full Optimization (E5-base + Turn-level + Rerank)      0.880      0.720      ✅ TARGET MET
================================================================================

🎯 Target: Recall@10 ≥ 0.80, MRR@10 ≥ 0.50

🏆 Best Configuration: Full Optimization (E5-base + Turn-level + Rerank)
   Recall@10: 0.880
   MRR@10: 0.720
================================================================================
```

## Tips 💡

1. **第一次运行会慢**：需要下载模型（约500MB-2GB）
2. **使用 --compare-all**：可以一次看到所有配置的效果对比
3. **先用小数据测试**：用少量数据（如5-10个threads）快速验证流程
4. **GPU加速**：如果有GPU，会自动使用，速度快10倍+

## 联系与支持

遇到问题可以查看脚本的详细输出，或者调整配置参数。祝你 Recall 拉满！🚀




