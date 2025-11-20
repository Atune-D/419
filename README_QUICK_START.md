# 🎯 检索优化 - 快速开始

## 你现在有什么？

✅ **数据生成脚本**: `generateDate.py` - 已经成功生成了数据  
✅ **原始评估脚本**: `eval_retrieval.py` - 基础版本  
✅ **增强评估脚本**: `eval_retrieval_enhanced.py` - ⭐ 新的完整版  
✅ **数据验证脚本**: `verify_data.py` - 检查数据格式  
✅ **使用指南**: `RETRIEVAL_OPTIMIZATION_GUIDE.md` - 详细文档  

## 🚀 三种启动方式（选一个）

### 方式1: 一键运行（最简单）⭐

```bash
# 1. 给脚本执行权限
chmod +x run_optimization.sh

# 2. 运行（会自动对比所有配置）
./run_optimization.sh output/threads_openai_20251110_1541.jsonl
```

### 方式2: 手动运行完整对比

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证数据
python verify_data.py output/threads_openai_20251110_1541.jsonl

# 3. 运行对比
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --compare-all
```

### 方式3: 逐步测试单个配置

```bash
# 测试 baseline
python eval_retrieval_enhanced.py --test output/threads_openai_20251110_1541.jsonl --k 10

# 测试更强模型
python eval_retrieval_enhanced.py --test output/threads_openai_20251110_1541.jsonl --k 10 --model e5-base-v2

# 测试 turn-level
python eval_retrieval_enhanced.py --test output/threads_openai_20251110_1541.jsonl --k 10 --model e5-base-v2 --turn-level

# 测试全部优化
python eval_retrieval_enhanced.py --test output/threads_openai_20251110_1541.jsonl --k 10 --model e5-base-v2 --turn-level --rerank
```

## 📊 期望看到什么？

运行后会输出类似这样的对比表格：

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
```

## ⏱️ 预计运行时间

| 配置 | 首次（下载模型） | 后续运行 |
|------|-----------------|---------|
| 仅换模型 | 5-10分钟 | 30秒-2分钟 |
| Turn-level | 1-2分钟 | 30秒-1分钟 |
| 加重排 | 3-5分钟 | 1-3分钟 |
| 完整对比（5个配置） | 15-30分钟 | 5-10分钟 |

*注: 首次运行需要下载模型（约1-2GB），后续运行会直接使用缓存*

## 🎯 优化路径（按顺序尝试）

```
Step 1: Baseline
   ↓ (换模型)
Step 2: E5-base-v2 模型
   ↓ (Turn切分)  
Step 3: Turn-level 文档
   ↓ (加重排)
Step 4: Cross-encoder 重排
   ↓ (如果还不够)
Step 5: 微调训练（需要 train_embedding.py）
```

## 📦 依赖说明

新增的依赖（已在 requirements.txt）：
- `sentence-transformers` - 嵌入模型
- `faiss-cpu` - 向量检索
- `tqdm` - 进度条
- `numpy` - 数值计算

## 💡 实用技巧

1. **先用小数据测试**: 如果你的数据很大，可以先取前10-20条测试流程
   ```bash
   head -20 output/threads_openai_20251110_1541.jsonl > test_small.jsonl
   python eval_retrieval_enhanced.py --test test_small.jsonl --k 10 --compare-all
   ```

2. **GPU加速**: 如果你有GPU，脚本会自动使用，速度快10倍+

3. **查看详细输出**: 脚本会显示每个配置的详细进度

4. **保存结果**: 可以将输出重定向到文件
   ```bash
   ./run_optimization.sh output/threads_openai_20251110_1541.jsonl | tee results.txt
   ```

## 🔧 常见问题

### Q1: 模型下载太慢？
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: 内存不足？
使用更小的模型：
```bash
python eval_retrieval_enhanced.py --test <file> --k 10 --model minilm --turn-level
```

### Q3: 想用自己的数据？
只要是 JSONL 格式，包含 `thread_id` 和 `turns` 字段即可。先用 `verify_data.py` 检查格式。

### Q4: 达标后下一步？
如果 Recall@10 ≥ 0.80，恭喜！可以：
- 用这个配置构建生产系统
- 或继续优化其他指标
- 或添加更多高级功能（如混合检索）

### Q5: 未达标怎么办？
按顺序尝试：
1. 生成更多训练数据（用 `generateDate.py`）
2. 使用更强模型（`e5-large`）
3. 微调训练（需要 `train_embedding.py`）
4. 调整检索参数（K值、rerank候选数等）

## 📚 更多资源

- 详细指南: `RETRIEVAL_OPTIMIZATION_GUIDE.md`
- 数据生成: `generateDate.py --help`
- 评估脚本: `eval_retrieval_enhanced.py --help`

## 🎉 现在就开始！

```bash
# 确保在虚拟环境中
source venv/bin/activate

# 一键运行
./run_optimization.sh output/threads_openai_20251110_1541.jsonl

# 或者手动运行
python eval_retrieval_enhanced.py \
  --test output/threads_openai_20251110_1541.jsonl \
  --k 10 \
  --compare-all
```

祝你 Recall 拉满！🚀📈




