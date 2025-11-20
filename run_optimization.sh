#!/bin/bash
# 一键运行检索优化实验

set -e

echo "🚀 检索优化实验 - 快速启动脚本"
echo "================================"
echo ""

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# 检查依赖
echo "📋 Checking dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# 设置数据文件（可修改）
DATA_FILE=${1:-"output/threads_openai_20251110_1541.jsonl"}

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    echo "Usage: ./run_optimization.sh <path-to-jsonl-file>"
    echo ""
    echo "Available files:"
    ls -lh output/*.jsonl 2>/dev/null || echo "  (no files in output/)"
    exit 1
fi

echo "📂 Using data file: $DATA_FILE"
echo ""

# 验证数据格式
echo "🔍 Step 1: Verifying data format..."
echo "-----------------------------------"
python verify_data.py "$DATA_FILE"
echo ""

# 运行对比评估
echo "🚀 Step 2: Running comprehensive comparison..."
echo "----------------------------------------------"
python eval_retrieval_enhanced.py --test "$DATA_FILE" --k 10 --compare-all

echo ""
echo "✅ Optimization experiment completed!"
echo ""
echo "💡 Tips:"
echo "  - If targets not met, consider training (see RETRIEVAL_OPTIMIZATION_GUIDE.md)"
echo "  - To test individual configs, see the guide for specific commands"
echo ""




