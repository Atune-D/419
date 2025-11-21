#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化实验结果

从 experiments.csv 生成：
- 对比表格（Markdown）
- 性能图表（PNG）
- 详细报告

Usage:
    python visualize_results.py --input report/experiments.csv --output report/
    python visualize_results.py --input report/experiments.csv --format png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(csv_path):
    """加载实验结果"""
    df = pd.read_csv(csv_path)
    return df

def create_markdown_table(df, output_path):
    """创建 Markdown 对比表格"""
    # 排序：按 Recall 降序
    df_sorted = df.sort_values(["recall", "mrr"], ascending=False)
    
    # 添加状态列
    df_sorted["status"] = df_sorted.apply(
        lambda row: "✅ PASS" if row["recall"] >= 0.80 and row["mrr"] >= 0.50 else "❌ FAIL",
        axis=1
    )
    
    # 简化模型名
    df_sorted["model_short"] = df_sorted["model"].apply(lambda x: x.split("/")[-1])
    
    # 生成 Markdown
    md = "# Experiment Results\n\n"
    md += f"## Summary\n\n"
    md += f"- **Total Experiments**: {len(df)}\n"
    md += f"- **Passed (Recall≥0.80, MRR≥0.50)**: {len(df[(df['recall']>=0.80) & (df['mrr']>=0.50)])}\n"
    md += f"- **Best Recall@10**: {df['recall'].max():.3f}\n"
    md += f"- **Best MRR@10**: {df['mrr'].max():.3f}\n\n"
    
    # 最佳配置
    best = df_sorted.iloc[0]
    md += f"## Best Configuration\n\n"
    md += f"- **Model**: `{best['model']}`\n"
    md += f"- **Granularity**: {best['granularity']}\n"
    md += f"- **BM25**: {'Yes' if best['bm25'] else 'No'}\n"
    md += f"- **Rerank**: {'Yes' if best['rerank'] else 'No'}\n"
    md += f"- **Recall@10**: {best['recall']:.3f}\n"
    md += f"- **MRR@10**: {best['mrr']:.3f}\n"
    md += f"- **Time**: {best['sec']:.1f}s\n\n"
    
    # 详细表格
    md += f"## Detailed Results\n\n"
    
    # 选择要显示的列
    display_cols = ["model_short", "granularity", "bm25", "rerank", "recall", "mrr", "sec", "status"]
    df_display = df_sorted[display_cols].copy()
    
    # 重命名列
    df_display.columns = ["Model", "Granularity", "BM25", "Rerank", "Recall@10", "MRR@10", "Time(s)", "Status"]
    
    md += df_display.to_markdown(index=False)
    md += "\n\n"
    
    # 消融研究
    md += f"## Ablation Study\n\n"
    md += "### Impact of Model Choice\n\n"
    
    model_impact = df.groupby("model")[["recall", "mrr"]].mean().sort_values("recall", ascending=False)
    md += model_impact.to_markdown()
    md += "\n\n"
    
    md += "### Impact of Granularity\n\n"
    gran_impact = df.groupby("granularity")[["recall", "mrr"]].mean()
    md += gran_impact.to_markdown()
    md += "\n\n"
    
    if df["bm25"].any():
        md += "### Impact of BM25\n\n"
        bm25_impact = df.groupby("bm25")[["recall", "mrr"]].mean()
        md += bm25_impact.to_markdown()
        md += "\n\n"
    
    if df["rerank"].any():
        md += "### Impact of Reranking\n\n"
        rerank_impact = df.groupby("rerank")[["recall", "mrr"]].mean()
        md += rerank_impact.to_markdown()
        md += "\n\n"
    
    # 保存
    with open(output_path, "w") as f:
        f.write(md)
    
    print(f"✅ Markdown table saved to: {output_path}")

def create_comparison_plot(df, output_path):
    """创建性能对比图"""
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 简化模型名
    df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1][:20])
    
    # 1. Recall vs MRR 散点图
    ax = axes[0, 0]
    for gran in df["granularity"].unique():
        df_gran = df[df["granularity"] == gran]
        ax.scatter(df_gran["recall"], df_gran["mrr"], 
                  label=f"{gran}-level", s=100, alpha=0.7)
    
    ax.axvline(x=0.80, color='r', linestyle='--', alpha=0.5, label="Target Recall≥0.80")
    ax.axhline(y=0.50, color='r', linestyle='--', alpha=0.5, label="Target MRR≥0.50")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("MRR@10")
    ax.set_title("Recall vs MRR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 模型对比柱状图
    ax = axes[0, 1]
    model_perf = df.groupby("model_short")[["recall", "mrr"]].mean().sort_values("recall", ascending=False)
    x = range(len(model_perf))
    width = 0.35
    ax.bar([i - width/2 for i in x], model_perf["recall"], width, label="Recall@10", alpha=0.8)
    ax.bar([i + width/2 for i in x], model_perf["mrr"], width, label="MRR@10", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_perf.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 粒度对比
    ax = axes[1, 0]
    gran_perf = df.groupby("granularity")[["recall", "mrr"]].mean()
    x = range(len(gran_perf))
    width = 0.35
    ax.bar([i - width/2 for i in x], gran_perf["recall"], width, label="Recall@10", alpha=0.8)
    ax.bar([i + width/2 for i in x], gran_perf["mrr"], width, label="MRR@10", alpha=0.8)
    ax.set_xlabel("Granularity")
    ax.set_ylabel("Score")
    ax.set_title("Granularity Impact")
    ax.set_xticks(x)
    ax.set_xticklabels(gran_perf.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 时间 vs 性能
    ax = axes[1, 1]
    scatter = ax.scatter(df["sec"], df["recall"], 
                        c=df["mrr"], cmap="viridis", 
                        s=100, alpha=0.7)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Efficiency vs Performance")
    plt.colorbar(scatter, ax=ax, label="MRR@10")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to: {output_path}")

def create_ablation_plot(df, output_path):
    """创建消融研究图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 模型影响
    ax = axes[0]
    model_impact = df.groupby("model")[["recall", "mrr"]].mean().sort_values("recall", ascending=False)
    model_impact["model_short"] = [x.split("/")[-1][:15] for x in model_impact.index]
    
    x = range(len(model_impact))
    width = 0.35
    ax.barh([i - width/2 for i in x], model_impact["recall"], width, label="Recall@10", alpha=0.8)
    ax.barh([i + width/2 for i in x], model_impact["mrr"], width, label="MRR@10", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(model_impact["model_short"])
    ax.set_xlabel("Score")
    ax.set_title("Model Impact")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. 粒度影响
    ax = axes[1]
    gran_impact = df.groupby("granularity")[["recall", "mrr"]].mean()
    x = range(len(gran_impact))
    width = 0.35
    ax.bar([i - width/2 for i in x], gran_impact["recall"], width, label="Recall@10", alpha=0.8)
    ax.bar([i + width/2 for i in x], gran_impact["mrr"], width, label="MRR@10", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(gran_impact.index)
    ax.set_ylabel("Score")
    ax.set_title("Granularity Impact")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. BM25 & Rerank 影响
    ax = axes[2]
    features = []
    recalls = []
    mrrs = []
    
    if df["bm25"].any():
        for val in [False, True]:
            subset = df[df["bm25"] == val]
            features.append(f"BM25={val}")
            recalls.append(subset["recall"].mean())
            mrrs.append(subset["mrr"].mean())
    
    if df["rerank"].any():
        for val in [False, True]:
            subset = df[df["rerank"] == val]
            features.append(f"Rerank={val}")
            recalls.append(subset["recall"].mean())
            mrrs.append(subset["mrr"].mean())
    
    if features:
        x = range(len(features))
        width = 0.35
        ax.bar([i - width/2 for i in x], recalls, width, label="Recall@10", alpha=0.8)
        ax.bar([i + width/2 for i in x], mrrs, width, label="MRR@10", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=15)
        ax.set_ylabel("Score")
        ax.set_title("Feature Impact")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Ablation plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--input", required=True, help="Input CSV file (experiments.csv)")
    parser.add_argument("--output", default="report/", help="Output directory")
    parser.add_argument("--format", default="all", choices=["png", "md", "all"],
                       help="Output format")
    args = parser.parse_args()
    
    # 加载数据
    print(f"📂 Loading results from: {args.input}")
    
    try:
        df = load_results(args.input)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    if df.empty:
        print("❌ No data found in CSV!")
        return
    
    print(f"✅ Loaded {len(df)} experiments\n")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成可视化
    if args.format in ["md", "all"]:
        md_path = output_dir / "experiments.md"
        print(f"📝 Generating Markdown table...")
        create_markdown_table(df, md_path)
    
    if args.format in ["png", "all"]:
        try:
            print(f"📊 Generating comparison plot...")
            comparison_path = output_dir / "comparison.png"
            create_comparison_plot(df, comparison_path)
            
            print(f"📊 Generating ablation plot...")
            ablation_path = output_dir / "ablation.png"
            create_ablation_plot(df, ablation_path)
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            print(f"   (matplotlib/seaborn may not be installed)")
    
    print(f"\n✅ Visualization complete!")
    print(f"   Output directory: {output_dir}")

if __name__ == "__main__":
    main()


