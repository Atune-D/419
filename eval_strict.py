#!/usr/bin/env python3
"""
更严格的检索评估：基于文本相似度而不是主题标签

评估思路:
1. 对每个查询，找到"最相似"的K个文档（ground truth）
2. 评估模型检索结果与 ground truth 的重叠度
3. 使用更细粒度的指标（MAP, NDCG）
"""

import jsonlines
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple

def load_threads(path):
    return list(jsonlines.open(path))

def extract_text(thread, granularity="thread"):
    """提取 thread 的文本表示"""
    if granularity == "thread":
        texts = []
        for t in thread.get("turns", [])[:4]:
            role = t.get('role', '').upper()
            subj = t.get('subject', '')
            body = t.get('body', '')
            texts.append(f"[{role}] {subj}\n{body}")
        return "\n\n".join(texts)
    else:  # last customer turn
        for t in reversed(thread.get("turns", [])):
            if t.get("role") == "customer":
                return t.get("body", "")
        return ""

def compute_ground_truth(query_threads, index_threads, model_name, top_k=10):
    """
    使用强模型计算"理想"的检索结果作为 ground truth
    
    Args:
        query_threads: 查询 threads
        index_threads: 索引库 threads  
        model_name: 用于计算相似度的模型（最好的模型）
        top_k: 每个查询保留前K个最相似文档
    
    Returns:
        ground_truth: {query_idx: [doc_idx1, doc_idx2, ...]}
    """
    print(f"\n📋 计算 Ground Truth（使用 {model_name.split('/')[-1]}）...")
    
    emb = SentenceTransformer(model_name)
    
    # 提取文本
    query_texts = [extract_text(th, "thread") for th in query_threads]
    index_texts = [extract_text(th, "thread") for th in index_threads]
    
    # 编码
    print("  编码查询...")
    query_embs = emb.encode(query_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    
    print("  编码索引...")
    index_embs = emb.encode(index_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    
    # 计算每个查询的最相似文档
    ground_truth = {}
    
    for i, query_emb in enumerate(tqdm(query_embs, desc="  计算相似度")):
        # 与所有文档的余弦相似度
        similarities = np.dot(index_embs, query_emb)
        
        # 排序，取前K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        ground_truth[i] = top_indices.tolist()
    
    return ground_truth

def evaluate_with_ground_truth(
    query_threads, 
    index_threads, 
    model_name, 
    ground_truth, 
    granularity="thread",
    k_values=[5, 10]
):
    """
    基于 ground truth 评估模型
    
    指标:
    - Recall@K: 前K个中有多少是在 ground truth 中的
    - Precision@K: 前K个中正确的比例
    - MAP@K: Mean Average Precision
    """
    print(f"\n{'='*70}")
    print(f"🔬 评估: {model_name.split('/')[-1]} | {granularity}")
    print(f"{'='*70}")
    
    emb = SentenceTransformer(model_name)
    
    # 构建索引
    index_texts = [extract_text(th, granularity) for th in index_threads]
    query_texts = [extract_text(th, "thread") for th in query_threads]
    
    print(f"📚 编码 {len(index_texts)} 文档...")
    index_embs = emb.encode(index_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    
    # 构建 FAISS
    index = faiss.IndexFlatIP(index_embs.shape[1])
    index.add(index_embs)
    
    # 评估
    print(f"🔍 评估 {len(query_texts)} 查询...")
    
    results = {k: {"recall": [], "precision": [], "ap": []} for k in k_values}
    
    for i, query_text in enumerate(tqdm(query_texts)):
        query_emb = emb.encode([query_text], normalize_embeddings=True)
        
        # 检索
        max_k = max(k_values)
        D, I = index.search(query_emb, max_k)
        retrieved = I[0].tolist()
        
        # Ground truth
        gt = set(ground_truth[i])
        
        # 计算指标
        for k in k_values:
            retrieved_k = retrieved[:k]
            hits = [idx for idx in retrieved_k if idx in gt]
            
            # Recall@K
            recall = len(hits) / len(gt) if len(gt) > 0 else 0
            results[k]["recall"].append(recall)
            
            # Precision@K
            precision = len(hits) / k
            results[k]["precision"].append(precision)
            
            # Average Precision@K
            if len(hits) > 0:
                ap = sum((i+1) / (retrieved_k.index(h) + 1) for i, h in enumerate(hits)) / len(gt)
                results[k]["ap"].append(ap)
            else:
                results[k]["ap"].append(0.0)
    
    # 汇总
    print(f"\n📊 结果:")
    summary = {}
    for k in k_values:
        recall = np.mean(results[k]["recall"])
        precision = np.mean(results[k]["precision"])
        map_score = np.mean(results[k]["ap"])
        
        print(f"\n  K={k}:")
        print(f"    Recall@{k}:    {recall:.3f}")
        print(f"    Precision@{k}: {precision:.3f}")
        print(f"    MAP@{k}:       {map_score:.3f}")
        
        summary[f"recall@{k}"] = round(recall, 3)
        summary[f"precision@{k}"] = round(precision, 3)
        summary[f"map@{k}"] = round(map_score, 3)
    
    return {
        "model": model_name,
        "granularity": granularity,
        **summary
    }

def main():
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="索引库 jsonl 文件（可以是多个文件的合并）")
    parser.add_argument("--query", required=True, help="查询集 jsonl 文件")
    parser.add_argument("--models", nargs="+", default=["intfloat/e5-base-v2"])
    parser.add_argument("--granularities", nargs="+", default=["thread", "turn"])
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--gt-model", default="intfloat/e5-base-v2", help="用于计算 ground truth 的模型")
    args = parser.parse_args()
    
    # 加载数据
    print("📂 加载数据...")
    index_threads = load_threads(args.index)
    query_threads = load_threads(args.query)
    
    print(f"✅ 索引库: {len(index_threads)} threads")
    print(f"✅ 查询集: {len(query_threads)} threads")
    
    # 计算 ground truth
    ground_truth = compute_ground_truth(
        query_threads, 
        index_threads, 
        args.gt_model, 
        top_k=max(args.k)
    )
    
    # 评估每个模型
    results = []
    
    for model_name in args.models:
        for granularity in args.granularities:
            result = evaluate_with_ground_truth(
                query_threads,
                index_threads,
                model_name,
                ground_truth,
                granularity,
                args.k
            )
            results.append(result)
    
    # 显示结果
    print("\n" + "="*70)
    print("📊 最终结果")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 保存
    df.to_csv("results_strict_eval.csv", index=False)
    print("\n✅ 结果已保存: results_strict_eval.csv")

if __name__ == "__main__":
    main()


