#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab 专用评估脚本 - 简化版

直接复制到 Colab 运行，或者在 Colab 中：
!wget https://your-url/eval_colab.py
!python eval_colab.py --test data/Testing.jsonl
"""

import argparse
import jsonlines
import re
import numpy as np
import time
from tqdm.auto import tqdm
import pandas as pd

def setup_colab():
    """检测是否在 Colab 环境并设置"""
    try:
        from google.colab import files
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("ℹ️  Running locally")
    return IN_COLAB

def load_threads(path):
    """加载 threads"""
    return list(jsonlines.open(path))

def tokenize(s):
    """简单分词"""
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def build_docs(threads, granularity="turn"):
    """构建检索文档"""
    ids, texts = [], []
    seen = set()
    
    if granularity == "thread":
        for th in threads:
            scsa = "\n".join([t.get("scsa","") for t in th.get("turns",[]) 
                             if isinstance(t.get("scsa"), str)])
            raw = "\n\n".join([f"[{t.get('role','').upper()}] {t.get('subject','')}\n{t.get('body','')}" 
                              for t in th.get("turns",[])[:4]])
            text = (scsa + "\n" + raw).strip()
            if text and text not in seen:
                ids.append(th["thread_id"])
                texts.append(text)
                seen.add(text)
    else:  # turn-level
        for th in threads:
            for i, t in enumerate(th.get("turns",[])):
                scsa = t.get("scsa","") if isinstance(t.get("scsa"), str) else ""
                text = (scsa + "\n" + f"[{t.get('role','').upper()}] {t.get('subject','')}\n{t.get('body','')}").strip()
                if text and text not in seen:
                    ids.append(f"{th['thread_id']}#t{i}")
                    texts.append(text)
                    seen.add(text)
    
    return ids, texts

def last_customer_query(th):
    """提取最后一条客户邮件"""
    for t in reversed(th.get("turns",[])):
        if t.get("role") == "customer":
            scsa = t.get("scsa")
            if isinstance(scsa, str) and scsa.strip():
                return scsa
            body = t.get("body","").strip()
            if body:
                return body
    return None

def extract_thread_id(doc_id):
    """从文档ID提取thread_id"""
    return doc_id.split("#")[0]

def run_experiment(test_threads, model_name, granularity, k, use_rerank=False):
    """运行单次实验"""
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    
    print(f"\n{'='*60}")
    print(f"🔬 {model_name.split('/')[-1]} | {granularity}-level")
    if use_rerank:
        print(f"   + Cross-encoder reranking")
    print(f"{'='*60}")
    
    # 构建文档
    doc_ids, doc_texts = build_docs(test_threads, granularity)
    print(f"📚 {len(doc_texts)} documents")
    
    # 加载模型
    print(f"📦 Loading embedding model...")
    emb = SentenceTransformer(model_name)
    
    # 编码
    print(f"🔨 Encoding documents...")
    E = emb.encode(
        doc_texts, 
        batch_size=128,  # Colab GPU 可以用更大的 batch
        normalize_embeddings=True, 
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    
    # FAISS 索引
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    
    # Reranker（可选）
    reranker = None
    if use_rerank:
        print(f"🎯 Loading reranker...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 评估
    total, hit, rr_sum = 0, 0, 0.0
    t0 = time.time()
    
    print(f"🔍 Evaluating...")
    for th in tqdm(test_threads, desc="Queries"):
        q = last_customer_query(th)
        if not q:
            continue
        
        # 向量检索
        vq = emb.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        retrieve_k = 100 if use_rerank else k
        D, I = index.search(vq, min(retrieve_k, len(doc_texts)))
        
        candidates = [(doc_ids[idx], doc_texts[idx]) for idx in I[0]]
        
        # 重排（可选）
        if reranker:
            pairs = [(q, text) for _, text in candidates[:100]]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(candidates[:100], scores), key=lambda x: x[1], reverse=True)
            top_ids = [doc_id for (doc_id, _), _ in reranked[:k]]
        else:
            top_ids = [doc_id for doc_id, _ in candidates[:k]]
        
        # 计算指标
        total += 1
        retrieved = [extract_thread_id(d) for d in top_ids]
        if th["thread_id"] in retrieved:
            hit += 1
            rank = retrieved.index(th["thread_id"]) + 1
            rr_sum += 1.0 / rank
    
    elapsed = time.time() - t0
    recall = hit / total if total > 0 else 0.0
    mrr = rr_sum / total if total > 0 else 0.0
    
    # 打印结果
    status = "✅ PASS" if recall >= 0.80 and mrr >= 0.50 else "❌ FAIL"
    print(f"\n{status} Results:")
    print(f"  Recall@{k}: {recall:.3f}")
    print(f"  MRR@{k}:    {mrr:.3f}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Queries:   {total}")
    
    return {
        "model": model_name,
        "granularity": granularity,
        "rerank": int(use_rerank),
        "k": k,
        "recall": round(recall, 3),
        "mrr": round(mrr, 3),
        "time": round(elapsed, 1),
        "queries": total,
    }

def main():
    parser = argparse.ArgumentParser(description="Colab-optimized retrieval evaluation")
    parser.add_argument("--test", required=True, help="Test JSONL file")
    parser.add_argument("--models", nargs="+", 
                       default=["sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-base-v2"],
                       help="Models to evaluate")
    parser.add_argument("--granularities", nargs="+", default=["thread", "turn"],
                       help="Document granularities")
    parser.add_argument("--k", type=int, default=10, help="K for Recall@K and MRR@K")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--out", default="results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # 检测 Colab
    IN_COLAB = setup_colab()
    
    # 加载数据
    print(f"\n📂 Loading test data: {args.test}")
    test_threads = load_threads(args.test)
    print(f"✅ Loaded {len(test_threads)} threads\n")
    
    # 运行实验
    results = []
    rerank_options = [False, True] if args.rerank else [False]
    total_exp = len(args.models) * len(args.granularities) * len(rerank_options)
    exp_num = 0
    
    print(f"🚀 Running {total_exp} experiments...\n")
    
    for model in args.models:
        for gran in args.granularities:
            for rerank in rerank_options:
                exp_num += 1
                print(f"\n{'#'*60}")
                print(f"Experiment {exp_num}/{total_exp}")
                print(f"{'#'*60}")
                
                try:
                    result = run_experiment(test_threads, model, gran, args.k, rerank)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error: {e}")
    
    # 生成报告
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(["recall", "mrr"], ascending=False)
        
        # 添加状态列
        df["status"] = df.apply(
            lambda r: "✅ PASS" if r["recall"] >= 0.80 and r["mrr"] >= 0.50 else "❌ FAIL",
            axis=1
        )
        df["model_short"] = df["model"].apply(lambda x: x.split("/")[-1][:30])
        
        # 打印表格
        print(f"\n{'='*80}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*80}")
        
        display_df = df[["model_short", "granularity", "rerank", "recall", "mrr", "time", "status"]]
        display_df.columns = ["Model", "Granularity", "Rerank", "Recall@10", "MRR@10", "Time(s)", "Status"]
        print(display_df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # 最佳配置
        best = df.iloc[0]
        print(f"🏆 BEST CONFIGURATION:")
        print(f"  Model:       {best['model']}")
        print(f"  Granularity: {best['granularity']}")
        print(f"  Rerank:      {'Yes' if best['rerank'] else 'No'}")
        print(f"  Recall@10:   {best['recall']:.3f}")
        print(f"  MRR@10:      {best['mrr']:.3f}")
        print(f"\n🎯 Target: Recall@10 ≥ 0.80, MRR@10 ≥ 0.50\n")
        
        # 保存结果
        df.to_csv(args.out, index=False)
        print(f"✅ Results saved to: {args.out}")
        
        # 如果在 Colab，自动下载
        if IN_COLAB:
            from google.colab import files
            print(f"📥 Downloading {args.out}...")
            files.download(args.out)
        
        print(f"\n🎉 Done!")

if __name__ == "__main__":
    main()


