#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量实验矩阵评估 - 系统化对比不同配置

支持：
- 多个嵌入模型
- Thread-level vs Turn-level 粒度
- BM25 混合检索
- 交叉编码器重排

Usage:
    # 完整实验矩阵
    python eval_matrix.py \
      --test data/working/threads.test.jsonl \
      --models intfloat/e5-base-v2 BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 \
      --granularities thread turn \
      --k 10 --bm25 --rerank \
      --out report/experiments.csv
    
    # 快速测试
    python eval_matrix.py --test data/working/threads.test.jsonl --models intfloat/e5-base-v2
"""

import argparse
import jsonlines
import re
import numpy as np
import csv
import time
from pathlib import Path
from tqdm import tqdm

def tokenize(s):
    """简单分词"""
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def load_threads(path):
    """加载 threads"""
    return list(jsonlines.open(path))

def build_docs(threads, granularity="turn"):
    """
    构建检索文档
    
    Args:
        threads: list of thread dicts
        granularity: 'thread' or 'turn'
    
    Returns:
        (doc_ids, doc_texts) tuples
    """
    ids, texts = [], []
    seen_texts = set()  # 去重
    
    if granularity == "thread":
        # Thread-level: 每个thread一个文档
        for th in threads:
            # SCSA summary
            scsa_parts = []
            for t in th.get("turns", []):
                sc = t.get("scsa")
                if isinstance(sc, str) and sc.strip():
                    scsa_parts.append(sc)
            scsa = "\n".join(scsa_parts)
            
            # RAW content (前4个turns)
            raw_parts = []
            for t in th.get("turns", [])[:4]:
                role = t.get("role", "").upper()
                subj = t.get("subject", "")
                body = t.get("body", "")
                raw_parts.append(f"[{role}] {subj}\n{body}")
            raw = "\n\n".join(raw_parts)
            
            text = (scsa + "\n" + raw).strip()
            
            # 去重
            if text and text not in seen_texts:
                ids.append(th["thread_id"])
                texts.append(text)
                seen_texts.add(text)
    
    else:  # turn-level
        # Turn-level: 每个turn一个文档
        for th in threads:
            for i, t in enumerate(th.get("turns", [])):
                role = t.get("role", "").upper()
                subj = t.get("subject", "")
                body = t.get("body", "")
                scsa = t.get("scsa", "") if isinstance(t.get("scsa"), str) else ""
                
                text = (scsa + "\n" + f"[{role}] {subj}\n{body}").strip()
                
                # 去重
                if text and text not in seen_texts:
                    ids.append(f"{th['thread_id']}#t{i}")
                    texts.append(text)
                    seen_texts.add(text)
    
    return ids, texts

def last_customer_query(th):
    """提取最后一条客户邮件作为查询"""
    for t in reversed(th.get("turns", [])):
        if t.get("role") == "customer":
            # 优先SCSA，否则用body
            scsa = t.get("scsa")
            if isinstance(scsa, str) and scsa.strip():
                return scsa
            body = t.get("body", "")
            if body.strip():
                return body
    return None

def extract_thread_id(doc_id):
    """从文档ID提取thread_id"""
    return doc_id.split("#")[0]

def run_experiment(test_threads, model_name, granularity, k, use_bm25, use_rerank):
    """
    运行单次实验
    
    Returns:
        dict with metrics
    """
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # 构建文档
    doc_ids, doc_texts = build_docs(test_threads, granularity=granularity)
    
    if len(doc_texts) == 0:
        return {
            "model": model_name,
            "granularity": granularity,
            "bm25": int(use_bm25),
            "rerank": int(use_rerank),
            "k": k,
            "recall": 0.0,
            "mrr": 0.0,
            "sec": 0.0,
            "queries": 0,
            "docs": 0,
            "error": "No documents built"
        }
    
    # 加载嵌入模型
    print(f"  📦 Loading model: {model_name}")
    emb = SentenceTransformer(model_name)
    
    # 编码文档
    print(f"  🔨 Encoding {len(doc_texts)} documents...")
    E = emb.encode(
        doc_texts, 
        batch_size=256, 
        normalize_embeddings=True, 
        convert_to_numpy=True, 
        show_progress_bar=False
    )
    
    # 构建FAISS索引
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    
    # BM25 (可选)
    bm25 = None
    if use_bm25:
        try:
            from rank_bm25 import BM25Okapi
            print(f"  📊 Building BM25 index...")
            corpus = [tokenize(t) for t in doc_texts]
            bm25 = BM25Okapi(corpus)
        except ImportError:
            print(f"  ⚠️  rank_bm25 not installed, skipping BM25")
            use_bm25 = False
    
    # Reranker (可选)
    reranker = None
    if use_rerank:
        try:
            from sentence_transformers import CrossEncoder
            print(f"  🎯 Loading reranker...")
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"  ⚠️  Reranker load failed: {e}, skipping rerank")
            use_rerank = False
    
    # 评估
    total = 0
    hit = 0
    rr_sum = 0.0
    
    t0 = time.time()
    
    print(f"  🔍 Evaluating on {len(test_threads)} threads...")
    for th in tqdm(test_threads, desc="  Queries", leave=False):
        q = last_customer_query(th)
        if not q:
            continue
        
        # 编码查询
        vq = emb.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        
        # 向量检索
        retrieve_k = 100 if use_rerank else k
        D, I = index.search(vq, min(retrieve_k, len(doc_texts)))
        
        # 候选集
        candidates = []
        for j, idx in enumerate(I[0]):
            doc_id = doc_ids[idx]
            doc_text = doc_texts[idx]
            score = float(D[0][j])
            candidates.append((doc_id, doc_text, score))
        
        # BM25 融合 (可选)
        if bm25 and use_bm25:
            q_tokens = tokenize(q)
            bm25_scores = bm25.get_scores(q_tokens)
            # 归一化
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            
            # 融合: 70% vector + 30% BM25
            fused_candidates = []
            for doc_id, doc_text, vec_score in candidates:
                try:
                    # 找到对应的BM25分数
                    doc_idx = doc_ids.index(doc_id)
                    bm25_score = float(bm25_scores[doc_idx])
                    fused_score = 0.7 * vec_score + 0.3 * bm25_score
                    fused_candidates.append((doc_id, doc_text, fused_score))
                except (ValueError, IndexError):
                    fused_candidates.append((doc_id, doc_text, vec_score))
            
            candidates = sorted(fused_candidates, key=lambda x: x[2], reverse=True)
        
        # 重排 (可选)
        if reranker and use_rerank:
            pairs = [(q, text) for _, text, _ in candidates[:100]]
            rerank_scores = reranker.predict(pairs)
            reranked = sorted(
                zip(candidates[:100], rerank_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            top_ids = [doc_id for (doc_id, _, _), _ in reranked[:k]]
        else:
            top_ids = [doc_id for doc_id, _, _ in candidates[:k]]
        
        # 评估
        total += 1
        gt = th["thread_id"]
        
        # 提取thread_ids（处理turn-level的情况）
        retrieved_thread_ids = [extract_thread_id(doc_id) for doc_id in top_ids]
        
        if gt in retrieved_thread_ids:
            hit += 1
            # MRR: 第一次出现的位置
            rank = retrieved_thread_ids.index(gt) + 1
            rr_sum += 1.0 / rank
    
    elapsed = time.time() - t0
    
    recall_k = hit / total if total > 0 else 0.0
    mrr_k = rr_sum / total if total > 0 else 0.0
    
    return {
        "model": model_name,
        "granularity": granularity,
        "bm25": int(use_bm25),
        "rerank": int(use_rerank),
        "k": k,
        "recall": round(recall_k, 3),
        "mrr": round(mrr_k, 3),
        "sec": round(elapsed, 1),
        "queries": total,
        "docs": len(doc_texts),
    }

def main():
    parser = argparse.ArgumentParser(description="Batch retrieval evaluation with experiment matrix")
    parser.add_argument("--test", required=True, help="Test JSONL file")
    parser.add_argument("--models", nargs="+", required=True, 
                       help="Embedding models (e.g., intfloat/e5-base-v2)")
    parser.add_argument("--granularities", nargs="+", default=["thread", "turn"],
                       help="Document granularity: thread, turn")
    parser.add_argument("--k", type=int, default=10, help="K for Recall@K and MRR@K")
    parser.add_argument("--bm25", action="store_true", help="Enable BM25 hybrid retrieval")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--out", default="report/experiments.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # 加载测试集
    print(f"\n{'='*70}")
    print(f"📊 BATCH EVALUATION - Experiment Matrix")
    print(f"{'='*70}")
    print(f"Test file:     {args.test}")
    print(f"Models:        {', '.join(args.models)}")
    print(f"Granularities: {', '.join(args.granularities)}")
    print(f"K:             {args.k}")
    print(f"BM25:          {args.bm25}")
    print(f"Rerank:        {args.rerank}")
    print(f"Output:        {args.out}")
    print(f"{'='*70}\n")
    
    test_threads = load_threads(args.test)
    print(f"✅ Loaded {len(test_threads)} test threads\n")
    
    # 运行实验矩阵
    results = []
    total_experiments = len(args.models) * len(args.granularities)
    
    for i, model_name in enumerate(args.models, 1):
        for j, gran in enumerate(args.granularities, 1):
            exp_num = (i - 1) * len(args.granularities) + j
            print(f"\n{'#'*70}")
            print(f"Experiment {exp_num}/{total_experiments}: {model_name} | {gran}")
            print(f"{'#'*70}")
            
            try:
                result = run_experiment(
                    test_threads, 
                    model_name, 
                    gran, 
                    args.k, 
                    args.bm25, 
                    args.rerank
                )
                results.append(result)
                
                # 打印结果
                status = "✅" if result["recall"] >= 0.80 and result["mrr"] >= 0.50 else "❌"
                print(f"\n  {status} Results:")
                print(f"     Recall@{args.k}: {result['recall']:.3f}")
                print(f"     MRR@{args.k}:    {result['mrr']:.3f}")
                print(f"     Time:      {result['sec']:.1f}s")
                print(f"     Queries:   {result['queries']}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    "model": model_name,
                    "granularity": gran,
                    "bm25": int(args.bm25),
                    "rerank": int(args.rerank),
                    "k": args.k,
                    "recall": 0.0,
                    "mrr": 0.0,
                    "sec": 0.0,
                    "queries": 0,
                    "docs": 0,
                    "error": str(e)
                })
    
    # 保存结果
    if results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*70}")
        print(f"✅ Results saved to: {args.out}")
        print(f"{'='*70}\n")
        
        # 打印汇总表格
        print(f"📊 SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Gran':<8} {'Recall':>8} {'MRR':>8} {'Status':>10}")
        print(f"{'-'*70}")
        
        for r in sorted(results, key=lambda x: (x["recall"], x["mrr"]), reverse=True):
            status = "✅ PASS" if r["recall"] >= 0.80 and r["mrr"] >= 0.50 else "❌ FAIL"
            model_short = r["model"].split("/")[-1][:28]
            print(f"{model_short:<30} {r['granularity']:<8} {r['recall']:>8.3f} {r['mrr']:>8.3f} {status:>10}")
        
        print(f"{'='*70}\n")
        
        # 最佳配置
        best = max(results, key=lambda x: (x["recall"], x["mrr"]))
        print(f"🏆 Best Configuration:")
        print(f"   Model:       {best['model']}")
        print(f"   Granularity: {best['granularity']}")
        print(f"   BM25:        {bool(best['bm25'])}")
        print(f"   Rerank:      {bool(best['rerank'])}")
        print(f"   Recall@{args.k}:   {best['recall']:.3f}")
        print(f"   MRR@{args.k}:      {best['mrr']:.3f}")
        print(f"\n🎯 Target: Recall@10 ≥ 0.80, MRR@10 ≥ 0.50\n")
        
        # 下一步建议
        if best["recall"] < 0.80:
            print(f"💡 Suggestions:")
            print(f"   1. Generate more training data (current: ~{len(test_threads)*5} total)")
            print(f"   2. Try fine-tuning: python train_embedding.py")
            print(f"   3. Use larger model: intfloat/e5-large-v2")

if __name__ == "__main__":
    main()


