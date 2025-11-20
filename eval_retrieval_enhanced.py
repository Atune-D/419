#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版检索评估脚本 - 支持多种优化策略

⚠️  IMPORTANT: 为避免数据泄露，请使用 --train 提供独立的训练集用于构建索引！

Usage examples:
  # ✅ 正确用法：分别提供训练集和测试集
  python eval_retrieval_enhanced.py \
    --train threads.train.jsonl \
    --test threads.test.jsonl \
    --k 10 --compare-all
  
  # ❌ 不推荐：仅用测试集（数据泄露，结果虚高）
  python eval_retrieval_enhanced.py --test threads.test.jsonl --k 10
  
  # 使用更强模型
  python eval_retrieval_enhanced.py \
    --train threads.train.jsonl --test threads.test.jsonl \
    --k 10 --model e5-base-v2
  
  # Turn级切分
  python eval_retrieval_enhanced.py \
    --train threads.train.jsonl --test threads.test.jsonl \
    --k 10 --turn-level
  
  # 加重排
  python eval_retrieval_enhanced.py \
    --train threads.train.jsonl --test threads.test.jsonl \
    --k 10 --rerank
  
  # 全开
  python eval_retrieval_enhanced.py \
    --train threads.train.jsonl --test threads.test.jsonl \
    --k 10 --model e5-base-v2 --turn-level --rerank
"""

import argparse, jsonlines, re, numpy as np
from tqdm import tqdm
from pathlib import Path
import time

def tokenize(s): 
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def last_by_role(turns, role):
    for t in reversed(turns):
        if t.get("role")==role: 
            return t
    return None

def build_docs_thread_level(threads):
    """原版：每个 thread 作为一个文档"""
    docs, ids = [], []
    for th in threads:
        scsa_texts = []
        for t in th["turns"]:
            sc = t.get("scsa")
            if isinstance(sc, str): 
                scsa_texts.append(sc)
        scsa = "\n".join(scsa_texts)
        
        parts = []
        for t in th["turns"][:4]:
            parts.append(f"[{t.get('role','').upper()}] {t.get('subject','')}\n{t.get('body','')}")
        raw = "\n\n".join(parts)
        text = (scsa.strip() + "\n" + raw.strip()).strip()
        docs.append(text)
        ids.append(th["thread_id"])
    return ids, docs

def build_docs_turn_level(threads):
    """优化：每个 turn 作为一个文档（更细粒度检索）"""
    docs, ids = [], []
    for th in threads:
        for i, t in enumerate(th["turns"]):
            role = t.get("role", "")
            subj = t.get("subject", "") or ""
            body = t.get("body", "") or ""
            scsa = t.get("scsa") if isinstance(t.get("scsa"), str) else ""
            
            # 每个 turn 做成一个 doc（SCSA 优先 + RAW）
            text = (scsa + "\n" + f"[{role.upper()}] {subj}\n{body}").strip()
            docs.append(text)
            ids.append(f"{th['thread_id']}#t{i}")  # 保留 turn 位置信息
    return ids, docs

def extract_thread_id(doc_id):
    """从文档ID中提取thread_id（处理 turn-level 的情况）"""
    if "#t" in doc_id:
        return doc_id.split("#t")[0]
    return doc_id

def load_embedding_model(model_name):
    """加载嵌入模型"""
    from sentence_transformers import SentenceTransformer
    
    model_map = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 原版（快但弱）
        "e5-base-v2": "intfloat/e5-base-v2",                 # 更强的通用检索
        "bge-small": "BAAI/bge-small-en-v1.5",               # BGE 系列
        "e5-large": "intfloat/e5-large-v2",                  # 最强（但慢）
    }
    
    model_path = model_map.get(model_name, model_name)
    print(f"📦 Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path)
    return model

def rerank_with_cross_encoder(query, candidates, top_k=10):
    """使用交叉编码器重排"""
    from sentence_transformers import CrossEncoder
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 准备 (query, doc) 对
    pairs = [(query, doc_text) for doc_id, doc_text in candidates]
    
    # 打分并重排
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # 返回 top_k 个文档ID
    top_ids = [doc_id for (doc_id, _text), _score in reranked[:top_k]]
    return top_ids

def evaluate(test_threads, doc_ids, doc_texts, model, k=10, use_rerank=False):
    """执行检索评估"""
    import faiss
    
    # 构建 FAISS 索引
    print("🔨 Building FAISS index...")
    E_docs = model.encode(
        doc_texts, 
        batch_size=256, 
        normalize_embeddings=True, 
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    dim = E_docs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(E_docs)
    
    # 评估指标
    K = k
    hit = 0
    rr_sum = 0.0
    total = 0
    
    # 如果使用重排，先检索更多候选
    retrieve_k = 100 if use_rerank else K
    
    print(f"🔍 Evaluating on {len(test_threads)} test threads...")
    for th in tqdm(test_threads, desc="Evaluating"):
        # 查询 = 最后一条客户邮件：优先SCSA，否则BODY
        q_turn = last_by_role(th["turns"], "customer")
        if not q_turn: 
            continue
        
        q = q_turn.get("scsa") if isinstance(q_turn.get("scsa"), str) else q_turn.get("body", "")
        if not q: 
            continue
        
        # 向量检索
        vq = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(vq, retrieve_k)
        
        # 是否使用重排
        if use_rerank:
            candidates = [(doc_ids[i], doc_texts[i]) for i in I[0]]
            top_ids = rerank_with_cross_encoder(q, candidates, top_k=K)
        else:
            top_ids = [doc_ids[i] for i in I[0][:K]]
        
        # 提取 thread_id（处理 turn-level 的情况）
        top_thread_ids = [extract_thread_id(doc_id) for doc_id in top_ids]
        
        total += 1
        # Recall@K
        if th["thread_id"] in top_thread_ids:
            hit += 1
            # MRR@K
            rank = top_thread_ids.index(th["thread_id"]) + 1
            rr_sum += 1.0 / rank
    
    recall_k = hit / total if total else 0.0
    mrr_k = rr_sum / total if total else 0.0
    
    return {
        "total": total,
        "recall": recall_k,
        "mrr": mrr_k,
        "k": K
    }

def run_evaluation(test_path, model_name="minilm", k=10, turn_level=False, use_rerank=False, train_path=None):
    """运行单次评估
    
    Args:
        test_path: 测试集路径（用于查询）
        train_path: 训练集路径（用于构建索引）。如果为None，则用测试集构建索引（⚠️ 数据泄露！）
    """
    # 读取测试集
    print(f"\n{'='*60}")
    print(f"📊 Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Doc Level: {'Turn' if turn_level else 'Thread'}")
    print(f"  Reranking: {'Yes' if use_rerank else 'No'}")
    print(f"  K: {k}")
    print(f"{'='*60}\n")
    
    test_threads = list(jsonlines.open(test_path))
    assert len(test_threads) > 0, "empty test set"
    
    # 决定用什么数据构建索引
    if train_path:
        print(f"📚 Loading train set for index: {train_path}")
        index_threads = list(jsonlines.open(train_path))
        print(f"   Index: {len(index_threads)} threads")
        print(f"   Query: {len(test_threads)} threads")
    else:
        print(f"⚠️  WARNING: No train set provided!")
        print(f"   Using test set for BOTH index and queries (DATA LEAKAGE!)")
        print(f"   Results will be artificially high and NOT reliable!")
        print(f"   Use --train flag to provide separate training data.\n")
        index_threads = test_threads
    
    # 构建文档（用于索引）
    if turn_level:
        doc_ids, doc_texts = build_docs_turn_level(index_threads)
    else:
        doc_ids, doc_texts = build_docs_thread_level(index_threads)
    
    print(f"🔨 Building index with {len(doc_texts)} documents from {len(index_threads)} threads")
    
    # 加载模型
    model = load_embedding_model(model_name)
    
    # 评估
    start_time = time.time()
    results = evaluate(test_threads, doc_ids, doc_texts, model, k=k, use_rerank=use_rerank)
    elapsed = time.time() - start_time
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"📈 Results:")
    print(f"  Total queries: {results['total']}")
    print(f"  Recall@{results['k']}: {results['recall']:.3f}")
    print(f"  MRR@{results['k']}: {results['mrr']:.3f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    
    return results

def compare_all_configs(test_path, k=10, train_path=None):
    """对比所有配置组合"""
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE COMPARISON - Running all configurations...")
    print("="*70)
    
    if not train_path:
        print("\n⚠️  " + "="*66)
        print("⚠️  WARNING: No separate train set! Using test set for indexing too!")
        print("⚠️  This causes DATA LEAKAGE - results will be artificially high!")  
        print("⚠️  " + "="*66 + "\n")
    
    configs = [
        # (model_name, turn_level, use_rerank, description)
        ("minilm", False, False, "Baseline (MiniLM + Thread-level)"),
        ("e5-base-v2", False, False, "Better Model (E5-base + Thread-level)"),
        ("minilm", True, False, "Turn-level Split (MiniLM + Turn-level)"),
        ("e5-base-v2", True, False, "Model + Turn-level (E5-base + Turn-level)"),
        ("e5-base-v2", True, True, "Full Optimization (E5-base + Turn-level + Rerank)"),
    ]
    
    results_table = []
    
    for i, (model_name, turn_level, use_rerank, desc) in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"Config {i}/{len(configs)}: {desc}")
        print(f"{'#'*70}")
        
        try:
            result = run_evaluation(test_path, model_name, k, turn_level, use_rerank, train_path)
            results_table.append({
                "config": desc,
                "model": model_name,
                "turn_level": turn_level,
                "rerank": use_rerank,
                "recall": result["recall"],
                "mrr": result["mrr"]
            })
        except Exception as e:
            print(f"❌ Error in config {i}: {e}")
            results_table.append({
                "config": desc,
                "model": model_name,
                "turn_level": turn_level,
                "rerank": use_rerank,
                "recall": 0.0,
                "mrr": 0.0
            })
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("📊 FINAL COMPARISON TABLE")
    print("="*100)
    print(f"{'Configuration':<50} {'Recall@10':<12} {'MRR@10':<12} {'Status':<15}")
    print("-"*100)
    
    for r in results_table:
        status = "✅ TARGET MET" if r["recall"] >= 0.80 and r["mrr"] >= 0.50 else "❌ Below Target"
        print(f"{r['config']:<50} {r['recall']:>10.3f}  {r['mrr']:>10.3f}  {status:<15}")
    
    print("="*100)
    print("\n🎯 Target: Recall@10 ≥ 0.80, MRR@10 ≥ 0.50")
    
    # 找出最佳配置
    best = max(results_table, key=lambda x: (x["recall"], x["mrr"]))
    print(f"\n🏆 Best Configuration: {best['config']}")
    print(f"   Recall@10: {best['recall']:.3f}")
    print(f"   MRR@10: {best['mrr']:.3f}")
    print("="*100 + "\n")

def main():
    ap = argparse.ArgumentParser(description="Enhanced retrieval evaluation with multiple optimization strategies")
    ap.add_argument("--test", required=True, help="Path to test JSONL file (for queries)")
    ap.add_argument("--train", default=None, help="Path to train JSONL file (for building index). If not provided, uses test set (⚠️ DATA LEAKAGE!)")
    ap.add_argument("--k", type=int, default=10, help="K for Recall@K and MRR@K")
    ap.add_argument("--model", default="minilm", 
                    choices=["minilm", "e5-base-v2", "bge-small", "e5-large"],
                    help="Embedding model to use")
    ap.add_argument("--turn-level", action="store_true", 
                    help="Use turn-level documents instead of thread-level")
    ap.add_argument("--rerank", action="store_true", 
                    help="Use cross-encoder reranking")
    ap.add_argument("--compare-all", action="store_true", 
                    help="Run comparison across all configurations")
    args = ap.parse_args()
    
    if args.compare_all:
        compare_all_configs(args.test, k=args.k, train_path=args.train)
    else:
        run_evaluation(args.test, args.model, args.k, args.turn_level, args.rerank, args.train)

if __name__ == "__main__":
    main()

