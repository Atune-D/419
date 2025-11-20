#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# eval_retrieval.py
# Usage: python eval_retrieval.py --test Jupiter/data/working/threads.test.jsonl --k 10

import argparse, jsonlines, re, numpy as np
from tqdm import tqdm
from pathlib import Path

def tokenize(s): return re.findall(r"[a-z0-9]+", (s or "").lower())

def last_by_role(turns, role):
    for t in reversed(turns):
        if t.get("role")==role: return t
    return None

def build_docs(threads):
    # 每个 thread 作为一个文档，内容= SCSA(若有)+前几条RAW拼接
    docs, ids = [], []
    for th in threads:
        scsa_texts = []
        for t in th["turns"]:
            sc = t.get("scsa")
            if isinstance(sc, str): scsa_texts.append(sc)
        scsa = "\n".join(scsa_texts)
        parts = []
        for t in th["turns"][:4]:
            parts.append(f"[{t.get('role','').upper()}] {t.get('subject','')}\n{t.get('body','')}")
        raw = "\n\n".join(parts)
        text = (scsa.strip() + "\n" + raw.strip()).strip()
        docs.append(text)
        ids.append(th["thread_id"])
    return ids, docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # 读取测试集
    test_threads = list(jsonlines.open(args.test))
    assert len(test_threads) > 0, "empty test set"

    # 准备文档
    doc_ids, doc_texts = build_docs(test_threads)

    # 向量模型
    from sentence_transformers import SentenceTransformer
    import faiss
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E_docs = model.encode(doc_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    dim = E_docs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(E_docs)

    # 评估
    K = args.k
    hit = 0
    rr_sum = 0.0
    total = 0

    for th in tqdm(test_threads, desc="Evaluating"):
        # 查询 = 最后一条客户邮件：优先SCSA，否则BODY
        q_turn = last_by_role(th["turns"], "customer")
        if not q_turn: continue
        q = q_turn.get("scsa") if isinstance(q_turn.get("scsa"), str) else q_turn.get("body","")
        if not q: continue

        vq = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(vq, K)
        top_ids = [doc_ids[i] for i in I[0]]

        total += 1
        # Recall@K
        if th["thread_id"] in top_ids:
            hit += 1
            # MRR@K
            rank = top_ids.index(th["thread_id"]) + 1
            rr_sum += 1.0 / rank

    recall_k = hit / total if total else 0.0
    mrr_k = rr_sum / total if total else 0.0
    print(f"Total queries: {total}")
    print(f"Recall@{K}: {recall_k:.3f}")
    print(f"MRR@{K}: {mrr_k:.3f}")

if __name__ == "__main__":
    main()
