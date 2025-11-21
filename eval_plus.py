#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多维度评估脚本 - 适配版
- Strict Recall@K / MRR@K
- NDCG@K（分级相关性）
- Entity@K（实体匹配率）
- Macro/Micro 统计
"""

import argparse, math, jsonlines, re, csv, time, numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# ---------- utils ----------
def tokenize(s): 
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def last_customer_turn(th):
    for t in reversed(th["turns"]):
        if t.get("role") == "customer":
            return t
    return None

def extract_entities_from_scsa(scsa: str):
    """
    从 SCSA 提取实体
    格式示例：
    **Intent:** ...
    **Key Entities:** Order ID: #695367, Product: Sneakers
    **Requested Action:** ...
    """
    s = (scsa or "").lower()
    ents = {}
    
    # 提取 Intent (动作)
    m = re.search(r"\*\*intent:\*\*\s*([^\n\*]+)", s)
    if m:
        intent = m.group(1).strip()
        # 简化为主要动作
        if "refund" in intent or "return" in intent:
            ents["action"] = "refund"
        elif "exchange" in intent or "replace" in intent:
            ents["action"] = "exchange"
        elif "invoice" in intent:
            ents["action"] = "invoice"
        elif "track" in intent or "shipping" in intent or "shipment" in intent or "delivery" in intent:
            ents["action"] = "shipping"
        elif "quote" in intent or "wholesale" in intent:
            ents["action"] = "quote"
    
    # 提取 Key Entities
    m = re.search(r"\*\*key entities:\*\*\s*([^\n\*]+)", s)
    if m:
        entities_str = m.group(1).strip()
        
        # Order ID
        order_match = re.search(r"order\s*id:\s*[#]?(\d+)", entities_str)
        if order_match:
            ents["order_id"] = order_match.group(1)
        
        # Product
        product_match = re.search(r"product:\s*([^,\n]+?)(?:,|$)", entities_str)
        if product_match:
            product = product_match.group(1).strip()
            if product.lower() not in ["not specified", "n/a", "none"]:
                ents["product"] = product
    
    return ents

def overlap_key_entities(e1, e2):
    """检查两个实体字典是否有重叠"""
    keys = ["order_id", "product", "action"]
    
    for k in keys:
        if k in e1 and k in e2 and e1[k] and e2[k]:
            if k == "product":
                # Product 用宽松匹配
                p1, p2 = e1[k].lower(), e2[k].lower()
                if len(p1) >= 4 and len(p2) >= 4:
                    if p1 in p2 or p2 in p1:
                        return True
            elif k == "order_id":
                # Order ID 前4位匹配即可
                if len(e1[k]) >= 4 and len(e2[k]) >= 4:
                    if e1[k][:4] == e2[k][:4]:
                        return True
            else:
                # Action 精确匹配
                if e1[k] == e2[k]:
                    return True
    
    return False

def ndcg_at_k(gains, k=10):
    """计算 NDCG@K"""
    dcg = 0.0
    for i, g in enumerate(gains[:k]):
        dcg += (2**g - 1) / math.log2(i + 2)
    
    ideal = sorted(gains, reverse=True)
    idcg = 0.0
    for i, g in enumerate(ideal[:k]):
        idcg += (2**g - 1) / math.log2(i + 2)
    
    return dcg / (idcg + 1e-9)

# ---------- build corpus ----------
def build_docs(threads, granularity="turn", scsa_first=True, max_turns=4):
    """构建文档库"""
    ids, texts, metas = [], [], []
    
    if granularity == "thread":
        for th in threads:
            # 合并所有 SCSA
            scsa_parts = [t.get("scsa", "") for t in th["turns"] if isinstance(t.get("scsa"), str)]
            scsa = "\n".join(scsa_parts)
            
            # 合并前几个 turns
            raw_parts = []
            for t in th["turns"][:max_turns]:
                role = t.get('role', '').upper()
                subj = t.get('subject', '')
                body = t.get('body', '')
                raw_parts.append(f"[{role}] {subj}\n{body}")
            raw = "\n\n".join(raw_parts)
            
            doc = (scsa + "\n" + raw).strip() if scsa_first else (raw + "\n" + scsa).strip()
            
            ids.append(th["thread_id"])
            texts.append(doc)
            metas.append({
                "topic": th.get("topic", "other"),
                "entities": extract_entities_from_scsa(scsa)
            })
    
    else:  # turn-level
        for th in threads:
            for i, t in enumerate(th["turns"][:max_turns]):
                scsa = t.get("scsa", "") if isinstance(t.get("scsa"), str) else ""
                role = t.get('role', '').upper()
                subj = t.get('subject', '')
                body = t.get('body', '')
                
                parts = f"[{role}] {subj}\n{body}"
                doc = (scsa + "\n" + parts).strip() if scsa_first else (parts + "\n" + scsa).strip()
                
                ids.append(f"{th['thread_id']}#t{i}")
                texts.append(doc)
                metas.append({
                    "topic": th.get("topic", "other"),
                    "entities": extract_entities_from_scsa(scsa)
                })
    
    return ids, texts, metas

# ---------- evaluation ----------
def evaluate(test_threads, emb_model_name, granularity, k, use_bm25, use_rerank, candidates=100):
    """
    多维度评估
    
    Returns:
        micro_stats: 总体统计
        topic_rows: 按 topic 的统计
        macro: macro 平均
    """
    from rank_bm25 import BM25Okapi
    
    print(f"\n{'='*70}")
    print(f"🔬 评估: {emb_model_name.split('/')[-1]} | {granularity}")
    print(f"{'='*70}")
    
    # 构建文档库
    print("📚 构建文档库...")
    doc_ids, doc_texts, doc_meta = build_docs(test_threads, granularity=granularity)
    print(f"   文档数: {len(doc_texts)}")
    
    # 加载模型
    print(f"📦 加载模型...")
    emb = SentenceTransformer(emb_model_name)
    
    # 编码
    print(f"🔨 编码文档...")
    E = emb.encode(doc_texts, batch_size=128, normalize_embeddings=True, 
                   convert_to_numpy=True, show_progress_bar=True)
    
    # 构建索引
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    
    # BM25
    bm25 = None
    if use_bm25:
        print("📝 构建 BM25...")
        corp = [tokenize(t) for t in doc_texts]
        bm25 = BM25Okapi(corp)
    
    # Reranker
    reranker = None
    if use_rerank:
        print("🎯 加载 Reranker...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 统计
    micro = {"strict_hit": 0, "rr_sum": 0.0, "ndcg_sum": 0.0, "entity_sum": 0.0, "total": 0}
    by_topic = defaultdict(lambda: {"strict_hit": 0, "rr_sum": 0.0, "ndcg_sum": 0.0, "entity_sum": 0.0, "total": 0})
    
    print(f"🔍 评估中...")
    t0 = time.time()
    
    from tqdm.auto import tqdm
    for th in tqdm(test_threads, desc="Queries"):
        q_turn = last_customer_turn(th)
        if not q_turn: 
            continue
        
        # 提取查询文本
        q_text = q_turn.get("scsa") if isinstance(q_turn.get("scsa"), str) and q_turn.get("scsa") else q_turn.get("body", "")
        if not q_text:
            continue
        
        # Gold标准
        gold_tid = th["thread_id"]
        gold_topic = th.get("topic", "other")
        q_entities = extract_entities_from_scsa(q_turn.get("scsa") or "")
        
        # 向量检索
        vq = emb.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(vq, candidates if reranker else k)
        cand = [(doc_ids[i], doc_texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
        
        # BM25 融合
        if bm25 is not None:
            bs = bm25.get_scores(tokenize(q_text))
            bs = (bs - bs.min()) / (bs.ptp() + 1e-9)  # 归一化
            
            fused = []
            for cid, txt, sim in cand:
                idx = doc_ids.index(cid)
                fused.append((cid, txt, 0.7 * sim + 0.3 * float(bs[idx])))
            cand = sorted(fused, key=lambda x: x[2], reverse=True)
        
        # Rerank
        if reranker:
            pairs = [(q_text, c[1]) for c in cand[:candidates]]
            scores = reranker.predict(pairs)
            order = np.argsort(-scores)[:k]
            top_ids = [cand[i][0] for i in order]
            top_texts = [cand[i][1] for i in order]
        else:
            top_ids = [cid for cid, _t, _s in cand[:k]]
            top_texts = [txt for _cid, txt, _s in cand[:k]]
        
        # 计算分级相关性
        gains = []
        ent_matches = 0
        
        for cid in top_ids:
            tid = cid.split("#")[0]
            meta_idx = doc_ids.index(cid)
            m = doc_meta[meta_idx]
            
            if tid == gold_tid:
                gains.append(3)  # 完全匹配
            elif (m["topic"] == gold_topic) and overlap_key_entities(q_entities, m["entities"]):
                gains.append(2)  # 同意图+实体匹配
                ent_matches += 1
            elif m["topic"] == gold_topic:
                gains.append(1)  # 只有意图匹配
            else:
                gains.append(0)  # 不相关
        
        # Strict Recall / MRR
        strict_hit = 0
        rr = 0.0
        top_tids = [x.split("#")[0] for x in top_ids]
        if gold_tid in top_tids:
            strict_hit = 1
            rr = 1.0 / (top_tids.index(gold_tid) + 1)
        
        # NDCG
        ndcg = ndcg_at_k(gains, k=k)
        
        # Entity@K
        entity_k = ent_matches / float(k)
        
        # 更新统计
        micro["strict_hit"] += strict_hit
        micro["rr_sum"] += rr
        micro["ndcg_sum"] += ndcg
        micro["entity_sum"] += entity_k
        micro["total"] += 1
        
        # 按 topic 统计
        b = by_topic[gold_topic]
        b["strict_hit"] += strict_hit
        b["rr_sum"] += rr
        b["ndcg_sum"] += ndcg
        b["entity_sum"] += entity_k
        b["total"] += 1
    
    elapsed = time.time() - t0
    
    # 汇总 Micro
    micro_stats = {
        "recall": round(micro["strict_hit"] / max(1, micro["total"]), 3),
        "mrr": round(micro["rr_sum"] / max(1, micro["total"]), 3),
        "ndcg": round(micro["ndcg_sum"] / max(1, micro["total"]), 3),
        "entity": round(micro["entity_sum"] / max(1, micro["total"]), 3),
        "queries": micro["total"],
        "time_sec": round(elapsed, 1),
    }
    
    # 按 topic 汇总
    topic_rows = []
    for tpc, b in by_topic.items():
        topic_rows.append({
            "topic": tpc,
            "recall": round(b["strict_hit"] / max(1, b["total"]), 3),
            "mrr": round(b["rr_sum"] / max(1, b["total"]), 3),
            "ndcg": round(b["ndcg_sum"] / max(1, b["total"]), 3),
            "entity": round(b["entity_sum"] / max(1, b["total"]), 3),
            "queries": b["total"],
        })
    
    # Macro 平均
    if topic_rows:
        macro = {
            "recall": round(sum(r["recall"] for r in topic_rows) / len(topic_rows), 3),
            "mrr": round(sum(r["mrr"] for r in topic_rows) / len(topic_rows), 3),
            "ndcg": round(sum(r["ndcg"] for r in topic_rows) / len(topic_rows), 3),
            "entity": round(sum(r["entity"] for r in topic_rows) / len(topic_rows), 3),
        }
    else:
        macro = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "entity": 0.0}
    
    # 显示结果
    print(f"\n📊 结果:")
    print(f"  Recall@{k}:  {micro_stats['recall']:.3f}")
    print(f"  MRR@{k}:     {micro_stats['mrr']:.3f}")
    print(f"  NDCG@{k}:    {micro_stats['ndcg']:.3f}")
    print(f"  Entity@{k}:  {micro_stats['entity']:.3f}")
    print(f"  Time:        {elapsed:.1f}s")
    
    return micro_stats, topic_rows, macro

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="测试集 jsonl 文件")
    ap.add_argument("--model", required=True, help="模型名称，如 intfloat/e5-base-v2")
    ap.add_argument("--granularity", choices=["thread", "turn"], default="turn")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bm25", action="store_true", help="启用 BM25 融合")
    ap.add_argument("--rerank", action="store_true", help="启用 reranking")
    ap.add_argument("--out", default="eval_plus", help="输出文件前缀")
    args = ap.parse_args()
    
    # 加载测试数据
    print(f"📂 加载测试数据: {args.test}")
    test_threads = list(jsonlines.open(args.test))
    print(f"✅ 加载了 {len(test_threads)} threads")
    
    # 评估
    micro, topic_rows, macro = evaluate(
        test_threads,
        emb_model_name=args.model,
        granularity=args.granularity,
        k=args.k,
        use_bm25=args.bm25,
        use_rerank=args.rerank,
        candidates=100,
    )
    
    # 显示汇总
    print("\n" + "="*70)
    print("📊 MICRO (整体)")
    print("="*70)
    for k, v in micro.items():
        print(f"  {k:12s}: {v}")
    
    print("\n" + "="*70)
    print("📊 MACRO (topic 平均)")
    print("="*70)
    for k, v in macro.items():
        print(f"  {k:12s}: {v}")
    
    # 保存 CSV
    micro_path = f"{args.out}.summary.csv"
    topics_path = f"{args.out}.by_topic.csv"
    
    Path(micro_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(micro_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(micro.keys()))
        w.writeheader()
        w.writerow(micro)
    
    with open(topics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "recall", "mrr", "ndcg", "entity", "queries"])
        w.writeheader()
        w.writerows(sorted(topic_rows, key=lambda x: x["topic"]))
    
    print(f"\n✅ 已保存:")
    print(f"   {micro_path}")
    print(f"   {topics_path}")

if __name__ == "__main__":
    main()


