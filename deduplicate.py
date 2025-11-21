#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去除重复或高度相似的 threads

支持：
- 完全重复（基于内容哈希）
- 近似重复（基于首条邮件的文本相似度）
- 按主题+订单ID去重

Usage:
    # 基于哈希去重（快速）
    python deduplicate.py --input output/threads_*.jsonl --output data/raw/threads_dedup.jsonl
    
    # 基于相似度去重（更准确但慢）
    python deduplicate.py --input output/threads_*.jsonl --output data/raw/threads_dedup.jsonl --similarity 0.95
    
    # 从标准输入读取
    cat output/threads_*.jsonl | python deduplicate.py > data/raw/threads.jsonl
"""

import argparse
import jsonlines
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
import glob

def compute_hash(thread):
    """
    计算 thread 的哈希值（用于精确去重）
    基于所有 turn 的 body
    """
    bodies = []
    for turn in thread.get("turns", []):
        body = turn.get("body", "").strip()
        if body:
            bodies.append(body)
    
    content = "\n".join(bodies)
    return hashlib.md5(content.encode()).hexdigest()

def compute_first_turn_hash(thread):
    """
    计算首条邮件的哈希（用于粗粒度去重）
    """
    if thread.get("turns") and len(thread["turns"]) > 0:
        first_turn = thread["turns"][0]
        body = first_turn.get("body", "").strip()
        return hashlib.md5(body.encode()).hexdigest()
    return None

def get_metadata_key(thread):
    """
    获取元数据键（主题 + 订单ID）
    """
    topic = thread.get("topic", "unknown")
    order_id = thread.get("meta", {}).get("order_id", "")
    return f"{topic}_{order_id}"

def deduplicate_exact(threads):
    """
    精确去重：完全相同的 threads
    """
    seen = set()
    unique = []
    duplicates = 0
    
    for thread in threads:
        h = compute_hash(thread)
        if h not in seen:
            seen.add(h)
            unique.append(thread)
        else:
            duplicates += 1
    
    print(f"  Exact dedup: {len(threads)} → {len(unique)} ({duplicates} duplicates removed)")
    return unique

def deduplicate_first_turn(threads):
    """
    首条邮件去重：首条邮件相同的视为重复
    """
    seen = set()
    unique = []
    duplicates = 0
    
    for thread in threads:
        h = compute_first_turn_hash(thread)
        if h and h not in seen:
            seen.add(h)
            unique.append(thread)
        elif not h:
            # 没有 turn 的也保留
            unique.append(thread)
        else:
            duplicates += 1
    
    print(f"  First-turn dedup: {len(threads)} → {len(unique)} ({duplicates} duplicates removed)")
    return unique

def deduplicate_by_metadata(threads):
    """
    基于元数据去重：相同主题+订单ID的保留最长的一个
    """
    groups = defaultdict(list)
    
    for thread in threads:
        key = get_metadata_key(thread)
        groups[key].append(thread)
    
    unique = []
    duplicates = 0
    
    for key, group in groups.items():
        if len(group) == 1:
            unique.append(group[0])
        else:
            # 保留 turns 最多的
            best = max(group, key=lambda t: len(t.get("turns", [])))
            unique.append(best)
            duplicates += len(group) - 1
    
    print(f"  Metadata dedup: {len(threads)} → {len(unique)} ({duplicates} duplicates removed)")
    return unique

def deduplicate_similarity(threads, threshold=0.95):
    """
    基于相似度去重（使用简单的 Jaccard 相似度）
    注意：这个方法较慢，适用于小规模数据
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        print("  ⚠️  sklearn not installed, skipping similarity dedup")
        return threads
    
    # 提取文本
    texts = []
    for thread in threads:
        bodies = []
        for turn in thread.get("turns", []):
            body = turn.get("body", "").strip()
            if body:
                bodies.append(body)
        texts.append(" ".join(bodies))
    
    if not texts:
        return threads
    
    # 计算相似度矩阵
    print(f"  Computing similarity matrix for {len(texts)} threads...")
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf)
    
    # 去重
    keep = [True] * len(threads)
    duplicates = 0
    
    for i in range(len(threads)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(threads)):
            if not keep[j]:
                continue
            if similarity_matrix[i, j] >= threshold:
                # 保留 turns 更多的
                if len(threads[i].get("turns", [])) >= len(threads[j].get("turns", [])):
                    keep[j] = False
                else:
                    keep[i] = False
                    break
                duplicates += 1
    
    unique = [t for i, t in enumerate(threads) if keep[i]]
    print(f"  Similarity dedup (threshold={threshold}): {len(threads)} → {len(unique)} ({duplicates} duplicates removed)")
    return unique

def main():
    parser = argparse.ArgumentParser(description="Deduplicate threads")
    parser.add_argument("--input", nargs="*", help="Input JSONL file(s) (supports glob patterns)")
    parser.add_argument("--output", help="Output JSONL file (if not provided, writes to stdout)")
    parser.add_argument("--method", default="all", 
                       choices=["exact", "first_turn", "metadata", "similarity", "all"],
                       help="Deduplication method")
    parser.add_argument("--similarity", type=float, default=0.95,
                       help="Similarity threshold for similarity dedup (0.0-1.0)")
    args = parser.parse_args()
    
    # 读取输入
    threads = []
    
    if args.input:
        # 从文件读取
        files = []
        for pattern in args.input:
            files.extend(glob.glob(pattern))
        
        if not files:
            print("❌ No input files found!")
            return
        
        print(f"📂 Reading from {len(files)} file(s):")
        for f in files:
            print(f"  - {f}")
        
        for f in files:
            try:
                with jsonlines.open(f) as reader:
                    threads.extend(list(reader))
            except Exception as e:
                print(f"  ⚠️  Error reading {f}: {e}")
    else:
        # 从标准输入读取
        print(f"📂 Reading from stdin...")
        try:
            with jsonlines.Reader(sys.stdin) as reader:
                threads = list(reader)
        except Exception as e:
            print(f"❌ Error reading stdin: {e}")
            return
    
    if not threads:
        print("❌ No threads found!")
        return
    
    print(f"✅ Loaded {len(threads)} threads\n")
    
    # 去重
    print(f"🔄 Deduplicating (method: {args.method})...")
    
    if args.method == "exact" or args.method == "all":
        threads = deduplicate_exact(threads)
    
    if args.method == "first_turn" or args.method == "all":
        threads = deduplicate_first_turn(threads)
    
    if args.method == "metadata" or args.method == "all":
        threads = deduplicate_by_metadata(threads)
    
    if args.method == "similarity":
        threads = deduplicate_similarity(threads, threshold=args.similarity)
    
    print(f"\n✅ Final: {len(threads)} unique threads\n")
    
    # 输出
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_path, "w") as writer:
            for thread in threads:
                writer.write(thread)
        
        print(f"✅ Saved to: {args.output}")
    else:
        # 输出到标准输出
        with jsonlines.Writer(sys.stdout) as writer:
            for thread in threads:
                writer.write(thread)

if __name__ == "__main__":
    main()


