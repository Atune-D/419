#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按主题分层抽样分割 threads 为 train/valid/test (80/10/10)
确保各集合的主题分布一致，避免数据泄露

Usage:
    python split_threads.py --input data/raw/threads.jsonl --output data/working
    python split_threads.py --input output/all_threads.jsonl
"""

import argparse
import jsonlines
import random
import collections
from pathlib import Path

def split_threads_stratified(threads, train_ratio=0.8, valid_ratio=0.1, seed=7):
    """
    按主题分层抽样分割
    
    Args:
        threads: list of thread dicts
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        (train, valid, test) tuples of thread lists
    """
    random.seed(seed)
    
    # 按 topic 分桶
    buckets = collections.defaultdict(list)
    no_topic = []
    
    for th in threads:
        topic = th.get("topic", None)
        if topic:
            buckets[topic].append(th)
        else:
            no_topic.append(th)
    
    if no_topic:
        print(f"⚠️  Warning: {len(no_topic)} threads without topic, assigning to 'other'")
        buckets["other"] = no_topic
    
    # 分层抽样
    train, valid, test = [], [], []
    
    print("\n📊 Stratified sampling by topic:")
    print("-" * 70)
    print(f"{'Topic':<20} {'Total':>8} {'Train':>8} {'Valid':>8} {'Test':>8}")
    print("-" * 70)
    
    for topic, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        random.shuffle(items)
        n = len(items)
        
        # 计算分割点
        n_train = int(train_ratio * n)
        n_valid = int(valid_ratio * n)
        
        # 至少保证每个集合有1个样本（如果总数>=3）
        if n >= 3:
            n_train = max(1, n_train)
            n_valid = max(1, n_valid)
            # 调整确保总和不超过 n
            if n_train + n_valid >= n:
                n_train = n - 2
                n_valid = 1
        
        # 分割
        train_items = items[:n_train]
        valid_items = items[n_train:n_train + n_valid]
        test_items = items[n_train + n_valid:]
        
        train.extend(train_items)
        valid.extend(valid_items)
        test.extend(test_items)
        
        print(f"{topic:<20} {n:>8} {len(train_items):>8} {len(valid_items):>8} {len(test_items):>8}")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {len(threads):>8} {len(train):>8} {len(valid):>8} {len(test):>8}")
    print("-" * 70)
    
    # 打乱最终顺序
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    
    return train, valid, test

def save_jsonl(threads, path):
    """保存 threads 到 JSONL 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(path, "w") as writer:
        for th in threads:
            writer.write(th)
    
    print(f"✅ Saved {len(threads)} threads → {path}")

def main():
    parser = argparse.ArgumentParser(description="Split threads by stratified sampling on topics")
    parser.add_argument("--input", required=True, help="Input JSONL file with all threads")
    parser.add_argument("--output", default="data/working", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Valid ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    args = parser.parse_args()
    
    # 验证比例
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if test_ratio < 0.05:
        print("❌ Error: test_ratio too small! Adjust train/valid ratios.")
        return
    
    print(f"📂 Loading threads from: {args.input}")
    
    try:
        with jsonlines.open(args.input) as reader:
            threads = list(reader)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    if len(threads) == 0:
        print("❌ Error: No threads found in input file!")
        return
    
    print(f"✅ Loaded {len(threads)} threads")
    
    # 数据质量检查
    missing_topic = sum(1 for th in threads if not th.get("topic"))
    missing_turns = sum(1 for th in threads if not th.get("turns"))
    
    if missing_topic > 0:
        print(f"⚠️  Warning: {missing_topic} threads missing 'topic' field")
    if missing_turns > 0:
        print(f"❌ Error: {missing_turns} threads missing 'turns' field!")
        return
    
    # 分层分割
    train, valid, test = split_threads_stratified(
        threads, 
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed
    )
    
    # 保存
    output_dir = Path(args.output)
    save_jsonl(train, output_dir / "threads.train.jsonl")
    save_jsonl(valid, output_dir / "threads.valid.jsonl")
    save_jsonl(test, output_dir / "threads.test.jsonl")
    
    # 统计摘要
    print("\n📈 Summary:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}/threads.{{train,valid,test}}.jsonl")
    print(f"  Total:  {len(threads)} threads")
    print(f"  Split:  {len(train)} / {len(valid)} / {len(test)} = {args.train_ratio:.0%} / {args.valid_ratio:.0%} / {test_ratio:.0%}")
    print(f"  Seed:   {args.seed}")
    
    print("\n💡 Next steps:")
    print(f"  # Verify data quality")
    print(f"  python quality_check.py --file {output_dir}/threads.train.jsonl")
    print(f"")
    print(f"  # Run evaluation")
    print(f"  python eval_matrix.py --test {output_dir}/threads.test.jsonl --models intfloat/e5-base-v2")

if __name__ == "__main__":
    main()


