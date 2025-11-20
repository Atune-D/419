#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 JSONL 数据格式是否适合用于检索评估"""

import argparse
import jsonlines
from collections import Counter

def verify_jsonl(filepath):
    """验证 JSONL 文件格式"""
    print(f"📂 Reading file: {filepath}\n")
    
    try:
        threads = list(jsonlines.open(filepath))
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    if len(threads) == 0:
        print("❌ File is empty!")
        return False
    
    print(f"✅ Loaded {len(threads)} threads\n")
    
    # 统计信息
    total_turns = 0
    topics = Counter()
    turn_counts = []
    missing_scsa = 0
    
    print("📊 Analyzing threads...")
    print("-" * 60)
    
    for i, thread in enumerate(threads):
        # 检查必需字段
        if "thread_id" not in thread:
            print(f"⚠️  Thread {i}: Missing 'thread_id'")
        if "turns" not in thread or not isinstance(thread["turns"], list):
            print(f"⚠️  Thread {i}: Missing or invalid 'turns'")
            continue
        
        turns = thread["turns"]
        turn_counts.append(len(turns))
        total_turns += len(turns)
        
        if "topic" in thread:
            topics[thread["topic"]] += 1
        
        # 检查 SCSA
        for j, turn in enumerate(turns):
            if not turn.get("scsa"):
                missing_scsa += 1
            if "role" not in turn:
                print(f"⚠️  Thread {i}, Turn {j}: Missing 'role'")
            if "body" not in turn:
                print(f"⚠️  Thread {i}, Turn {j}: Missing 'body'")
    
    # 输出统计
    print("\n" + "=" * 60)
    print("📈 Statistics:")
    print("=" * 60)
    print(f"Total threads:        {len(threads)}")
    print(f"Total turns:          {total_turns}")
    print(f"Avg turns/thread:     {total_turns / len(threads):.2f}")
    print(f"Min turns:            {min(turn_counts)}")
    print(f"Max turns:            {max(turn_counts)}")
    print(f"Turns with SCSA:      {total_turns - missing_scsa} / {total_turns} ({(total_turns - missing_scsa) / total_turns * 100:.1f}%)")
    
    if topics:
        print(f"\n📑 Topics distribution:")
        for topic, count in topics.most_common():
            print(f"  {topic:<20} {count:>4} ({count / len(threads) * 100:.1f}%)")
    
    print("=" * 60)
    
    # 显示样例
    print("\n📄 Sample thread (first one):")
    print("-" * 60)
    sample = threads[0]
    print(f"Thread ID: {sample.get('thread_id', 'N/A')}")
    print(f"Topic:     {sample.get('topic', 'N/A')}")
    print(f"Labels:    {sample.get('labels', [])}")
    print(f"Turns:     {len(sample.get('turns', []))}")
    
    if sample.get('turns'):
        print(f"\nFirst turn:")
        first_turn = sample['turns'][0]
        print(f"  Role:    {first_turn.get('role', 'N/A')}")
        print(f"  Subject: {first_turn.get('subject', 'N/A')[:60]}...")
        print(f"  Body:    {first_turn.get('body', 'N/A')[:100]}...")
        if first_turn.get('scsa'):
            print(f"  SCSA:    {first_turn.get('scsa', 'N/A')[:100]}...")
    
    print("=" * 60)
    
    # 最终判断
    print("\n✅ Data format looks good! Ready for evaluation.")
    print(f"\n💡 Quick test command:")
    print(f"   python eval_retrieval_enhanced.py --test {filepath} --k 10 --compare-all")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify JSONL data format for retrieval evaluation")
    parser.add_argument("file", help="Path to JSONL file")
    args = parser.parse_args()
    
    verify_jsonl(args.file)

if __name__ == "__main__":
    main()




