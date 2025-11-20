#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split JSONL data into train/test sets"""

import argparse
import jsonlines
import random
from pathlib import Path

def split_data(input_file, output_dir, train_ratio=0.8, seed=42):
    """Split data into train and test sets"""
    
    # Read data
    print(f"📂 Reading: {input_file}")
    threads = list(jsonlines.open(input_file))
    print(f"   Total threads: {len(threads)}")
    
    if len(threads) < 10:
        print(f"⚠️  WARNING: Only {len(threads)} threads. Recommend at least 50-100 for reliable evaluation.")
        print(f"   Consider generating more data first.")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(threads)
    
    # Split
    split_point = int(len(threads) * train_ratio)
    train = threads[:split_point]
    test = threads[split_point:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train)} threads ({len(train)/len(threads)*100:.1f}%)")
    print(f"   Test:  {len(test)} threads ({len(test)/len(threads)*100:.1f}%)")
    
    if len(test) < 5:
        print(f"\n⚠️  WARNING: Test set too small ({len(test)} threads)")
        print(f"   Results may not be reliable. Generate more data!")
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    train_file = output_dir / "threads.train.jsonl"
    test_file = output_dir / "threads.test.jsonl"
    
    with jsonlines.open(train_file, 'w') as f:
        f.write_all(train)
    
    with jsonlines.open(test_file, 'w') as f:
        f.write_all(test)
    
    print(f"\n✅ Saved:")
    print(f"   Train: {train_file}")
    print(f"   Test:  {test_file}")
    
    print(f"\n💡 Next steps:")
    print(f"   # Verify test data")
    print(f"   python verify_data.py {test_file}")
    print(f"")
    print(f"   # Run evaluation (test set only used for queries)")
    print(f"   python eval_retrieval_enhanced.py --test {test_file} --k 10 --compare-all")
    
    return train_file, test_file

def main():
    parser = argparse.ArgumentParser(description="Split JSONL data into train/test sets")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    split_data(args.input, args.output_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()




