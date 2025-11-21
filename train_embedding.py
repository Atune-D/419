#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练/微调嵌入模型（当检索不达标时使用）

使用 MultipleNegativesRankingLoss 训练嵌入模型
适用于提升检索性能

Usage:
    # 基础训练
    python train_embedding.py \
      --train data/working/threads.train.jsonl \
      --valid data/working/threads.valid.jsonl \
      --base-model intfloat/e5-base-v2 \
      --output models/e5-finetuned
    
    # 快速训练（1 epoch）
    python train_embedding.py \
      --train data/working/threads.train.jsonl \
      --base-model intfloat/e5-base-v2 \
      --epochs 1 \
      --output models/e5-finetuned-quick
"""

import argparse
import jsonlines
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

def load_threads(path):
    """加载 threads"""
    return list(jsonlines.open(path))

def last_customer_turn(thread):
    """获取最后一条客户邮件"""
    for turn in reversed(thread.get("turns", [])):
        if turn.get("role") == "customer":
            scsa = turn.get("scsa")
            body = turn.get("body", "")
            return scsa if isinstance(scsa, str) and scsa else body
    return None

def thread_to_text(thread):
    """将 thread 转换为文本（用于正样本）"""
    parts = []
    for turn in thread.get("turns", [])[:4]:  # 前4个turns
        scsa = turn.get("scsa", "")
        if isinstance(scsa, str) and scsa:
            parts.append(scsa)
        role = turn.get("role", "").upper()
        subj = turn.get("subject", "")
        body = turn.get("body", "")
        parts.append(f"[{role}] {subj}\n{body}")
    return "\n\n".join(parts)

def create_training_samples(threads, samples_per_thread=1):
    """
    创建训练样本 (query, positive)
    
    Args:
        threads: list of thread dicts
        samples_per_thread: 每个thread生成多少个样本
    
    Returns:
        list of (query, positive) tuples
    """
    samples = []
    
    for thread in threads:
        query = last_customer_turn(thread)
        if not query:
            continue
        
        positive = thread_to_text(thread)
        if not positive:
            continue
        
        for _ in range(samples_per_thread):
            samples.append((query, positive))
    
    return samples

def train_model(train_samples, base_model, output_dir, 
                epochs=2, batch_size=16, valid_samples=None, 
                warmup_steps=100, learning_rate=2e-5):
    """
    训练嵌入模型
    
    Args:
        train_samples: list of (query, positive) tuples
        base_model: 基础模型名称
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: batch size
        valid_samples: 验证集样本
        warmup_steps: warmup steps
        learning_rate: 学习率
    """
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("❌ sentence-transformers not installed!")
        print("   Install with: pip install sentence-transformers")
        return False
    
    print(f"📦 Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # 转换为 InputExample
    print(f"🔄 Converting {len(train_samples)} samples to InputExamples...")
    train_examples = [
        InputExample(texts=[query, positive])
        for query, positive in train_samples
    ]
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # 定义 loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 验证集（可选）
    evaluator = None
    if valid_samples:
        from sentence_transformers.evaluation import InformationRetrievalEvaluator
        
        print(f"🔄 Preparing validation evaluator with {len(valid_samples)} samples...")
        
        # 构建验证用的查询和语料库
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, (query, positive) in enumerate(valid_samples[:100]):  # 最多100个验证样本
            query_id = f"q{i}"
            doc_id = f"d{i}"
            
            queries[query_id] = query
            corpus[doc_id] = positive
            relevant_docs[query_id] = {doc_id}
        
        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs,
            name="validation"
        )
    
    # 训练
    print(f"\n🚀 Starting training...")
    print(f"  Base model:    {base_model}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output:        {output_dir}\n")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=500 if evaluator else 0,
        output_path=output_dir,
        save_best_model=True if evaluator else False,
        show_progress_bar=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune embedding model")
    parser.add_argument("--train", required=True, help="Training JSONL file")
    parser.add_argument("--valid", help="Validation JSONL file (optional)")
    parser.add_argument("--base-model", default="intfloat/e5-base-v2",
                       help="Base model to fine-tune")
    parser.add_argument("--output", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--samples-per-thread", type=int, default=1,
                       help="Training samples per thread")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载数据
    print(f"📂 Loading training data from: {args.train}")
    train_threads = load_threads(args.train)
    print(f"✅ Loaded {len(train_threads)} training threads")
    
    valid_threads = None
    if args.valid:
        print(f"📂 Loading validation data from: {args.valid}")
        valid_threads = load_threads(args.valid)
        print(f"✅ Loaded {len(valid_threads)} validation threads")
    
    # 创建训练样本
    print(f"\n🔨 Creating training samples...")
    train_samples = create_training_samples(
        train_threads, 
        samples_per_thread=args.samples_per_thread
    )
    print(f"✅ Created {len(train_samples)} training samples")
    
    valid_samples = None
    if valid_threads:
        print(f"🔨 Creating validation samples...")
        valid_samples = create_training_samples(valid_threads, samples_per_thread=1)
        print(f"✅ Created {len(valid_samples)} validation samples")
    
    # 检查样本质量
    if len(train_samples) < 100:
        print(f"\n⚠️  Warning: Only {len(train_samples)} training samples!")
        print(f"   Consider generating more data or increasing --samples-per-thread")
    
    # 训练
    output_dir = Path(args.output)
    success = train_model(
        train_samples=train_samples,
        base_model=args.base_model,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        valid_samples=valid_samples,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
    )
    
    if success:
        print(f"\n💡 Next steps:")
        print(f"  # Evaluate the fine-tuned model")
        print(f"  python eval_matrix.py \\")
        print(f"    --test data/working/threads.test.jsonl \\")
        print(f"    --models {output_dir} \\")
        print(f"    --granularities thread turn")
        print(f"")
        print(f"  # Compare with base model")
        print(f"  python eval_matrix.py \\")
        print(f"    --test data/working/threads.test.jsonl \\")
        print(f"    --models {args.base_model} {output_dir} \\")
        print(f"    --granularities turn")

if __name__ == "__main__":
    main()


