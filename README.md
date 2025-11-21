# Customer Service Retrieval Evaluation

Evaluation of embedding models for customer service conversation retrieval.

## Quick Start

```bash
pip install -r requirements.txt

# Evaluate with default settings
python eval_matrix.py \
  --test data/threads.test.jsonl \
  --models intfloat/e5-base-v2 \
  --granularities turn \
  --k 10
```

## Project Overview

**Task**: Retrieve similar customer service conversations given a query.

**Key Results**: 
- E5-base-v2: 96.4% Recall@10, 75.0% MRR@10
- MiniLM-L6-v2: 53.6% Recall@10, 28.3% MRR@10
- E5 outperforms MiniLM by +42.8%

## Core Scripts

| Script | Purpose |
|--------|---------|
| `generateDate.py` | Generate synthetic conversation data |
| `deduplicate.py` | Remove duplicate threads |
| `split_threads.py` | Split into train/valid/test (80/10/10) |
| `eval_matrix.py` | Run evaluation experiments |
| `eval_plus.py` | Enhanced evaluation with multiple metrics |
| `eval_strict.py` | Strict cross-split evaluation |
| `train_embedding.py` | Fine-tune embedding models |
| `visualize_results.py` | Generate plots and reports |

## Repository Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── generateDate.py              # Data generation
├── deduplicate.py               # Deduplication
├── split_threads.py             # Train/valid/test split
│
├── eval_matrix.py               # Main evaluation
├── eval_plus.py                 # Enhanced evaluation
├── eval_strict.py               # Strict evaluation
│
├── train_embedding.py           # Model fine-tuning
└── visualize_results.py         # Visualization
```

## Usage Examples

### Basic Evaluation

```bash
python eval_matrix.py \
  --test data/threads.test.jsonl \
  --models intfloat/e5-base-v2 \
  --granularities turn \
  --k 10
```

### Multi-Model Comparison

```bash
python eval_matrix.py \
  --test data/threads.test.jsonl \
  --models intfloat/e5-base-v2 sentence-transformers/all-MiniLM-L6-v2 \
  --granularities thread turn \
  --k 10
```

### Enhanced Evaluation (Multiple Metrics)

```bash
python eval_plus.py \
  --corpus data/threads.train.jsonl data/threads.valid.jsonl \
  --queries data/threads.test.jsonl \
  --models intfloat/e5-base-v2 \
  --granularities turn \
  --k 10
```

## Key Results

| Model | Granularity | Recall@10 | MRR@10 | NDCG@10 |
|-------|-------------|-----------|--------|---------|
| **E5-base-v2** | **Turn** | **96.4%** | **75.0%** | **89.3%** |
| E5-base-v2 | Thread | 83.3% | 66.0% | 84.8% |
| MiniLM-L6-v2 | Turn | 53.6% | 28.3% | 89.1% |

**Finding**: E5-base-v2 with turn-level granularity achieves production-ready performance.

## Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

Core packages:
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector search
- `jsonlines` - Data I/O
- `pandas` - Data processing
- `tqdm` - Progress bars

## Data Pipeline

1. **Generate data**:
   ```bash
   python generateDate.py --count 800
   ```

2. **Deduplicate**:
   ```bash
   python deduplicate.py --input data/raw.jsonl --output data/dedup.jsonl
   ```

3. **Split dataset**:
   ```bash
   python split_threads.py --input data/dedup.jsonl --output data/
   ```

4. **Evaluate**:
   ```bash
   python eval_matrix.py --test data/threads.test.jsonl
   ```

## Evaluation Metrics

- **Recall@K**: Proportion of queries with correct answer in top K
- **MRR@K**: Mean Reciprocal Rank (ranking quality)
- **NDCG@K**: Normalized Discounted Cumulative Gain (graded relevance)
- **Entity@K**: Entity-level matching rate

## Citation

If you use this code, please cite:

```
[Your Name]. (2025). Evaluation of Embedding Models for Customer 
Service Retrieval. CMPT 419 Course Project, Simon Fraser University.
```

## License

MIT License

