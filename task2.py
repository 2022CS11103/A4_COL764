"""
Task 2: SPLADE Retrieval - Final Version (No Comparison)
Author: Ananya Singh
Dataset: msmarco-passage/trec-dl-hard (first 50 queries)
"""

import os
import sys
import json
import time
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

from pyserini.search.lucene import LuceneImpactSearcher

print("=" * 80)
print("TASK 2: SPLADE (Learned Sparse Retrieval)")
print("=" * 80)


# ----------------------------------------------------------------------
# STEP 1 — LOAD 50 QUERIES AND QRELS
# ----------------------------------------------------------------------
def load_queries_qrels():
    """Load first 50 queries and qrels from msmarco-passage/trec-dl-hard."""
    print("\nStep 1: Loading Queries & Qrels (TREC-DL-HARD)")
    print("-" * 80)

    try:
        import ir_datasets
    except ImportError:
        print("✗ ir_datasets not installed.")
        print("→ Run: pip install ir-datasets")
        sys.exit(1)

    dataset_name = "msmarco-passage/trec-dl-hard"
    print(f"Loading dataset: {dataset_name} ...")
    dataset = ir_datasets.load(dataset_name)

    # ---- Load first 50 queries ----
    queries = {}
    for i, q in enumerate(dataset.queries_iter()):
        if i >= 50:
            break
        queries[q.query_id] = q.text
    print(f"✓ Loaded {len(queries)} queries (limited to 50)")

    # ---- Save queries.json ----
    queries_path = "queries.json"
    with open(queries_path, "w") as f:
        json.dump({"queries": [{"id": str(k), "text": v} for k, v in queries.items()]}, f, indent=2)
    print(f"✓ Saved queries to {queries_path}")

    # ---- Save qrels.txt ----
    qrels_path = "qrels.txt"
    selected_qids = set(queries.keys())
    count = 0
    with open(qrels_path, "w") as f:
        for q in dataset.qrels_iter():
            if q.query_id in selected_qids:
                f.write(f"{q.query_id}\t0\t{q.doc_id}\t{q.relevance}\n")
                count += 1
    print(f"✓ Saved qrels ({count} relevance judgments) → {qrels_path}")
    print("-" * 80)
    return queries, qrels_path


# ----------------------------------------------------------------------
# STEP 2 — INITIALIZE SPLADE
# ----------------------------------------------------------------------
def initialize_splade():
    """Initialize SPLADE searcher."""
    print("\nStep 2: Initialize SPLADE Searcher")
    print("-" * 80)

    query_encoder = "naver/splade-cocondenser-ensembledistil"
    prebuilt_index = "msmarco-v1-passage-splade-pp-ed"

    print(f"Using SPLADE index: {prebuilt_index}")
    print(f"Using encoder: {query_encoder}")

    try:
        searcher = LuceneImpactSearcher.from_prebuilt_index(prebuilt_index, query_encoder)
        print(f"✓ SPLADE initialized successfully!")
        print(f"  → Index documents: {searcher.num_docs:,}")
        return searcher
    except Exception as e:
        print(f"✗ Error loading SPLADE: {e}")
        sys.exit(1)


# ----------------------------------------------------------------------
# STEP 3 — RETRIEVE DOCUMENTS
# ----------------------------------------------------------------------
def batch_search(searcher, queries: Dict[str, str], k: int = 1000):
    """Perform retrieval for 50 queries."""
    print("\nStep 3: Retrieving Documents")
    print("-" * 80)
    print(f"→ Running SPLADE retrieval for {len(queries)} queries...")
    print(f"→ Top-k per query: {k}")

    results = {}
    start = time.time()

    for i, (qid, text) in enumerate(queries.items(), 1):
        hits = searcher.search(text, k=k)
        results[qid] = [(hit.docid, hit.score) for hit in hits]
        if i % 10 == 0 or i == len(queries):
            print(f"  Progress: {i}/{len(queries)}")

    elapsed = time.time() - start
    print(f"\n✓ Retrieval complete in {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"  Avg per query: {elapsed/len(queries):.2f}s")
    print(f"  Queries per second: {len(queries)/elapsed:.2f}")
    return results


# ----------------------------------------------------------------------
# STEP 4 — SAVE RUN FILE
# ----------------------------------------------------------------------
def save_run(results: Dict, path: str):
    """Save run file in TREC format."""
    print("\nStep 4: Saving Run File")
    print("-" * 80)
    with open(path, "w") as f:
        for qid, docs in results.items():
            for rank, (docid, score) in enumerate(docs, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} splade\n")
    print(f"✓ Saved run file to {path}")


# ----------------------------------------------------------------------
# STEP 5 — EVALUATE
# ----------------------------------------------------------------------
def evaluate(run_path: str, qrels_path: str):
    """Evaluate SPLADE results using JTrecEval (Java) with metrics up to @100."""
    print("\nStep 5: Evaluate")
    print("-" * 80)

    eval_jar = os.path.expanduser("~/.cache/pyserini/eval/jtreceval-0.0.5-jar-with-dependencies.jar")
    if not os.path.exists(eval_jar):
        print("✗ JTrecEval not found. Please run once with internet to auto-download.")
        return {}

    # ✅ Include MAP, MRR, and Precision/Recall/NDCG for 1,5,10,20,50,100
    metric_pairs = [
        ("map", "Mean Average Precision (MAP)"),
        ("recip_rank", "Mean Reciprocal Rank (MRR)"),

        # Precision metrics
        ("P_1", "Precision@1"), ("P.1", "Precision@1"),
        ("P_5", "Precision@5"), ("P.5", "Precision@5"),
        ("P_10", "Precision@10"), ("P.10", "Precision@10"),
        ("P_20", "Precision@20"), ("P.20", "Precision@20"),
        ("P_50", "Precision@50"), ("P.50", "Precision@50"),
        ("P_100", "Precision@100"), ("P.100", "Precision@100"),

        # Recall metrics
        ("recall_1", "Recall@1"), ("recall.1", "Recall@1"),
        ("recall_5", "Recall@5"), ("recall.5", "Recall@5"),
        ("recall_10", "Recall@10"), ("recall.10", "Recall@10"),
        ("recall_20", "Recall@20"), ("recall.20", "Recall@20"),
        ("recall_50", "Recall@50"), ("recall.50", "Recall@50"),
        ("recall_100", "Recall@100"), ("recall.100", "Recall@100"),

        # NDCG metrics
        ("ndcg_cut_1", "NDCG@1"), ("ndcg_cut.1", "NDCG@1"),
        ("ndcg_cut_5", "NDCG@5"), ("ndcg_cut.5", "NDCG@5"),
        ("ndcg_cut_10", "NDCG@10"), ("ndcg_cut.10", "NDCG@10"),
        ("ndcg_cut_20", "NDCG@20"), ("ndcg_cut.20", "NDCG@20"),
        ("ndcg_cut_50", "NDCG@50"), ("ndcg_cut.50", "NDCG@50"),
        ("ndcg_cut_100", "NDCG@100"), ("ndcg_cut.100", "NDCG@100")
    ]

    results = {}
    printed = set()
    import subprocess

    print("\nMetric                              | Score")
    print("-" * 80)

    for metric_key, metric_name in metric_pairs:
        # Avoid duplicates (print each metric name only once)
        if metric_name in printed:
            continue

        try:
            cmd = ["java", "-jar", eval_jar, "-m", metric_key, qrels_path, run_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            score = None

            # Parse the output
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] == "all":
                    try:
                        score = float(parts[2])
                        break
                    except ValueError:
                        continue

            if score is not None:
                results[metric_name] = {"name": metric_name, "score": score}
                print(f"{metric_name:35s} | {score:.4f}")
                printed.add(metric_name)
        except Exception:
            continue

    print("-" * 80)
    return results

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    OUTPUT_DIR = "task2_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 80)

    queries, qrels_path = load_queries_qrels()
    searcher = initialize_splade()
    results = batch_search(searcher, queries, k=1000)

    run_file = os.path.join(OUTPUT_DIR, "splade_run.txt")
    save_run(results, run_file)

    eval_results = evaluate(run_file, qrels_path)
    eval_file = os.path.join(OUTPUT_DIR, "splade_eval.json")
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 80)
    print("TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Results saved in folder: {OUTPUT_DIR}")
    print("Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f:25s} ({size:6.1f} KB)")

    print("\n" + "=" * 80)
    print("Key Metrics (SPLADE):")
    print("=" * 80)
    for key, val in eval_results.items():
        print(f"  {val['name']:35s}: {val['score']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
