import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import os


class TREC_Evaluator:
    """Evaluator for TREC-COVID results with comprehensive metrics."""
    
    def __init__(self, qrels_path: str):
        """
        Initialize evaluator with qrels file.
        
        Args:
            qrels_path: Path to qrels file (format: qid 0 docid relevance)
        """
        self.qrels = self.load_qrels(qrels_path)
        print(f"✅ Loaded qrels for {len(self.qrels)} queries")
    
    def load_qrels(self, qrels_path: str) -> Dict[str, Dict[str, int]]:
        """
        Load TREC qrels file.
        
        Returns:
            dict[query_id][doc_id] = relevance_score
        """
        qrels = defaultdict(dict)
        
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    qrels[qid][docid] = rel
        
        return qrels
    
    def load_run(self, run_path: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Load TREC run file.
        
        Returns:
            dict[query_id] = [(doc_id, score), ...]
        """
        run = defaultdict(list)
        
        with open(run_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, rank, score, _ = parts
                    run[qid].append((docid, float(score)))
        
        # Sort by score descending
        for qid in run:
            run[qid].sort(key=lambda x: x[1], reverse=True)
        
        return run
    
    def precision_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate Precision@k."""
        if k == 0 or len(retrieved) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant and relevant[doc] > 0)
        
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate Recall@k."""
        if len(relevant) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant and relevant[doc] > 0)
        total_relevant = sum(1 for rel in relevant.values() if rel > 0)
        
        if total_relevant == 0:
            return 0.0
        
        return relevant_retrieved / total_relevant
    
    def f1_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate F1@k."""
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def dcg_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate DCG@k."""
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevant.get(doc, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because rank starts at 1
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate NDCG@k."""
        dcg = self.dcg_at_k(retrieved, relevant, k)
        
        # Calculate ideal DCG
        ideal_rels = sorted(relevant.values(), reverse=True)
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mrr_at_k(self, retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
        """Calculate MRR@k (Mean Reciprocal Rank)."""
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant and relevant[doc] > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_run(self, run: Dict[str, List[Tuple[str, float]]], 
                     k_values: List[int]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate a run file across multiple k values.
        
        Returns:
            dict[metric][k] = score
        """
        metrics = {
            'precision': defaultdict(list),
            'recall': defaultdict(list),
            'f1': defaultdict(list),
            'ndcg': defaultdict(list),
            'mrr': defaultdict(list)
        }
        
        for qid in self.qrels:
            if qid not in run:
                # No results for this query - all metrics are 0
                for k in k_values:
                    metrics['precision'][k].append(0.0)
                    metrics['recall'][k].append(0.0)
                    metrics['f1'][k].append(0.0)
                    metrics['ndcg'][k].append(0.0)
                    metrics['mrr'][k].append(0.0)
                continue
            
            retrieved_docs = [doc for doc, _ in run[qid]]
            relevant_docs = self.qrels[qid]
            
            for k in k_values:
                metrics['precision'][k].append(self.precision_at_k(retrieved_docs, relevant_docs, k))
                metrics['recall'][k].append(self.recall_at_k(retrieved_docs, relevant_docs, k))
                metrics['f1'][k].append(self.f1_at_k(retrieved_docs, relevant_docs, k))
                metrics['ndcg'][k].append(self.ndcg_at_k(retrieved_docs, relevant_docs, k))
                metrics['mrr'][k].append(self.mrr_at_k(retrieved_docs, relevant_docs, k))
        
        # Average across queries
        averaged_metrics = {}
        for metric_name, k_dict in metrics.items():
            averaged_metrics[metric_name] = {k: np.mean(scores) for k, scores in k_dict.items()}
        
        return averaged_metrics


def plot_evaluation_results(results: Dict[str, Dict[str, Dict[int, float]]], 
                           k_values: List[int],
                           output_dir: str = 'evaluation_plots'):
    """
    Create comprehensive evaluation plots.
    
    Args:
        results: dict[run_name][metric][k] = score
        k_values: List of k values to plot
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['precision', 'recall', 'f1', 'ndcg', 'mrr']
    run_names = list(results.keys())
    
    # Color scheme
    colors = {
        'bm25': '#1f77b4',      # Blue
        'rm3': '#ff7f0e',       # Orange
        'doc2query': '#2ca02c'  # Green
    }
    
    markers = {
        'bm25': 'o',
        'rm3': 's',
        'doc2query': '^'
    }
    
    # 1. Individual metric plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Retrieval Performance Comparison Across Methods', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for run_name in run_names:
            scores = [results[run_name][metric][k] for k in k_values]
            ax.plot(k_values, scores, 
                   marker=markers.get(run_name, 'o'),
                   linewidth=2.5,
                   markersize=8,
                   label=run_name.upper(),
                   color=colors.get(run_name, None))
        
        ax.set_xlabel('k', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()}@k', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/all_metrics_comparison.png")
    plt.close()
    
    # 2. Side-by-side comparison at specific k values
    important_k = [1, 5, 10, 20, 50, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Method Comparison at Different k Values', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, k in enumerate(important_k):
        ax = axes[idx]
        
        metric_scores = {metric: [] for metric in metrics}
        
        for run_name in run_names:
            for metric in metrics:
                metric_scores[metric].append(results[run_name][metric][k])
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, run_name in enumerate(run_names):
            scores = [metric_scores[metric][i] for metric in metrics]
            ax.bar(x + i * width, scores, width, 
                  label=run_name.upper(),
                  color=colors.get(run_name, None),
                  alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'k = {k}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/k_value_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/k_value_comparison.png")
    plt.close()
    
    # 3. Heatmap for each method
    for run_name in run_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for heatmap
        data = np.array([[results[run_name][metric][k] for k in k_values] 
                        for metric in metrics])
        
        im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(k_values)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(k_values)
        ax.set_yticklabels([m.upper() for m in metrics])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(k_values)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'{run_name.upper()} Performance Heatmap', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('k', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{run_name}_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/{run_name}_heatmap.png")
        plt.close()
    
    print(f"\n📊 All plots saved to: {output_dir}/")


def print_evaluation_table(results: Dict[str, Dict[str, Dict[int, float]]], 
                          k_values: List[int]):
    """Print evaluation results in a formatted table."""
    
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    
    for run_name in results:
        print(f"\n{'='*100}")
        print(f"Method: {run_name.upper()}")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Metric':<15}"
        for k in k_values:
            header += f"k={k:<6}"
        print(header)
        print("-" * 100)
        
        # Each metric
        for metric in ['precision', 'recall', 'f1', 'ndcg', 'mrr']:
            row = f"{metric.upper():<15}"
            for k in k_values:
                score = results[run_name][metric][k]
                row += f"{score:.4f}  "
            print(row)
    
    print("\n" + "="*100)
    
    # Comparison table
    print("\nCOMPARATIVE SUMMARY (Best scores in each category)")
    print("="*100)
    
    for k in k_values:
        print(f"\nk = {k}")
        print("-" * 80)
        
        for metric in ['precision', 'recall', 'f1', 'ndcg', 'mrr']:
            scores = {run_name: results[run_name][metric][k] for run_name in results}
            best_run = max(scores, key=scores.get)
            best_score = scores[best_run]
            
            print(f"{metric.upper():<15}", end="")
            for run_name in results:
                score = scores[run_name]
                marker = " ★" if run_name == best_run else "  "
                print(f"{run_name.upper()}: {score:.4f}{marker}  ", end="")
            print()


def save_results_to_json(results: Dict[str, Dict[str, Dict[int, float]]], 
                         output_path: str = 'evaluation_results.json'):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to: {output_path}")


def evaluate_all_methods(qrels_path: str, 
                        bm25_file: str, 
                        rm3_file: str, 
                        qrf_file: str,
                        k_values: List[int] = [1, 5, 10, 20, 50, 100],
                        output_dir: str = 'evaluation_plots'):
    """
    Main evaluation function for all methods.
    
    Args:
        qrels_path: Path to qrels file
        bm25_file: Path to BM25 results
        rm3_file: Path to RM3 results
        qrf_file: Path to query reformulation results
        k_values: List of k values for evaluation
        output_dir: Directory to save plots
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION")
    print("="*100)
    
    # Initialize evaluator
    evaluator = TREC_Evaluator(qrels_path)
    
    # Load runs
    print("\nLoading run files...")
    runs = {
        'bm25': evaluator.load_run(bm25_file),
        'rm3': evaluator.load_run(rm3_file),
        'doc2query': evaluator.load_run(qrf_file)
    }
    
    for name, run in runs.items():
        print(f"  ✓ {name.upper()}: {len(run)} queries")
    
    # Evaluate each run
    print(f"\nEvaluating at k = {k_values}...")
    results = {}
    
    for name, run in runs.items():
        print(f"\n  Evaluating {name.upper()}...")
        results[name] = evaluator.evaluate_run(run, k_values)
    
    # Print results
    print_evaluation_table(results, k_values)
    
    # Save to JSON
    save_results_to_json(results, output_path=f'{output_dir}/evaluation_results.json')
    
    # Generate plots
    print("\nGenerating plots...")
    plot_evaluation_results(results, k_values, output_dir)
    
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETED!")
    print("="*100)


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate TREC-COVID retrieval results')
    parser.add_argument('--qrels', type=str, required=True, help='Path to qrels file')
    parser.add_argument('--bm25', type=str, required=True, help='Path to BM25 results')
    parser.add_argument('--rm3', type=str, required=True, help='Path to RM3 results')
    parser.add_argument('--qrf', type=str, required=True, help='Path to query reformulation results')
    parser.add_argument('--output_dir', type=str, default='evaluation_plots', 
                       help='Output directory for plots')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100],
                       help='K values for evaluation')
    
    args = parser.parse_args()
    
    evaluate_all_methods(
        qrels_path=args.qrels,
        bm25_file=args.bm25,
        rm3_file=args.rm3,
        qrf_file=args.qrf,
        k_values=args.k_values,
        output_dir=args.output_dir
    )