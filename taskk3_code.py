import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict, Counter
from pyserini.search import SimpleSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class QueryReformulator:
   
    def __init__(self, model_name='castorini/doc2query-t5-base-msmarco', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Doc2Query model: {model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.load_sentence_encoder()
        print(f"Model loaded on device: {self.device}")

    def load_sentence_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_encoder.to(self.device)
            print("Loaded sentence encoder for semantic filtering.")
        except ImportError:
            print("⚠️ sentence-transformers not available, skipping semantic filtering.")
            self.sentence_encoder = None

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        if self.sentence_encoder is None:
            return 1.0
        
        with torch.no_grad():
            emb = self.sentence_encoder.encode([text1, text2], convert_to_tensor=True)
            similarity = F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0)).item()
        return similarity

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'it', 'its', 'they', 'them', 'their', 'i', 's', 'also',
            'we', 'our', 'such', 'more', 'been', 'other', 'into', 'through'
        }
        
        medical_indicators = {
            'covid', 'sars', 'cov', 'coronavirus', 'virus', 'viral', 'infection',
            'disease', 'patient', 'clinical', 'treatment', 'therapy', 'vaccine',
            'antibody', 'immune', 'pneumonia', 'respiratory', 'syndrome', 
            'diagnosis', 'symptom', 'epidemic', 'pandemic', 'transmission',
            'protein', 'cell', 'molecular', 'genetic', 'rna', 'dna', 'study'
        }
        
        words = []
        for w in text.split():
            w_clean = w.lower().strip('.,;:!?()')
            if len(w_clean) > 2 and w_clean not in stopwords:
                
                if any(ind in w_clean for ind in medical_indicators):
                    words.extend([w_clean, w_clean])  
                else:
                    words.append(w_clean)
        
        counter = Counter(words)
        
        
        ranked_terms = []
        for word, freq in counter.most_common(top_k * 2):
            
            length_bonus = min(len(word) / 5.0, 2.0)  
            medical_bonus = 2.0 if any(ind in word for ind in medical_indicators) else 1.0
            score = freq * length_bonus * medical_bonus
            ranked_terms.append((word, score))

        ranked_terms.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in ranked_terms[:top_k]]

    def generate_expansions(self, query: str, num_sequences=15) -> List[str]:
       
        input_text = f"Generate document: {query}"
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt',
            truncation=True, 
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=180, 
                num_return_sequences=num_sequences,
                num_beams=num_sequences * 2,
                do_sample=True,
                top_k=70,  # More diversity
                top_p=0.95,  # Higher for more varied generations
                temperature=1.0,  # Increased temperature for diversity
                no_repeat_ngram_size=3,
                early_stopping=True,
                diversity_penalty=0.8  # Stronger diversity penalty
            )
        
        expansion_terms = []
        for out in outputs:
            generated = self.tokenizer.decode(out, skip_special_tokens=True)
            # Extract more keywords per document
            expansion_terms.extend(self.extract_keywords(generated, top_k=7))
        
        return list(set(expansion_terms))

    def reformulate_query(self, query: str, num_expansions: int = 15, 
                         similarity_threshold: float = 0.20) -> str:
       
        print(f"\n🔹 Reformulating: '{query[:100]}...'")
        
        expansion_terms = self.generate_expansions(query, num_sequences=15)
        
        if not expansion_terms:
            print("   No expansions generated, returning original query")
            return query
        
        filtered_terms = []
        for term in expansion_terms:
            sim = self.get_semantic_similarity(query, term)
            if sim > similarity_threshold:
                filtered_terms.append((term, sim))
        
        # Apply diversity selection with more terms
        if len(filtered_terms) > num_expansions:
            diverse_terms = self._select_diverse_terms(filtered_terms, num_expansions)
        else:
            diverse_terms = filtered_terms
        
        # Sort by similarity
        diverse_terms = sorted(diverse_terms, key=lambda x: x[1], reverse=True)
        
        query_parts = [query, query, query, query]
        
        for term, sim in diverse_terms[:num_expansions]:
            if sim > 0.5:  # High similarity - boost heavily
                query_parts.extend([term, term, term])  # 3x boost
            elif sim > 0.35:  # Medium-high similarity
                query_parts.extend([term, term])  # 2x boost
            elif sim > 0.25:  # Medium similarity
                query_parts.append(term)  # 1x
            else:  # Lower similarity - still useful for recall
                query_parts.append(term)
        
        reformulated = ' '.join(query_parts)
        
        print(f"   Expansions: {[t for t, _ in diverse_terms[:8]]}")
        print(f"   ✅ Reformulated query length: {len(reformulated.split())} words")
        
        return reformulated
    
    def _select_diverse_terms(self, terms: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
        
        if not terms or k <= 0:
            return []
        
        # Start with highest similarity term
        selected = [terms[0]]
        remaining = terms[1:]
        
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, (term, relevance) in enumerate(remaining):
                # Calculate max similarity to already selected terms
                max_sim = 0.0
                for sel_term, _ in selected:
                    sim = self.get_semantic_similarity(term, sel_term)
                    max_sim = max(max_sim, sim)
                
                mmr_score = 0.75 * relevance - 0.25 * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining[best_idx])
            remaining = remaining[:best_idx] + remaining[best_idx+1:]
        
        return selected


class CORD19DataLoader:
    """Load CORD-19 TREC-COVID queries."""
    
    @staticmethod
    def load_queries_from_file(query_path: str) -> Dict[str, str]:
        
        queries = {}
        
        print(f"Loading queries from: {query_path}")
        
        with open(query_path, 'r', encoding='utf-16') as f:
            for line in f:
                line = line.strip()
                if line:
                    query_obj = json.loads(line)
                    query_id = str(query_obj["query_id"])
                    
                    # Combine title and description (weighted)
                    title = query_obj.get("title", "").strip()
                    description = query_obj.get("description", "").strip()
                    
                    # Title appears twice for emphasis, then description
                    combined_query = f"{title} {title} {description}".strip()
                    queries[query_id] = combined_query
        
        print(f"✅ Loaded {len(queries)} queries from {query_path}")
        return queries


# ---------------------------------------------------------------------
# Retrieval System
# ---------------------------------------------------------------------
class CORD19Retrieval:
    def __init__(self, index_name: str = 'trec-covid-r1-full-text'):
        """Initialize retrieval system with TREC-COVID Round 1 index."""
        print(f"Loading prebuilt index: {index_name}")
        try:
            self.searcher = SimpleSearcher.from_prebuilt_index(index_name)
            print(f"✅ Index loaded: {self.searcher.num_docs} documents")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            print("\nTrying alternative: beir-v1.0.0-trec-covid.flat")
            try:
                from pyserini.search.lucene import LuceneSearcher
                self.searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-covid.flat')
                print(f"✅ Index loaded: {self.searcher.num_docs} documents")
            except Exception as e2:
                print(f"❌ Failed to load alternative index: {e2}")
                raise
        
        self.reformulator = None

    def load_reformulator(self):
        """Load query reformulator."""
        if self.reformulator is None:
            self.reformulator = QueryReformulator()

    def retrieve_bm25(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Standard BM25 retrieval."""
        hits = self.searcher.search(query, k=k)
        return [(h.docid, h.score) for h in hits]

    def retrieve_rm3(self, query: str, k: int) -> List[Tuple[str, float]]:
        """RM3 pseudo-relevance feedback retrieval."""
        self.searcher.set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5)
        hits = self.searcher.search(query, k=k)
        self.searcher.unset_rm3()
        return [(h.docid, h.score) for h in hits]

    def retrieve_reformulated(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self.reformulator is None:
            self.load_reformulator()
        
        # Stage 1: Reformulate query using Doc2Query
        reformulated_query = self.reformulator.reformulate_query(query, num_expansions=15)
        
        # Stage 2: Initial retrieval with reformulated query (get 3x documents)
        initial_hits = self.searcher.search(reformulated_query, k=min(k*3, 200))
        
        if len(initial_hits) == 0:
            # Fallback to original query
            return self.retrieve_bm25(query, k)
        
        # Stage 3: Enhanced pseudo-relevance feedback from top-10 documents
        if len(initial_hits) >= 10:
            feedback_terms = []
            term_frequencies = defaultdict(int)
            
            for hit in initial_hits[:10]:  # Use top-10 for PRF
                try:
                    doc = self.searcher.doc(hit.docid)
                    if doc.raw():
                        doc_text = json.loads(doc.raw())
                        title = doc_text.get('title', '')
                        abstract = doc_text.get('abstract', '')
                        
                        # Combine with title weighted more
                        combined_text = f"{title} {title} {abstract}"
                        
                        # Extract keywords with higher yield
                        terms = self.reformulator.extract_keywords(combined_text, top_k=5)
                        for term in terms:
                            term_frequencies[term] += 1
                except:
                    continue
            
            # Select top feedback terms by frequency
            if term_frequencies:
                sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
                feedback_terms = [term for term, _ in sorted_terms[:8]]  # Top-8 terms
                
                enhanced_query = f"{query} {query} {reformulated_query} {' '.join(feedback_terms)}"
                
                # Stage 5: Final retrieval with enhanced query
                final_hits = self.searcher.search(enhanced_query, k=k*2)
                
                # Return top-k with score normalization
                return [(h.docid, h.score) for h in final_hits[:k]]
        
        # Fallback: return initial results
        return [(h.docid, h.score) for h in initial_hits[:k]]


def task3_improve(query_path: str, bm25_output_file: str, rm3_output_file: str, 
                  qrf_output_file: str, k: int = 50):
    print("\n" + "="*70)
    print("CORD-19 TREC-COVID Task 3: Query Reformulation")
    print("="*70)
    print(f"Query file: {query_path}")
    print(f"Retrieval depth: k={k}")
    print(f"Output files:")
    print(f"  - BM25: {bm25_output_file}")
    print(f"  - RM3: {rm3_output_file}")
    print(f"  - QRF: {qrf_output_file}")
    
    # Initialize retrieval system
    print("\nInitializing retrieval system...")
    retrieval = CORD19Retrieval(index_name='trec-covid-r1-full-text')
    
    # Load queries
    queries = CORD19DataLoader.load_queries_from_file(query_path)
    
    # Process each method
    methods = [
        ('bm25', bm25_output_file, retrieval.retrieve_bm25),
        ('rm3', rm3_output_file, retrieval.retrieve_rm3),
        ('doc2query', qrf_output_file, retrieval.retrieve_reformulated)
    ]
    
    for method_name, output_file, retrieve_fn in methods:
        print(f"\n{'='*70}")
        print(f"Running {method_name.upper()}")
        print(f"{'='*70}")
        
        results = []
        
        for qid, qtext in sorted(queries.items(), key=lambda x: int(x[0])):
            try:
                # Retrieve documents
                retrieved = retrieve_fn(qtext, k)
                
                # Write in TREC format: qid Q0 docid rank score run_name
                for rank, (docid, score) in enumerate(retrieved, start=1):
                    results.append(f"{qid} Q0 {docid} {rank} {score:.6f} {method_name}\n")
                
                if int(qid) % 5 == 0:
                    print(f"✓ Processed query {qid} ({len(retrieved)} results)")
            
            except Exception as e:
                print(f"❌ Error processing query {qid}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Write results to file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            f.writelines(results)
        
        print(f"✅ Saved {len(results)} results to {output_file}")
    
    print("\n" + "="*70)
    print("✅ Task 3 completed successfully!")
    print("="*70)

def main():
    """Main entry point for standalone testing."""
    parser = argparse.ArgumentParser(
        description='CORD-19 Query Reformulation and Retrieval (Task 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--query_path',
        type=str,
        required=True,
        help='Path to query file (UTF-16 encoded JSONL)'
    )
    
    parser.add_argument(
        '--bm25_output_file',
        type=str,
        required=True,
        help='Path to write BM25 results (TREC format)'
    )
    
    parser.add_argument(
        '--rm3_output_file',
        type=str,
        required=True,
        help='Path to write RM3 results (TREC format)'
    )
    
    parser.add_argument(
        '--qrf_output_file',
        type=str,
        required=True,
        help='Path to write query reformulation results (TREC format)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=50,
        help='Number of documents to retrieve per query (default: 50)'
    )
    
    args = parser.parse_args()
    
    try:
        task3_improve(
            query_path=args.query_path,
            bm25_output_file=args.bm25_output_file,
            rm3_output_file=args.rm3_output_file,
            qrf_output_file=args.qrf_output_file,
            k=args.k
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())