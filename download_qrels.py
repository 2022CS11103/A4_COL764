"""
Download TREC-COVID Round 1 queries and qrels that match the Pyserini index
"""

import urllib.request
import json
import os


def download_round1_queries(output_path='queries-round1.jsonl'):
    """Download TREC-COVID Round 1 queries."""
    
    print("="*80)
    print("Downloading TREC-COVID Round 1 Queries")
    print("="*80)
    
    # TREC-COVID Round 1 topics (30 queries)
    url = 'https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml'
    
    print(f"\nDownloading from: {url}")
    
    try:
        # Download XML file
        xml_file = 'topics-rnd1.xml'
        urllib.request.urlretrieve(url, xml_file)
        
        # Parse XML and convert to JSONL
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        queries = []
        for topic in root.findall('topic'):
            number = topic.get('number')
            query = topic.find('query').text.strip()
            question = topic.find('question').text.strip()
            narrative = topic.find('narrative').text.strip()
            
            query_obj = {
                'query_id': number,
                'title': query,
                'description': question,
                'narrative': narrative
            }
            queries.append(query_obj)
        
        # Write to JSONL (UTF-16 for compatibility)
        with open(output_path, 'w', encoding='utf-16') as f:
            for q in queries:
                f.write(json.dumps(q) + '\n')
        
        print(f"✅ Downloaded {len(queries)} queries")
        print(f"   Saved to: {output_path}")
        
        # Show sample
        print(f"\n📄 Sample query:")
        print(f"   Query ID: {queries[0]['query_id']}")
        print(f"   Title: {queries[0]['title']}")
        
        # Clean up
        os.remove(xml_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_round1_qrels(output_path='qrels-round1.txt'):
    """Download TREC-COVID Round 1 qrels."""
    
    print("\n" + "="*80)
    print("Downloading TREC-COVID Round 1 Qrels")
    print("="*80)
    
    # Try multiple sources
    urls = [
        'https://ir.nist.gov/covidSubmit/data/qrels-covid_d1_j0.5-1.txt',
        'https://ir.nist.gov/trec-covid/qrels/qrels-covid_d1_j0.5-1.txt'
    ]
    
    for url in urls:
        try:
            print(f"\nTrying: {url}")
            urllib.request.urlretrieve(url, output_path)
            
            # Verify it's a valid qrels file
            with open(output_path, 'r') as f:
                first_line = f.readline()
                if 'html' in first_line.lower() or 'xml' in first_line.lower():
                    print(f"  ❌ Downloaded HTML/XML, not qrels")
                    continue
                
                # Count lines
                f.seek(0)
                num_lines = sum(1 for _ in f)
                
                print(f"  ✅ Downloaded qrels")
                print(f"     Lines: {num_lines}")
                print(f"     Saved to: {output_path}")
                
                # Show sample
                f.seek(0)
                print(f"\n📄 Sample qrels (first 3 lines):")
                for i, line in enumerate(f):
                    if i < 3:
                        print(f"     {line.strip()}")
                    else:
                        break
                
                return True
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    print(f"\n❌ Could not download qrels from any source")
    return False


def verify_compatibility(queries_path, qrels_path):
    """Verify that queries and qrels are compatible."""
    
    print("\n" + "="*80)
    print("Verifying Compatibility")
    print("="*80)
    
    # Load query IDs
    query_ids = set()
    with open(queries_path, 'r', encoding='utf-16') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                query_ids.add(str(q['query_id']))
    
    # Load qrels query IDs
    qrels_ids = set()
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qrels_ids.add(parts[0])
    
    common = query_ids & qrels_ids
    
    print(f"\nQuery IDs:")
    print(f"  Queries file: {len(query_ids)} queries")
    print(f"  Qrels file: {len(qrels_ids)} queries")
    print(f"  ✓ Common: {len(common)}")
    
    if len(common) == len(query_ids) == len(qrels_ids):
        print(f"\n✅ Perfect match! All {len(common)} queries are compatible.")
        return True
    else:
        print(f"\n⚠️  Some queries don't match")
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TREC-COVID Round 1 Data Download")
    print("="*80)
    print("\nThis script downloads:")
    print("  1. Round 1 queries (30 queries)")
    print("  2. Round 1 qrels (compatible with pyserini index)")
    
    # Download queries
    queries_success = download_round1_queries('queries-round1.jsonl')
    
    # Download qrels
    qrels_success = download_round1_qrels('qrels-round1.txt')
    
    if queries_success and qrels_success:
        # Verify compatibility
        verify_compatibility('queries-round1.jsonl', 'qrels-round1.txt')
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print("\nNext steps:")
        print("\n1. Run retrieval:")
        print("   python your_retrieval_script.py \\")
        print("     --query_path queries-round1.jsonl \\")
        print("     --bm25_output_file results/bm25.txt \\")
        print("     --rm3_output_file results/rm3.txt \\")
        print("     --qrf_output_file results/qrf.txt \\")
        print("     --k_values 1 5 10 20 50 100")
        
        print("\n2. Run evaluation:")
        print("   python task3_plot.py \\")
        print("     --qrels qrels-round1.txt \\")
        print("     --bm25 results/bm25.txt \\")
        print("     --rm3 results/rm3.txt \\")
        print("     --qrf results/qrf.txt \\")
        print("     --output_dir evaluation_plots")
        
    else:
        print("\n" + "="*80)
        print("❌ FAILED")
        print("="*80)
        print("\nManual download:")
        print("  Queries: https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml")
        print("  Qrels: https://ir.nist.gov/covidSubmit/data/qrels-covid_d1_j0.5-1.txt")