import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine
import json

def test_filtering():
    print("Testing RAG filtering...")
    rag = RAGEngine(db_path="./chroma_db")
    
    # Load Gen 1-2 IDs
    allowed_ids = set()
    with open("data/pokemon/train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            if "images" in entry:
                allowed_ids.add(os.path.basename(entry["images"][0]))
    
    print(f"\nAllowed IDs (Gen 1-2): {len(allowed_ids)} Pokemon")
    
    # Test with a Gen 3+ Pokemon (Piplup - pokemon_371.jpg)
    test_image = "data/pokemon/images/pokemon_371.jpg"
    
    print(f"\n{'='*60}")
    print(f"TEST: Searching for Piplup (Gen 4 - NOT in train set)")
    print(f"{'='*60}")
    
    print("\n1. WITHOUT filtering (should find Piplup):")
    results_unfiltered = rag.search(test_image, top_k=3, allowed_ids=None)
    for i, (idx, dist) in enumerate(zip(results_unfiltered['ids'][0], results_unfiltered['distances'][0])):
        meta = results_unfiltered['metadatas'][0][i]
        caption = meta.get('caption', 'N/A')[:80]
        print(f"   Rank {i+1}: {idx} (dist: {dist:.4f}) - {caption}...")
    
    print("\n2. WITH filtering (should find Gen 1-2 similar Pokemon only):")
    results_filtered = rag.search(test_image, top_k=3, allowed_ids=allowed_ids)
    if results_filtered['ids'][0]:
        for i, (idx, dist) in enumerate(zip(results_filtered['ids'][0], results_filtered['distances'][0])):
            meta = results_filtered['metadatas'][0][i]
            caption = meta.get('caption', 'N/A')[:80]
            print(f"   Rank {i+1}: {idx} (dist: {dist:.4f}) - {caption}...")
    else:
        print("   No results (all filtered out)")
    
    print(f"\n{'='*60}")
    print("✅ Filtering test complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_filtering()
