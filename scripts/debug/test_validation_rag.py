import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_engine import RAGEngine
import json

def test_validation_rag():
    print("Loading RAG Engine...")
    rag = RAGEngine(db_path="./chroma_db")
    
    # Load Gen 1-2 allowed IDs
    allowed_ids = set()
    with open("data/pokemon/train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            if "images" in entry:
                allowed_ids.add(os.path.basename(entry["images"][0]))
    
    print(f"Loaded {len(allowed_ids)} Gen 1-2 Pokemon IDs for filtering.\n")
    
    # Validation images from evaluation reports
    validation_images = [
        ("pokemon_440.jpg", "Riolu", "data/pokemon/images/pokemon_440.jpg"),
        ("pokemon_411.jpg", "Gastrodon", "data/pokemon/images/pokemon_411.jpg")
    ]
    
    print("="*70)
    print("RAG RESULTS FOR VALIDATION SET (Gen 1-2 Filtering Applied)")
    print("="*70)
    
    for img_id, name, img_path in validation_images:
        print(f"\n### {img_id} ({name}) ###")
        
        # Search with filtering
        results = rag.search(img_path, top_k=1, allowed_ids=allowed_ids)
        
        if results['ids'][0]:
            best_id = results['ids'][0][0]
            best_dist = results['distances'][0][0]
            best_meta = results['metadatas'][0][0]
            best_caption = best_meta.get('caption', 'N/A')
            
            print(f"Top Match: {best_id}")
            print(f"Distance: {best_dist:.4f}")
            print(f"Caption: {best_caption[:150]}...")
            
            # Extract Pokemon name from caption
            if "This is" in best_caption:
                # Format: "This is PokemonName (한글이름). ..."
                parts = best_caption.split("This is")[1].split(".")[0].strip()
                print(f"\n✅ RAG Result: {parts}")
        else:
            print("❌ No results found (all filtered out)")
        print("-" * 70)

if __name__ == "__main__":
    test_validation_rag()
