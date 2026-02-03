import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_engine import RAGEngine

def debug_rag():
    print("Loading RAG Engine...")
    rag = RAGEngine()
    
    # Test cases that failed in evaluation
    test_images = [
        "data/pokemon/images/pokemon_025.jpg", # Staryu -> Failed (Staraptor)
        "data/pokemon/images/pokemon_440.jpg", # Riolu -> Failed (Glaceon)
        "data/pokemon/images/pokemon_411.jpg", # Gastrodon -> Failed (Gallade)
        "data/pokemon/images/pokemon_117.jpg"  # Umbreon -> Success
    ]
    
    print("\n" + "="*50)
    print("DEBUGGING RAG RETRIEVAL")
    print("="*50)
    
    for img_path in test_images:
        print(f"\nTarget Image: {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            print("  [ERROR] File not found!")
            continue
            
        # Search
        results = rag.search(query=img_path, top_k=3) # Get top 3 to see candidates
        
        if not results or not results.get('documents'):
            print("  [RESULT] No documents found in Vector DB.")
            continue
            
        print("  [RETRIEVED DOCUMENTS]")
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            content = doc[:100] if doc else "[NO CONTENT]"
            meta = results['metadatas'][0][i] if results['metadatas'] else "{}"
            print(f"    Rank {i+1} (Dist: {dist:.4f}): Content={content} | Meta={meta}")

if __name__ == "__main__":
    debug_rag()
