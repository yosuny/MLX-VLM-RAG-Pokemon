import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.rag_engine import RAGEngine

def test_similarity():
    print("Initializing RAG Engine...")
    rag = RAGEngine()
    
    query_image = "assets/images/test_similarity.png"
    print(f"\n--- Testing Similarity Search with: {query_image} ---")
    
    if not os.path.exists(query_image):
        print(f"Error: Test image not found at {query_image}")
        return

    # Search for top 3 similar images
    results = rag.search(query_image, top_k=3)
    
    print("\n[Retrieval Results]")
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        
        # Check if the retrieved result is actually Bulbasaur (pokemon_000.jpg)
        is_bulbasaur = "Bulbasaur" in meta.get('caption', '') or "000" in doc_id
        status = "✅ MATCH" if is_bulbasaur else "❓ POSSIBLE MATCH"
        
        print(f"{status} | ID: {doc_id} | Dist: {dist:.4f}")
        print(f"   Caption: {meta.get('caption', 'No caption')[:100]}...")

if __name__ == "__main__":
    test_similarity()
