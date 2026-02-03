import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag_engine import RAGEngine

def test_rag(img_path):
    print(f"Loading RAG Engine and searching for: {img_path}")
    rag = RAGEngine(db_path="./chroma_db")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    results = rag.search(query=img_path, top_k=5)
    
    print("\n" + "="*50)
    print("RAG SEARCH RESULTS")
    print("="*50)
    
    if not results or not results.get('ids'):
        print("No results found.")
        return

    for i in range(len(results['ids'][0])):
        dist = results['distances'][0][i]
        meta = results['metadatas'][0][i]
        print(f"Rank {i+1} (Dist: {dist:.4f})")
        print(f"  Path: {meta.get('path')}")
        print(f"  Caption: {meta.get('caption')}")
        print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/test_rag_input.py <image_path>")
    else:
        test_rag(sys.argv[1])
