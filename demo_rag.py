from rag_engine import RAGEngine
import os
import glob

def run_demo():
    rag = RAGEngine()
    
    # 1. Index Images (from Pokemon data)
    image_dir = "data_pokemon/images"
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found. Run setup_pokemon_data.py first.")
        return

    images = glob.glob(os.path.join(image_dir, "*.jpg")) # Note: jpg for pokemon data
    if not images:
        print("No images found.")
        return
        
    # Load captions from JSONL to use as metadata
    jsonl_path = "data_pokemon/train.jsonl"
    caption_map = {}
    if os.path.exists(jsonl_path):
        import json
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Support both formats we might have generated
                    # 1. {"images": ["path"], "messages": [...]}
                    # 2. {"image": "path", "text": "..."}
                    
                    img_path = ""
                    caption = ""
                    
                    if "images" in entry:
                        img_path = entry["images"][0]
                        # Extract assistant answer
                        for msg in entry["messages"]:
                            if msg["role"] == "assistant":
                                caption = msg["content"]
                                break
                    elif "image" in entry:
                        img_path = entry["image"]
                        caption = entry.get("text", "")
                        
                    if img_path:
                        filename = os.path.basename(img_path)
                        caption_map[filename] = caption
                except:
                    continue
    
    # Prepare metadata list corresponding to 'images' list
    metadatas = []
    for img_path in images:
        fname = os.path.basename(img_path)
        caption = caption_map.get(fname, "Unknown")
        metadatas.append({"path": img_path, "caption": caption})
        
    print(f"Found {len(images)} images. Indexing with captions...")
    rag.index_images(images, custom_metadatas=metadatas)
    
    # 2. Search
    queries = [
        "a yellow mouse like pokemon", # Should match Pikachu-like
        "a fire dragon", 
        "something blue and water type",
        "A fierce Rhydon" # Exact caption match
    ]
    
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        results = rag.search(q, top_k=1)
        
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            print(f"Found: {doc_id} (Path: {meta['path']}, dist: {dist:.4f})")

if __name__ == "__main__":
    run_demo()
