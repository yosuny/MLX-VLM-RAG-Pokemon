import os
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoImageProcessor
from rag_engine import RAGEngine

# Configuration
# Use Repo ID instead of absolute path so HF hub resolves the snapshot correctly
MODELS_DIR = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
ADAPTER_PATH = "adapters"
TEST_IMAGES = [
    "data_pokemon/images/pokemon_000.jpg",
    "data_pokemon/images/pokemon_024.jpg", # Likely Pikachu or Arbok depending on index, will see in report
    "data_pokemon/images/pokemon_043.jpg"
]
REPORT_FILE = "EVALUATION_REPORT_v4.md"

# PATCH: explicit slow processor loading to avoid MLX/Torch tensor mismatch
def load_model_safe(model_path, adapter_path=None):
    print(f"Loading model from {model_path} (Adapter: {adapter_path})...")
    
    if adapter_path:
        model, processor = load(model_path, adapter_path=adapter_path, processor_config={"trust_remote_code": True})
    else:
        model, processor = load(model_path, processor_config={"trust_remote_code": True})

    # Force slow processor usage pattern
    try:
        # We manually load the slow processor and inject it if possible, 
        # but mlx_vlm.load usually returns a Qwen2VLProcessor which wraps the image processor.
        # For inference, the standard generate() function might handle it, but let's be safe.
        patched_image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if hasattr(processor, "image_processor"):
            processor.image_processor = patched_image_processor
            print(" injected slow image_processor into processor")
    except Exception as e:
        print(f"Warning: Failed to patch image processor: {e}")
        
    return model, processor

def run_inference(model, processor, image_path, prompt="What is this pokemon's name? Answer in English and Korean."):
    if not os.path.exists(image_path):
        return "Error: Image not found"
        
    formatted_prompt = apply_chat_template(
        processor, 
        config=model.config.__dict__,
        prompt=[
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]},
        ], 
        num_images=1
    )
    
    # Generate
    output = generate(model, processor, formatted_prompt, verbose=False, max_tokens=100)
    return output.strip()

def main():
    results = {}
    
    # 0. Initialize RAG
    print("Initializing RAG Index...")
    rag = RAGEngine()
    
    # 1. Test Vanilla Model
    model, processor = load_model_safe(MODELS_DIR)
    
    print("Evaluating Vanilla Model...")
    results['vanilla'] = {}
    for img in TEST_IMAGES:
        res = run_inference(model, processor, img)
        results['vanilla'][img] = res
        print(f" > {img}: {res}")
        
    # 2. Test RAG Retrieval (We evaluate retrieval accuracy visually in report)
    print("Evaluating RAG Retrieval...")
    results['rag'] = {}
    for img in TEST_IMAGES:
        search_res = rag.search(img, top_k=2) # Fetch top 2 (self + neighbor)
        
        # ChromaDB returns nested lists
        # ids=[['id1', ...]], distances=[[0.0, ...]], metadatas=[[{'path':...}, ...]]
        
        found = False
        retrieved_path = "None"
        dist = 0.0
        
        if search_res and search_res['ids'] and len(search_res['ids']) > 0:
            # We want the nearest neighbor that is NOT the image itself (distance > 0 ideally, or just the second one if distance is 0)
            # Since we are querying with the exact image from the set, the detailed result will be the image itself (dist~0).
            # RAG usually wants *similar* images, not identical.
            # So let's pick the second result if available, or the first if only 1 exists (which shouldn't happen with top_k=2).
            
            ids = search_res['ids'][0]
            dists = search_res['distances'][0]
            metas = search_res['metadatas'][0]
            
            if len(ids) >= 1:
                # Pick the first one for display
                retrieved_meta = metas[0]
                retrieved_path = retrieved_meta.get('path', 'None')
                dist = dists[0]
                found = True
                
        if found:
            results['rag'][img] = {'path': retrieved_path, 'distance': dist, 'metadata': retrieved_meta}
        else:
            results['rag'][img] = {'path': 'None', 'distance': 0.0, 'metadata': {}}

    # Free memory
    del model
    del processor
    
    # 3. Test Tuned Model
    print("Evaluating Tuned Model...")
    model, processor = load_model_safe(MODELS_DIR, adapter_path=ADAPTER_PATH)
    
    results['tuned'] = {}
    for img in TEST_IMAGES:
        res = run_inference(model, processor, img)
        results['tuned'][img] = res
        print(f" > {img}: {res}")

    # 4. Generate Markdown Report
    
    print(f"Generating {REPORT_FILE}...")
    with open(REPORT_FILE, "w") as f:
        f.write("# 🕵️ VLM Evaluation Report: Pokemon Identification\n")
        f.write(f"**Report Version**: Final (Ground Truth Comparison)\n\n")
        f.write("Comparing Vanilla Qwen2-VL, RAG Retrieval, and LoRA Tuned Model.\n\n")
        
        for img in TEST_IMAGES:
            img_name = os.path.basename(img)
            vanilla_ans = results['vanilla'].get(img, "N/A")
            tuned_ans = results['tuned'].get(img, "N/A")
            rag_info = results['rag'].get(img, {})
            rag_img = rag_info.get('path', '')
            
            # Extract Ground Truth from RAG metadata
            rag_caption_full = rag_info.get('metadata', {}).get('caption', '')
            ground_truth_display = "Unknown"
            
            # Simple parser for "This is Name (KoreanName)"
            gt_en = ""
            gt_kr = ""
            if "This is " in rag_caption_full:
                try:
                    # Expected format: "This is Bulbasaur (이상해씨)."
                    parts = rag_caption_full.replace("This is ", "").split("(")
                    if len(parts) >= 2:
                        gt_en = parts[0].strip()
                        gt_kr = parts[1].split(")")[0].strip()
                        ground_truth_display = f"**{gt_en} ({gt_kr})**"
                except:
                    ground_truth_display = rag_caption_full[:50] + "..."
            elif rag_caption_full:
                ground_truth_display = rag_caption_full[:50] + "..."

            # Helper to check correctness
            def check_answer(ans, en, kr):
                if not ans or ans.startswith("!"): return "⚠️ Error" # Tuned model failure
                ans_lower = ans.lower()
                is_correct = False
                if en and en.lower() in ans_lower: is_correct = True
                if kr and kr in ans: is_correct = True
                
                return "✅ Correct" if is_correct else "❌ Incorrect"

            # Evaluate Models
            vanilla_status = check_answer(vanilla_ans, gt_en, gt_kr)
            tuned_status = check_answer(tuned_ans, gt_en, gt_kr)
            
            f.write(f"## Test Case: {img_name}\n")
            f.write(f"| Query Image | Model Results |\n")
            f.write(f"| :---: | --- |\n")
            f.write(f"| ![Query]({img}) | **Ground Truth**: {ground_truth_display} <br><br> **Vanilla Model**: {vanilla_status} <br> *\"{vanilla_ans}\"* <br><br> **Tuned Model**: {tuned_status} <br> *\"{tuned_ans}\"* |\n\n")
            
            f.write(f"### RAG Context\n")
            if rag_img and os.path.exists(rag_img):
                # Extract caption from metadata if available
                rag_caption = rag_info.get('metadata', {}).get('caption', 'No caption available')
                # Truncate if too long
                if len(rag_caption) > 100: rag_caption = rag_caption[:100] + "..."
                
                f.write(f"Retrieved Similar Image (Dist: {rag_info.get('distance',0):.4f}):\n")
                f.write(f"**Info**: *{rag_caption}*\n\n")
                f.write(f"![RAG]({rag_img})\n\n")
            else:
                f.write("No similar image retrieved.\n\n")
                
            f.write("---\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
