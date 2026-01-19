import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from src.rag_engine import RAGEngine
import json

# Monkey patch for Qwen2-VL
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model
import mlx.core as mx

def _merge_input_ids_with_image_features_patched(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index
    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index
    image_features = image_features.astype(mx.float32)
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    if pad_size < 0:
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    inputs_embeds = mx.where(image_positions[:, :, None], image_features, inputs_embeds)
    return inputs_embeds

Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched

class RobustImageProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor
        if hasattr(processor, "image_processor"):
            self.processor = processor.image_processor
        for attr in dir(self.processor):
            if not attr.startswith("__"):
                try: setattr(self, attr, getattr(self.processor, attr))
                except: pass
    def __call__(self, images=None, text=None, **kwargs):
        if "return_tensors" in kwargs: kwargs["return_tensors"] = "pt"
        out = self.processor(images, text, **kwargs)
        for k, v in out.items():
            if hasattr(v, "numpy"): out[k] = v.numpy()
            elif isinstance(v, list) and hasattr(v[0], "numpy"): out[k] = [x.numpy() for x in v]
        return out
    def preprocess(self, images, **kwargs):
        if "return_tensors" in kwargs: kwargs["return_tensors"] = "pt"
        out = self.processor.preprocess(images, **kwargs)
        if hasattr(out, "pixel_values") and hasattr(out["pixel_values"], "numpy"):
            out["pixel_values"] = out["pixel_values"].numpy()
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(v, "numpy"): out[k] = v.numpy()
        return out
    def __getattr__(self, name): return getattr(self.processor, name)

def reevaluate_validation_rag():
    print("="*70)
    print("RE-EVALUATING VALIDATION SET RAG RESULTS")
    print("="*70)
    
    # Load Gen 1-2 allowed IDs
    allowed_ids = set()
    with open("data/pokemon/train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            if "images" in entry:
                allowed_ids.add(os.path.basename(entry["images"][0]))
    
    print(f"Loaded {len(allowed_ids)} Gen 1-2 Pokemon IDs for filtering.\n")
    
    # Initialize RAG
    print("Loading RAG Engine...")
    rag = RAGEngine(db_path="./chroma_db")
    
    # Validation images
    validation_cases = [
        {
            "image": "data/pokemon/images/pokemon_440.jpg",
            "name": "Riolu (Gen 4)",
            "v3_prompt": "What is this? Answer in English and Korean.",
            "v4_prompt": "What pokemon is this? Answer in English and Korean."
        },
        {
            "image": "data/pokemon/images/pokemon_411.jpg",
            "name": "Gastrodon (Gen 4)",
            "v3_prompt": "What is this? Answer in English and Korean.",
            "v4_prompt": "What pokemon is this? Answer in English and Korean."
        }
    ]
    
    # Get RAG captions
    for case in validation_cases:
        results = rag.search(case["image"], top_k=1, allowed_ids=allowed_ids)
        if results['ids'][0]:
            caption = results['metadatas'][0][0].get('caption', 'N/A')
            case["rag_caption"] = caption
            print(f"✓ {case['name']}: {caption[:60]}...")
        else:
            case["rag_caption"] = ""
    
    # Load Vanilla Model
    print("\nLoading Vanilla Model (mlx-community/Qwen2-VL-7B-Instruct-4bit)...")
    model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-4bit", 
                           processor_config={"trust_remote_code": True})
    
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
    
    print("\n" + "="*70)
    print("RUNNING RAG INFERENCE (Vanilla + RAG Hints)")
    print("="*70)
    
    results = {"v3": [], "v4": []}
    
    for case in validation_cases:
        print(f"\n### {case['name']} ###")
        image_path = case["image"]
        rag_hint = f"\n\nHint: {case['rag_caption']}"
        
        # v3 (Generic)
        prompt_v3 = case["v3_prompt"] + rag_hint
        formatted_v3 = apply_chat_template(processor, config=model.config, 
                                          prompt=prompt_v3, num_images=1)
        output_v3 = generate(model, processor, prompt=formatted_v3, 
                           image=image_path, max_tokens=100, verbose=False)
        
        print(f"  v3 (Generic + RAG): {output_v3.strip()[:80]}...")
        results["v3"].append({
            "image": case["image"],
            "name": case["name"],
            "output": output_v3.strip()
        })
        
        # v4 (Hinted)
        prompt_v4 = case["v4_prompt"] + rag_hint
        formatted_v4 = apply_chat_template(processor, config=model.config,
                                          prompt=prompt_v4, num_images=1)
        output_v4 = generate(model, processor, prompt=formatted_v4,
                           image=image_path, max_tokens=100, verbose=False)
        
        print(f"  v4 (Hinted + RAG):  {output_v4.strip()[:80]}...")
        results["v4"].append({
            "image": case["image"],
            "name": case["name"],
            "output": output_v4.strip()
        })
    
    # Save results
    with open("rag_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print("\nResults saved to: rag_validation_results.json")
    print("\nv3 Results:")
    for r in results["v3"]:
        print(f"  - {r['name']}: {r['output'][:60]}...")
    
    print("\nv4 Results:")
    for r in results["v4"]:
        print(f"  - {r['name']}: {r['output'][:60]}...")

if __name__ == "__main__":
    reevaluate_validation_rag()
