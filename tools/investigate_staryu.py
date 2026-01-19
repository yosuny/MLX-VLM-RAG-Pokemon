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

def investigate_staryu():
    print("="*70)
    print("INVESTIGATING STARYU (pokemon_025.jpg) ERROR")
    print("="*70)
    
    # 1. Check RAG Retrieval
    print("Loading RAG Engine...")
    rag = RAGEngine(db_path="./chroma_db")
    
    # Load allowed IDs
    allowed_ids = set()
    with open("data/pokemon/train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            if "images" in entry:
                allowed_ids.add(os.path.basename(entry["images"][0]))
    
    img_path = "data/pokemon/images/pokemon_025.jpg"
    print(f"\nSearching RAG for: {img_path}")
    
    results = rag.search(img_path, top_k=1, allowed_ids=allowed_ids)
    
    rag_caption = ""
    if results['ids'][0]:
        best_id = results['ids'][0][0]
        dist = results['distances'][0][0]
        rag_caption = results['metadatas'][0][0].get('caption', '')
        print(f"Top Match: {best_id}")
        print(f"Distance: {dist:.6f}")
        print(f"Caption: {rag_caption}")
        
        if best_id == "pokemon_025.jpg":
            print("✅ RAG correctly retrieved the exact image caption.")
        else:
            print("❌ RAG retrieved a different image.")
    else:
        print("❌ No results found.")
        return

    # 2. Run Inference with Hint
    print("\nLoading Vanilla Model...")
    model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-4bit", 
                           processor_config={"trust_remote_code": True})
    
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
        
    print("\nRunning Inference...")
    neutral_prompt = "What pokemon is this? Answer in English and Korean."
    rag_hint = f"\n\nHint: {rag_caption}"
    full_prompt = neutral_prompt + rag_hint
    
    print(f"Prompt with Hint:\n{full_prompt}\n")
    
    formatted_prompt = apply_chat_template(processor, config=model.config, 
                                          prompt=full_prompt, num_images=1)
    
    output = generate(model, processor, prompt=formatted_prompt, 
                    image=img_path, max_tokens=100, verbose=False)
    
    print(f"Model Output: {output.strip()}")

if __name__ == "__main__":
    investigate_staryu()
