import json
import os
import sys
import gc
import random
import glob

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from src.rag_engine import RAGEngine

# --- MONKEY PATCH FOR QWEN2-VL PADDING BUG ---
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model
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
# ---------------------------------------------

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


def prepare_dataset(output_path="data/eval_v4.json"):
    print("Selecting 4 random samples (2 Train, 2 Valid)...")
    
    def load_jsonl(path, tag):
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['split'] = tag
                item['image_path'] = item['images'][0].replace("data_pokemon/", "data/pokemon/")
                data.append(item)
        return data

    train_data = load_jsonl("data/pokemon/train.jsonl", "train")
    valid_data = load_jsonl("data/pokemon/validation.jsonl", "valid")
    
    random.seed(42) # Fixed seed for reproducibility
    selected = random.sample(train_data, 2) + random.sample(valid_data, 2)
    
    for i, item in enumerate(selected):
        item['id'] = f"sample_{i:02d}"
    
    with open(output_path, 'w') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
        
    return selected

def evaluate_v4():
    dataset_path = "data/eval_v4.json"
    if not os.path.exists(dataset_path):
        dataset = prepare_dataset(dataset_path)
    else:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

    # HINTED PROMPT (New for v4)
    target_prompt = "What pokemon is this? Answer in English and Korean."
    
    # -------------------------------------------------------------
    # PASS 1: VANILLA & RAG EVALUATION (Base Model)
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("PASS 1: VANILLA & RAG EVALUATION (Base Model)")
    print("="*80)
    
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    print(f"Loading Base Model: {model_path}")
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
        
    print("Loading RAG Engine...")
    rag = RAGEngine()

    for i, item in enumerate(dataset):
        print(f"[{i+1}/4] {item['split'].upper()}: {item['image_path']}")
        image_path = item['image_path']
        
        # 1. Vanilla
        prompt_vanilla = apply_chat_template(processor, config=model.config, prompt=target_prompt, num_images=1)
        out_vanilla = generate(model, processor, prompt=prompt_vanilla, image=image_path, max_tokens=100, verbose=False)
        item['vanilla_result'] = out_vanilla
        
        # 2. RAG
        retrieved = rag.search(query=image_path, top_k=1)
        rag_context = ""
        if retrieved and retrieved.get('metadatas') and len(retrieved['metadatas'][0]) > 0:
            doc = retrieved['metadatas'][0][0].get('caption', '')
            rag_context = f"\n\nHint: {doc}"
        
        prompt_rag = apply_chat_template(processor, config=model.config, prompt=target_prompt + rag_context, num_images=1)
        out_rag = generate(model, processor, prompt=prompt_rag, image=image_path, max_tokens=100, verbose=False)
        item['rag_result'] = out_rag
        
        print(f"  Vanilla: {out_vanilla.strip()[:50]}...")
        print(f"  RAG:     {out_rag.strip()[:50]}...")

        # Save Progress
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Clean up Pass 1
    del model
    del processor
    del rag
    gc.collect()
    try: mx.metal.clear_cache()
    except: pass
    print("Memory cleared.")

    # -------------------------------------------------------------
    # PASS 2: TUNED FUSED MODEL
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("PASS 2: TUNED FUSED MODEL EVALUATION")
    print("="*80)

    tuned_model_path = "models/fused_qwen2_vl_4bit_quantized"
    print(f"Loading Tuned Model: {tuned_model_path}")
    model, processor = load(tuned_model_path, processor_config={"trust_remote_code": True})
    
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
        
    for i, item in enumerate(dataset):
        print(f"[{i+1}/4] {item['split'].upper()}: {item['image_path']}")
        image_path = item['image_path']
        
        prompt_tuned = apply_chat_template(processor, config=model.config, prompt=target_prompt, num_images=1)
        out_tuned = generate(model, processor, prompt=prompt_tuned, image=image_path, max_tokens=100, verbose=False)
        item['tuned_result'] = out_tuned
        
        print(f"  Tuned:   {out_tuned.strip()[:50]}...")

        # Save Progress
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Generate Report
    report_path = "EVALUATION_REPORT_v4.md"
    with open(report_path, "w") as f:
        f.write("# Evaluation Report v4: Hinted Prompt Comparison\n\n")
        f.write(f"**Prompt**: `{target_prompt}`\n")
        f.write("**Configurations**:\n")
        f.write("1. **Vanilla**: Base Qwen2-VL-7B-Instruct (4-bit)\n")
        f.write("2. **RAG**: Base + Vector Retrieval Hint\n")
        f.write("3. **Tuned**: Custom Fine-tuned + Fused + Quantized Model (Ours)\n\n")
        f.write("| Type | Image | Vanilla | RAG | Tuned (Ours) |\n")
        f.write("| :---: | :---: | --- | --- | --- |\n")
        
        for item in dataset:
            img_md = f"![](../{item['image_path']})<br><sub>{item['split']}</sub>"
            vanilla = item.get('vanilla_result', '').strip().replace('\n', ' ')
            rag = item.get('rag_result', '').strip().replace('\n', ' ')
            tuned = item.get('tuned_result', '').strip().replace('\n', ' ')
            
            f.write(f"| **{item['split'].upper()}** | {img_md} | {vanilla} | {rag} | **{tuned}** |\n")

    print(f"\nEvaluation Complete! Report saved to {report_path}")

if __name__ == "__main__":
    evaluate_v4()
