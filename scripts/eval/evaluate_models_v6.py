"""
Evaluation v6: Generalization Test
Tests the Tuned model's ability to recognize known Pokemon from UNSEEN images.
"""

import json
import os
import sys
import random
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from src.rag_engine import RAGEngine
from datasets import load_dataset

# --- MONKEY PATCH FOR QWEN2-VL ---
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
# ---------------------------------

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

# Target Pokemon for generalization test (well-known Gen 1-2)
TARGET_POKEMON = [
    "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Jigglypuff",
    "Eevee", "Meowth", "Psyduck", "Snorlax", "Gengar",
    "Machamp", "Alakazam", "Gyarados", "Dragonite", "Mewtwo"
]

def prepare_v6_dataset():
    """Download alternative images for target Pokemon from PokeAPI (official sprites)."""
    print("Preparing v6 Generalization Dataset using PokeAPI sprites...")
    
    eval_dir = Path("data/pokemon/eval_v6_images")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # PokeAPI provides official artwork (different from training images)
    # We use the "official-artwork" or "dream_world" sprites as alternative images
    
    dataset = []
    
    for name in TARGET_POKEMON:
        name_lower = name.lower()
        img_path = eval_dir / f"{name_lower}.png"
        
        # Skip if already downloaded
        if img_path.exists():
            print(f"  [CACHED] {name}")
            dataset.append({
                "name": name,
                "image_path": str(img_path),
                "ground_truth": name,
            })
            continue
        
        try:
            # Get Pokemon data from PokeAPI
            api_url = f"https://pokeapi.co/api/v2/pokemon/{name_lower}"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            poke_data = response.json()
            
            # Use official-artwork first (always PNG)
            sprite_url = poke_data['sprites']['other'].get('official-artwork', {}).get('front_default')
            
            # Fallback to home sprites if official-artwork not available
            if not sprite_url:
                sprite_url = poke_data['sprites']['other'].get('home', {}).get('front_default')
            
            if not sprite_url:
                print(f"  [SKIP] {name}: No alternative sprite found")
                continue
            
            # Download and save sprite
            img_response = requests.get(sprite_url, timeout=10)
            img_response.raise_for_status()
            
            img = Image.open(BytesIO(img_response.content))
            
            # Convert to RGB if necessary (PNG might have alpha channel)
            if img.mode in ('RGBA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = background
            
            img.save(img_path)
            
            dataset.append({
                "name": name,
                "image_path": str(img_path),
                "ground_truth": name,
            })
            print(f"  [OK] {name}: Downloaded from PokeAPI")
            
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            continue
    
    print(f"\nv6 Dataset prepared: {len(dataset)} samples")
    return dataset

def evaluate_v6():
    """Run v6 Generalization Evaluation."""
    dataset = prepare_v6_dataset()
    results_file = "docs/reports/raw_data/v6_generalization_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    models_config = {
        "Vanilla": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
        "Tuned": "models/fused_qwen2_vl_4bit_quantized"
    }
    
    prompt = "What pokemon is this? Answer in English and Korean."
    
    # Load RAG Engine
    print("\nInitializing RAG Engine...")
    rag = RAGEngine(db_path="./chroma_db")
    
    # Load Allowed IDs for RAG (Gen 1-2 only)
    allowed_ids = set()
    with open("data/pokemon/train.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            if "images" in entry:
                allowed_ids.add(os.path.basename(entry["images"][0]))
    print(f"Loaded {len(allowed_ids)} Allowed IDs for RAG filtering.")
    
    # ---------------------------------------------------------
    # PASS 1: Vanilla & RAG
    # ---------------------------------------------------------
    print("\n=== PASS 1: Vanilla & RAG ===")
    model, processor = load(models_config["Vanilla"], processor_config={"trust_remote_code": True})
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
    
    for i, item in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] Processing {item['name']}")
        
        # Vanilla
        formatted_prompt = apply_chat_template(processor, config=model.config, prompt=prompt, num_images=1)
        out_vanilla = generate(model, processor, prompt=formatted_prompt, image=item['image_path'], max_tokens=100, verbose=False)
        item['vanilla_output'] = out_vanilla.strip()
        
        # RAG
        retrieved = rag.search(query=item['image_path'], top_k=1, allowed_ids=allowed_ids)
        rag_hint = ""
        rag_log = "No match"
        if retrieved['ids'][0]:
            match_caption = retrieved['metadatas'][0][0].get('caption', '')
            match_dist = retrieved['distances'][0][0]
            rag_hint = f"\n\nHint: {match_caption}"
            rag_log = f"(Dist: {match_dist:.2f}) {match_caption[:50]}..."
        
        full_prompt = prompt + rag_hint
        formatted_prompt_rag = apply_chat_template(processor, config=model.config, prompt=full_prompt, num_images=1)
        out_rag = generate(model, processor, prompt=formatted_prompt_rag, image=item['image_path'], max_tokens=100, verbose=False)
        
        item['rag_output'] = out_rag.strip()
        item['rag_retrieval_log'] = rag_log
    
    # Cleanup Pass 1
    del model, processor
    import gc
    gc.collect()
    
    # ---------------------------------------------------------
    # PASS 2: Tuned Model
    # ---------------------------------------------------------
    print("\n=== PASS 2: Tuned Model ===")
    model, processor = load(models_config["Tuned"], processor_config={"trust_remote_code": True})
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
    
    for i, item in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] Processing {item['name']}")
        formatted_prompt = apply_chat_template(processor, config=model.config, prompt=prompt, num_images=1)
        out_tuned = generate(model, processor, prompt=formatted_prompt, image=item['image_path'], max_tokens=100, verbose=False)
        item['tuned_output'] = out_tuned.strip()
    
    # Save raw results
    with open(results_file, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")
    
    # Generate Report
    generate_report(dataset)

def generate_report(dataset):
    """Generate v6 Generalization Report."""
    report_path = "docs/reports/EVALUATION_REPORT_v6_GENERALIZATION.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Calculate Accuracy
    def check_accuracy(output, target):
        return target.lower() in output.lower()
    
    vanilla_correct = sum(1 for d in dataset if check_accuracy(d['vanilla_output'], d['ground_truth']))
    rag_correct = sum(1 for d in dataset if check_accuracy(d['rag_output'], d['ground_truth']))
    tuned_correct = sum(1 for d in dataset if check_accuracy(d['tuned_output'], d['ground_truth']))
    
    total = len(dataset)
    
    with open(report_path, 'w') as f:
        f.write("# 🧬 Generalization Evaluation Report (v6)\n\n")
        f.write("**Goal**: Test the Tuned model's ability to recognize known Pokemon from **unseen images**.\n\n")
        f.write("## 📊 Summary Statistics\n\n")
        f.write(f"| Model | Accuracy (n={total}) |\n")
        f.write("| :--- | :---: |\n")
        f.write(f"| **Vanilla** | {vanilla_correct}/{total} ({vanilla_correct/total*100:.1f}%) |\n")
        f.write(f"| **RAG** | {rag_correct}/{total} ({rag_correct/total*100:.1f}%) |\n")
        f.write(f"| **Tuned** | {tuned_correct}/{total} ({tuned_correct/total*100:.1f}%) |\n\n")
        
        f.write("## 📋 Detailed Results\n\n")
        f.write("| Pokemon | Vanilla | RAG | Tuned |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for item in dataset:
            v_icon = "✅" if check_accuracy(item['vanilla_output'], item['ground_truth']) else "❌"
            r_icon = "✅" if check_accuracy(item['rag_output'], item['ground_truth']) else "❌"
            t_icon = "✅" if check_accuracy(item['tuned_output'], item['ground_truth']) else "❌"
            
            f.write(f"| **{item['name']}** | {v_icon} {item['vanilla_output'][:40]}... | {r_icon} {item['rag_output'][:40]}... | {t_icon} {item['tuned_output'][:40]}... |\n")
        
        f.write("\n## 🔍 Analysis\n\n")
        f.write("### Key Findings\n")
        if tuned_correct > vanilla_correct:
            f.write("- **Tuned model shows improvement** in recognizing known Pokemon from new images.\n")
        elif tuned_correct == vanilla_correct:
            f.write("- Tuned model performs **similarly** to Vanilla on unseen images.\n")
        else:
            f.write("- **Tuned model underperforms** Vanilla on unseen images, suggesting possible overfitting to training image features.\n")
        
        if rag_correct >= tuned_correct:
            f.write("- RAG remains the **most reliable** approach for generalization.\n")
        
        f.write("\n### RAG Retrieval Analysis\n")
        f.write("For each image, RAG searched for visually similar images in the Gen 1-2 training set.\n\n")
        for item in dataset:
            f.write(f"- **{item['name']}**: {item['rag_retrieval_log']}\n")
    
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    evaluate_v6()
