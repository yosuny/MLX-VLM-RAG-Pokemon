import json
import os
import sys
import random
import glob
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from src.rag_engine import RAGEngine

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

def load_dataset_entries(jsonl_path):
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Normalize image path
                if 'images' in data:
                    data['image_path'] = data['images'][0].replace("data_pokemon/", "data/pokemon/")
                elif 'image' in data:
                     data['image_path'] = data['image'].replace("data_pokemon/", "data/pokemon/")
                
                # Extract caption (Ground Truth)
                if 'messages' in data:
                    for msg in data['messages']:
                        if msg['role'] == 'assistant':
                            data['ground_truth'] = msg['content']
                            break
                elif 'text' in data:
                    data['ground_truth'] = data['text']
                
                entries.append(data)
            except:
                continue
    return entries

def prepare_ood_trap_test():
    print("Preparing OOD Trap Test (30 samples)...")
    train_entries = load_dataset_entries("data/pokemon/train.jsonl")
    valid_entries = load_dataset_entries("data/pokemon/validation.jsonl")
    
    # 1. Select 24 Random Trained Samples
    random.seed(42)
    trained_samples = random.sample(train_entries, 24)
    for s in trained_samples:
        s['type'] = 'TRAINED (Gen 1-2)'
        
    # 2. Select 6 Specific Untrained "Trap" Samples
    trap_keywords = ["Electivire", "Munchlax", "Glaceon", "Lickilicky", "Togekiss", "Leafeon"]
    untrained_samples = []
    
    for kw in trap_keywords:
        found = False
        for entry in valid_entries:
            if kw in entry.get('ground_truth', ''):
                entry['type'] = 'UNTRAINED (Gen 3+ Trap)'
                untrained_samples.append(entry)
                found = True
                break # Take the first match
        if not found:
            print(f"Warning: Could not find trap sample for {kw}")
            
    ood_trap_test = trained_samples + untrained_samples
    print(f"OOD Trap Test prepared: {len(ood_trap_test)} samples ({len(trained_samples)} trained, {len(untrained_samples)} traps)")
    return ood_trap_test

def load_existing_results(results_file):
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_results(results_file, dataset):
    with open(results_file, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Progress saved to {results_file}")

def evaluate_ood_trap_test():
    dataset = prepare_ood_trap_test()
    results_file = "docs/reports/raw_data/ood_trap_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Models to test
    models_config = {
        "Vanilla": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
        # "RAG": handled via Vanilla model + special logic
        "Tuned": "models/fused_qwen2_vl_4bit_quantized"
    }

    prompt = "What pokemon is this? Answer in English and Korean."
    
    # Load existing results if available
    existing_dataset = load_existing_results(results_file)
    if existing_dataset and len(existing_dataset) == len(dataset):
        print("Loaded existing intermediate results. Resuming...")
        dataset = existing_dataset
    else:
        print("Starting fresh evaluation...")

    # Load RAG Engine once
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
    # PASS 1: Vanilla & RAG (Using Base Model)
    # ---------------------------------------------------------
    print("\n=== PASS 1: Vanilla & RAG ===")
    
    # Check if Pass 1 is already complete for all items
    pass1_needed = any('vanilla_output' not in item for item in dataset)
    
    if pass1_needed:
        model, processor = load(models_config["Vanilla"], processor_config={"trust_remote_code": True})
        if hasattr(processor, "image_processor"):
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
            
        for i, item in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] Processing {os.path.basename(item['image_path'])} ({item['type']})")
            
            if 'vanilla_output' in item and 'rag_output' in item:
                 print(f"Skipping {os.path.basename(item['image_path'])} (Already processed)")
                 continue

            # Vanilla Inference
            formatted_prompt = apply_chat_template(processor, config=model.config, prompt=prompt, num_images=1)
            out_vanilla = generate(model, processor, prompt=formatted_prompt, image=item['image_path'], max_tokens=100, verbose=False)
            item['vanilla_output'] = out_vanilla.strip()
            
            # RAG Inference
            # 1. Retrieve
            retrieved = rag.search(query=item['image_path'], top_k=1, allowed_ids=allowed_ids)
            rag_hint = ""
            rag_log = "No match"
            if retrieved['ids'][0]:
                match_caption = retrieved['metadatas'][0][0].get('caption', '')
                match_dist = retrieved['distances'][0][0]
                rag_hint = f"\n\nHint: {match_caption}"
                rag_log = f"(Dist: {match_dist:.2f}) {match_caption[:40]}..."
            
            # 2. Generate
            full_prompt = prompt + rag_hint
            formatted_prompt_rag = apply_chat_template(processor, config=model.config, prompt=full_prompt, num_images=1)
            out_rag = generate(model, processor, prompt=formatted_prompt_rag, image=item['image_path'], max_tokens=100, verbose=False)
            
            item['rag_output'] = out_rag.strip()
            item['rag_retrieval_log'] = rag_log
            
            # Checkpoint save every 5 items
            if (i + 1) % 5 == 0:
                save_results(results_file, dataset)

        # Cleanup Pass 1
        save_results(results_file, dataset) # Final save for Pass 1
        del model, processor
        import gc
        gc.collect()
    else:
        print("Pass 1 already completed. Skipping model load.")

    # ---------------------------------------------------------
    # PASS 2: Tuned Model
    # ---------------------------------------------------------
    print("\n=== PASS 2: Tuned Model ===")
    # Check if Pass 2 is already complete
    pass2_needed = any('tuned_output' not in item for item in dataset)
    
    if pass2_needed:
        model, processor = load(models_config["Tuned"], processor_config={"trust_remote_code": True})
        if hasattr(processor, "image_processor"):
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
            
        for i, item in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] Processing {os.path.basename(item['image_path'])}")
            
            if 'tuned_output' in item:
                 print(f"Skipping {os.path.basename(item['image_path'])} (Already processed)")
                 continue

            formatted_prompt = apply_chat_template(processor, config=model.config, prompt=prompt, num_images=1)
            out_tuned = generate(model, processor, prompt=formatted_prompt, image=item['image_path'], max_tokens=100, verbose=False)
            item['tuned_output'] = out_tuned.strip()
            
            # Checkpoint save everyone 2 items (Tuned is slower)
            if (i + 1) % 2 == 0:
                save_results(results_file, dataset)

    # Final Save
    save_results(results_file, dataset)
    
    # Generate Report
    generate_report_markdown(dataset)

def generate_report_markdown(dataset):
    report_path = "docs/reports/EVALUATION_REPORT_v5_OOD_TRAP_TEST.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# 🏆 OOD Trap Test Evaluation Report\n\n")
        f.write("**Evaluation Configuration**:\n")
        f.write("- **Total Samples:** 30\n")
        f.write("- **Trained (Gen 1-2):** 24 (80%)\n")
        f.write("- **Untrained Val (Gen 3+):** 6 (20%) - Curated Trap Cases\n\n")
        
        # Section 1: Untrained Trap Analysis (The most important part)
        f.write("## ⚠️ Untrained Data Analysis ('Trap' Cases)\n")
        f.write("Evaluation on Gen 3+ Pokemon that are evolutionary relatives or look-alikes of Gen 1-2.\n\n")
        f.write("| Image | Ground Truth | Vanilla | RAG (Gen 1-2 Filtered) | Tuned |\n")
        f.write("| :---: | :--- | :--- | :--- | :--- |\n")
        
        untrained = [d for d in dataset if "UNTRAINED" in d['type']]
        for item in untrained:
            img_filename = os.path.basename(item['image_path'])
            img_md = f"![]({item['image_path'].replace('data/pokemon/', '../../data/pokemon/')})<br>`{img_filename}`"
            
            # Extract simple name from GT for display
            gt_short = item['ground_truth'].split('.')[0] if '.' in item['ground_truth'] else item['ground_truth'][:30]
            
            f.write(f"| {img_md} | **{gt_short}** | {item['vanilla_output']} | {item['rag_output']}<br>*(Retrieved: {item.get('rag_retrieval_log', '')})* | {item['tuned_output']} |\n")
            
        f.write("\n## ✅ Trained Data Analysis (Sample of 5)\n")
        f.write("Checking retention of Gen 1-2 knowledge.\n\n")
        f.write("| Image | Ground Truth | Vanilla | RAG | Tuned |\n")
        f.write("| :---: | :--- | :--- | :--- | :--- |\n")
        
        trained = [d for d in dataset if "TRAINED" in d['type']][:5] # Show first 5
        for item in trained:
            img_filename = os.path.basename(item['image_path'])
            img_md = f"![]({item['image_path'].replace('data/pokemon/', '../../data/pokemon/')})<br>`{img_filename}`"
            gt_short = item['ground_truth'].split('.')[0]
            
            f.write(f"| {img_md} | **{gt_short}** | {item['vanilla_output']} | {item['rag_output']} | **{item['tuned_output']}** |\n")

    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    evaluate_ood_trap_test()
