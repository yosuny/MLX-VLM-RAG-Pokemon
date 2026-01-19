from datasets import load_dataset
import os
import sys
import json
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.pokemon_info import POKEMON_DB
except ImportError:
    print("Warning: pokemon_info.py not found. Run build_metadata.py first.")
    POKEMON_DB = {}

def setup_pokemon_data(num_samples=None, output_dir="data/pokemon"):
    print(f"Downloading samples from diffusers/pokemon-gpt4-captions...")
    
    ds = load_dataset("diffusers/pokemon-gpt4-captions", split="train", streaming=True)
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    train_entries = []
    valid_entries = []
    
    # Pre-compute lowercase name map for mapped lookup
    name_lookup = {v['name_en'].lower(): k for k, v in POKEMON_DB.items() if isinstance(k, str)}
    
    count_train = 0
    count_valid = 0

    for i, item in enumerate(ds):
        if num_samples and i >= num_samples:
            break
            
        image = item["image"]
        caption = item["text"]
        
        # Identify Pokemon
        found_id = None
        found_meta = None
        
        # Heuristic: Check if any known Pokemon name is in the caption
        # Sort names by length descending to match "Mega Charizard" before "Charizard"
        sorted_names = sorted(name_lookup.keys(), key=len, reverse=True)
        
        import re
        for name in sorted_names:
            # Use regex to match whole words only to avoid matching "Natu" in "signature"
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, caption.lower()):
                found_id = name_lookup[name]
                found_meta = POKEMON_DB[name]
                break
        
        # Save image
        image_name = f"pokemon_{i:03d}.jpg"
        image_path = os.path.join(images_dir, image_name)
        image.save(image_path)
        
        # Formulate Answer
        if found_meta:
            name_en = found_meta['name_en']
            name_kr = found_meta['name_kr']
            gen = found_meta['generation'] # e.g., "generation-i"
            
            # Format readable generation
            gen_readable = gen.replace("generation-", "Gen ").upper()
            
            answer_text = f"This is {name_en} ({name_kr}). It is a {gen_readable} Pokemon. {caption}"
            
            # Split Logic
            # Train: Gen 1, Gen 2
            # Valid: Gen 3+
            if gen in ["generation-i", "generation-ii"]:
                target_list = train_entries
                count_train += 1
                split_name = "TRAIN"
            else:
                target_list = valid_entries
                count_valid += 1
                split_name = "VALID"
        else:
            # Fallback: Unknown Pokemon (or not in our DB) -> Default to Train or skip?
            # Let's put in Valid to be safe/noisy? Or Train.
            # Default to Train for now.
            answer_text = caption
            target_list = train_entries
            count_train += 1
            split_name = "TRAIN(Unk)"
        
        entry = {
            "images": [image_path],
            "messages": [
                {
                    "role": "user", 
                    "content": "What pokemon is this?"
                },
                {
                    "role": "assistant", 
                    "content": answer_text
                }
            ]
        }
        
        target_list.append(entry)
        if i % 50 == 0:
            print(f"[{i}] {split_name}: {caption[:30]}...")

    # Save JSONLs
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")
            
    with open(valid_path, "w") as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nData preparation complete!")
    print(f"Total: {count_train + count_valid}")
    print(f"Train (Gen 1-2): {count_train} -> {train_path}")
    print(f"Valid (Gen 3+): {count_valid} -> {valid_path}")

if __name__ == "__main__":
    setup_pokemon_data()
