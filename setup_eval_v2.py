import json
import os
import random
import shutil
from PIL import Image

def setup_eval_v2(output_dir="eval_v2"):
    print("Setting up Evaluation v2 Dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load Data
    with open("data_pokemon/train.jsonl", "r") as f:
        train_data = [json.loads(line) for line in f]
        
    with open("data_pokemon/validation.jsonl", "r") as f:
        valid_data = [json.loads(line) for line in f]
        
    # Sample 7 from Train, 3 from Valid
    # Use fixed seed for reproducibility
    random.seed(42)
    selected_train = random.sample(train_data, 7)
    selected_valid = random.sample(valid_data, 3)
    
    eval_dataset = []
    
    def process_sample(entry, split_name, index):
        src_image_path = entry["images"][0]
        original_caption = entry["messages"][1]["content"] # assistant reply
        
        # Load and Resize Image for Report (Thumbnail)
        img = Image.open(src_image_path)
        img.thumbnail((300, 300)) # Resize for GitHub friendly view
        
        filename = f"{split_name}_{index}.jpg"
        dst_path = os.path.join(images_dir, filename)
        img.save(dst_path, quality=85)
        
        # Full path for model inference (use Original for better quality, or this one?)
        # Let's use the ORIGINAL path for Inference, but use this Thumbnail for Report.
        # Actually, for consistency, let's use the original path for inference.
        
        return {
            "id": f"{split_name}_{index}",
            "split": split_name,
            "image_path_original": os.path.abspath(src_image_path),
            "image_path_thumb": os.path.abspath(dst_path),
            "ground_truth": original_caption
        }

    print("Processing Train Samples (Gen 1-2)...")
    for i, entry in enumerate(selected_train):
        eval_dataset.append(process_sample(entry, "train", i))
        
    print("Processing Valid Samples (Gen 3+)...")
    for i, entry in enumerate(selected_valid):
        eval_dataset.append(process_sample(entry, "valid", i))
        
    # Save Metadata
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(eval_dataset, f, indent=2)
        
    print(f"Saved {len(eval_dataset)} samples to {json_path}")
    print(f"Thumbnails saved to {images_dir}")

if __name__ == "__main__":
    setup_eval_v2()
