
import json
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoImageProcessor
from PIL import Image
import os
import glob

def debug_v3_logic():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    print(f"Loading processor from {model_path}...")
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    # Patched Processor Load
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        processor.image_processor = image_processor
        print("Loaded slow image processor.")
    except Exception as e:
        print(f"Error loading slow processor: {e}")

    # Find a sample image
    img_files = glob.glob("data/pokemon/images/*.jpg")
    if not img_files:
        print("No images found in data/pokemon/images/")
        return
    
    img_path = img_files[0]
    print(f"Testing on image: {img_path}")
    
    # Simulate V3 Logic
    messages = [{"role": "user", "content": "What is this?"}]
    
    # 1. Base Template (1 Token)
    base_text = apply_chat_template(
        processor=processor,
        config=model.config,
        prompt=messages,
        return_messages=False # Get string
    )
    print(f"Original Text Token Count ('<|image_pad|>'): {base_text.count('<|image_pad|>')}")
    
    # 2. V3 Expansion
    image = Image.open(img_path)
    # Preprocess to get Grid Size
    out = processor.image_processor.preprocess(image, return_tensors='np')
    
    if 'image_grid_thw' in out:
        grid = out['image_grid_thw'][0] # [t, h, w]
        num_tokens = int(grid[1] * grid[2])
        print(f"Grid Size: {grid} (T, H, W)")
        print(f"Calculated Tokens: {num_tokens}")
        
        # Expand
        expanded_text = base_text.replace(
            "<|image_pad|>", 
            "<|image_pad|>" * num_tokens
        )
        
        final_count = expanded_text.count('<|image_pad|>')
        print(f"Final Text Token Count: {final_count}")
        
        if final_count == num_tokens:
            print("✅ SUCCESSS: Token expansion works correctly.")
        else:
            print("❌ FAILURE: Token count mismatch.")
            
        # Check Length
        print(f"Original Length: {len(base_text)}")
        print(f"Final Length: {len(expanded_text)}")
        
    else:
        print("❌ FAILURE: No image_grid_thw returned.")

if __name__ == "__main__":
    debug_v3_logic()
