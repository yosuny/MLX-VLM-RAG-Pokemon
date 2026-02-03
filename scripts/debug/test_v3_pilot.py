
import argparse
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoImageProcessor
from PIL import Image
import glob
import os
import random

# --- PATCHES for Qwen2-VL (Same as Training) ---
def load_image_processor_patched(model_path):
    try:
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return processor
    except Exception as e:
        print(f"Failed to load image processor: {e}")
        return None

# Verify V3 Logic
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_input_ids_with_image_features_patched_v3(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    # SMART PADDING: Align features to start of image tokens
    # Assume Batch Size = 1 for this fix (safety)
    
    # Find start index
    # input_ids: [1, L]
    # image_positions: [1, L] (Boolean)
    
    # Argmax returns index of first True
    start_idx = mx.argmax(image_positions, axis=1).item()
    
    total_len = inputs_embeds.shape[1]
    feat_len = image_features.shape[1]
    
    pad_left = start_idx
    pad_right = total_len - (start_idx + feat_len)
    
    if pad_right < 0:
         # Features > Slots available from start?
         # Truncate features
         image_features = image_features[:, :total_len - start_idx, :]
         pad_right = 0
         
    image_features = mx.pad(image_features, ((0, 0), (pad_left, pad_right), (0, 0)))
    
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched_v3


def run_comparison():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    adapter_path = "adapters_v3_final"
    
    img_files = glob.glob("data_pilot/images/*.jpg")
    test_img = img_files[0] if img_files else None
    if not test_img:
        print("No images found.")
        return

    print(f"--- TESTING BASE MODEL (No Adapter) with V3 Expansion ---")
    run_inference(model_path, None, test_img)
    
    print(f"\n--- TESTING LoRA MODEL ({adapter_path}) with V3 Expansion ---")
    run_inference(model_path, adapter_path, test_img)

def run_inference(model_path, adapter_path, img_path):
    print(f"Loading {model_path} (Adapter: {adapter_path})...")
    
    if adapter_path:
        model, processor = load(model_path, adapter_path=adapter_path, processor_config={"trust_remote_code": True})
    else:
        model, processor = load(model_path, processor_config={"trust_remote_code": True})

    # Patch processor
    image_processor = load_image_processor_patched(model_path)
    if image_processor:
        processor.image_processor = image_processor
        processor.image_processor.max_pixels = 512 * 512 

    # Prepare Prompt
    messages = [{"role": "user", "content": "Describe this character. What is it called in English and Korean?"}]
    
    # Force generation prompt for Qwen2-VL
    # We manually append if apply_chat_template doesn't? 
    # Usually apply_chat_template(..., add_generation_prompt=True) works.
    
    base_text = apply_chat_template(
        processor=processor,
        config=model.config,
        prompt=messages,
        return_messages=False,
        add_generation_prompt=True
    )
    
    # V3 Token Expansion
    image = Image.open(img_path)
    out = processor.image_processor.preprocess(image, return_tensors='np')
    grid = out['image_grid_thw'][0]
    num_tokens = int(grid[1] * grid[2])
    
    # DISABLE EXPANSION FOR DEBUGGING
    # final_prompt = base_text.replace("<|image_pad|>", "<|image_pad|>" * num_tokens)
    final_prompt = base_text
    
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Tokens Injected: 1 (DEBUG MODE)")
    print(f"Prompt Tail: {final_prompt[-50:]}") # Check the end

    output = generate(
        model, 
        processor, 
        prompt=final_prompt, 
        images=[img_path], 
        max_tokens=100,
        verbose=True
    )
    print(f"\nRESULT: {output}\n")

if __name__ == "__main__":
    run_comparison()

