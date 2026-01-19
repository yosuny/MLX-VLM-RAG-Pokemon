
import argparse
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoImageProcessor
from PIL import Image
import glob
import os
import mlx_vlm.models.qwen2_vl.qwen2_vl as qwen2_vl_model

# --- KEY PATCH: SMART PADDING for Merge ---
def _merge_input_ids_with_image_features_smart(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    # 1. Find Offset (Start Index of Image Tokens)
    # We assume batch size 1 for simplicity in inference
    start_idx = mx.argmax(image_positions, axis=1).item()
    
    total_len = inputs_embeds.shape[1]
    feat_len = image_features.shape[1]
    
    # 2. Pad Image Features to align with inputs_embeds
    # Left Pad = start_idx (Skipping text tokens)
    # Right Pad = Rest
    pad_left = start_idx
    pad_right = total_len - (start_idx + feat_len)
    
    if pad_right < 0:
         pad_right = 0
         
    image_features = mx.pad(image_features, ((0, 0), (pad_left, pad_right), (0, 0)))
    
    # 3. Apply Where
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

# Apply Patch
qwen2_vl_model.Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_smart


def sample(logits, temperature=0.7):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))

def load_image_processor_patched(model_path):
    try:
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return processor
    except Exception as e:
        print(f"Failed to load image processor: {e}")
        return None

def run_inference():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    # adapter_path = "adapters_v3_final" 
    
    print(f"Loading Model: {model_path}")
    # print(f"Loading Adapter: {adapter_path}")
    
    # Load Base Model Only
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    # Resize Patch
    image_processor = load_image_processor_patched(model_path)
    if image_processor:
        processor.image_processor = image_processor
        processor.image_processor.max_pixels = 512 * 512
    
    # Test Image
    img_files = glob.glob("data_pilot/images/*.jpg")
    img_path = img_files[0]
    print(f"Testing on: {os.path.basename(img_path)}")
    
    # 1. Prepare Inputs
    messages = [{"role": "user", "content": "Describe this character. What is it called in English and Korean?"}]
    text = apply_chat_template(
        processor, model.config, messages, return_messages=False, add_generation_prompt=True
    )
    
    # 2. Process Image & Expand Tokens
    image = Image.open(img_path)
    image_processed = processor.image_processor.preprocess(image, return_tensors='np')
    pixel_values = mx.array(image_processed['pixel_values'])
    grid = image_processed['image_grid_thw'][0]
    num_tokens = int(grid[1] * grid[2])
    
    final_prompt = text.replace("<|image_pad|>", "<|image_pad|>" * num_tokens)
    print(f"Tokens Injected: {num_tokens}")
    
    # 3. Tokenize
    input_ids = processor.tokenizer(final_prompt, return_tensors='np')['input_ids']
    input_ids = mx.array(input_ids)
    
    print("Starting Generation...")
    
    # 4. Custom Generation Loop
    tokens = []
    max_new_tokens = 200
    
    # Cache for autoregressive generation
    cache = None
    
    curr_input_ids = input_ids
    
    # Pass grid to model
    grid_mx = mx.array(image_processed['image_grid_thw'])

    for _ in range(max_new_tokens):
        # Forward Pass
        # Note: model() takes input_ids, pixel_values (optional), cache
        
        # First step: pass pixel_values AND grid
        if len(tokens) == 0:
            outputs = model(curr_input_ids, pixel_values=pixel_values, image_grid_thw=grid_mx, mask=None, cache=cache)
        else:
            # Subsequent steps: No pixel_values needed
            outputs = model(curr_input_ids, pixel_values=None, mask=None, cache=cache)
            
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            cache = getattr(outputs, 'cache', None) or getattr(outputs, 'past_key_values', None) or cache
        else:
            logits, cache = outputs
            
        # Get last token logits
        next_token_logits = logits[:, -1, :]
        
        # Sample
        next_token = sample(next_token_logits, temperature=0.7)
        token_item = next_token.item()
        
        if token_item == processor.tokenizer.eos_token_id:
            print("[EOS]")
            break
            
        tokens.append(token_item)
        curr_input_ids = next_token[None, :] # [1, 1]
        
        # Decode live
        print(processor.tokenizer.decode([token_item]), end="", flush=True)
        
    print("\nGeneration Complete.")

if __name__ == "__main__":
    run_inference()
