import mlx.core as mx
from mlx_vlm import load
from transformers import AutoProcessor
import numpy as np

def manual_test():
    model_path = "models/fused_qwen2_vl_4bit_quantized"
    image_path = "data/pokemon/images/pokemon_000.jpg"
    
    print(f"Loading model from {model_path}...")
    model, _ = load(model_path, processor_config={"trust_remote_code": True})
    
    print("Loading processor with use_fast=False...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    prompt = "What is this pokemon's name? Answer in English and Korean."
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    print("Processing inputs...")
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = None # Load image manually if needed, but processor expects path or PIL
    
    # Use processor to get inputs
    # Note: MLX VLM expects specific input keys. 
    # Qwen2-VL: input_ids, pixel_values, image_grid_thw
    from PIL import Image
    image = Image.open(image_path)
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="np" # Request Numpy!
    )
    
    # Convert to MLX arrays
    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    image_grid_thw = mx.array(inputs["image_grid_thw"])
    
    print("Inputs prepared.")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Pixel Values shape: {pixel_values.shape}")
    
    # Simple manual generation loop (Greedy)
    print("Generating...")
    
    # Pre-compute cache
    cache = None
    # Stateless generation loop (Pass full history each time)
    curr_tokens = input_ids # (1, N)
    
    for i in range(20):
        # Create mask for current length
        B, L = curr_tokens.shape
        mask = mx.ones((B, L))
        
        outputs = model(
            input_ids=curr_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mask=mask
        )
        logits = outputs.logits
        
        # Greedy decode last token
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(axis=-1)
        next_token_id = next_token.item()
        
        # Append
        curr_tokens = mx.concatenate([curr_tokens, next_token[None, :]], axis=1)

        # Stop on EOS
        if next_token_id == processor.tokenizer.eos_token_id:
            print("\nEOS reached.")
            break
            
        # Print
        print(processor.tokenizer.decode([next_token_id]), end="", flush=True)

    print("\n\nDone.")

if __name__ == "__main__":
    import traceback
    try:
        manual_test()
    except Exception:
        traceback.print_exc()
