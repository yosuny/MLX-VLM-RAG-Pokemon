import mlx.core as mx
from mlx_vlm import load
from transformers import AutoProcessor
import numpy as np

def manual_test_generic():
    model_path = "models/fused_qwen2_vl_4bit"
    image_path = "data/pokemon/images/pokemon_000.jpg" # Bulbasaur
    
    # Generic prompt without "Pokemon" hint
    prompt = "What is this? Answer in English and Korean."
    
    print(f"Loading model from {model_path}...")
    model, _ = load(model_path, processor_config={"trust_remote_code": True})
    
    print("Loading processor with use_fast=False...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    print(f"Testing Prompt: '{prompt}'")
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    from PIL import Image
    image = Image.open(image_path)
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="np"
    )
    
    # Convert to MLX arrays
    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    image_grid_thw = mx.array(inputs["image_grid_thw"])
    
    # Stateless generation loop
    curr_tokens = input_ids
    
    print("Generating...")
    for i in range(30):
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
        
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(axis=-1)
        next_token_id = next_token.item()
        
        curr_tokens = mx.concatenate([curr_tokens, next_token[None, :]], axis=1)

        if next_token_id == processor.tokenizer.eos_token_id:
            print("\nEOS reached.")
            break
            
        print(processor.tokenizer.decode([next_token_id]), end="", flush=True)

    print("\n\nDone.")

if __name__ == "__main__":
    manual_test_generic()
