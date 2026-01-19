import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import AutoProcessor
import os
import sys

# Ensure imports work if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fused():
    model_path = "models/fused_qwen2_vl_4bit"
    image_path = "data/pokemon/images/pokemon_000.jpg" # Bulbasaur/Ivysaur
    
    print(f"Loading FUSED model from {model_path}...")
    
    # Load model (Should be standard Qwen2-VL now, no adapter needed)
    try:
        # Explicitly load processor with use_fast=False to avoid PyTorch tensor error
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        # Shim for MLX compatibility (mlx_vlm expects detokenizer.reset())
        class DetokenizerShim:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
            def reset(self):
                pass
            def decode(self, tokens):
                return self.tokenizer.decode(tokens)
                
        processor.detokenizer = DetokenizerShim(processor.tokenizer) 
        
        model, _ = load(model_path, processor_config={"trust_remote_code": True})
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    prompt = "What is this pokemon's name? Answer in English and Korean."
    
    formatted_prompt = apply_chat_template(
        processor,
        config=model.config,
        prompt=prompt,
        num_images=1
    )
    
    print(f"Testing inference on {os.path.basename(image_path)}...")
    try:
        output = generate(
            model, 
            processor, 
            prompt=formatted_prompt,
            image=image_path,
            max_tokens=100, 
            temperature=0.1,
            verbose=False
        )
        print("\n--- Result ---")
        print(output)
        print("--------------\n")
        
        if "Bulbasaur" in output or "이상해씨" in output:
             print("✅ Success: Correct Pokemon identified!")
        else:
             print("⚠️ Warning: Model loaded but answer might be incorrect.")
             
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Inference FAILED: {e}")

if __name__ == "__main__":
    test_fused()
