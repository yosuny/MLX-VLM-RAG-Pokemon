#!/usr/bin/env python3
"""
Diagnostic test to identify the root cause of evaluation inference failure.
Tests both prompt formats (MLX vs HF) and detokenizer implementations.
"""
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
from transformers import AutoProcessor
from PIL import Image

# Test image
TEST_IMAGE = "data_pokemon/images/001_Bulbasaur.png"
MODEL_PATH = "mlx-community/Qwen2-VL-7B-Instruct-4bit"

print("="*80)
print("DIAGNOSTIC TEST: Comparing MLX vs HF prompt formatting")
print("="*80)

# Load model
print("\n[1/4] Loading model...")
model, mlx_processor = load(MODEL_PATH, processor_config={"trust_remote_code": True})

# Load HF processor separately
print("[2/4] Loading HF processor...")
hf_processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Test 1: MLX apply_chat_template (Working in server.py)
print("\n[3/4] TEST 1: MLX prompt format")
prompt = "What is the name of this Pokemon?"
mlx_formatted = mlx_apply_chat_template(
    mlx_processor,
    config=model.config,
    prompt=prompt,
    num_images=1
)
print(f"MLX Formatted Prompt:\n{mlx_formatted}\n")
print(f"Length: {len(mlx_formatted)} chars")

# Test 2: HF apply_chat_template (Used in evaluate_models_v2.py)
print("\n[4/4] TEST 2: HF prompt format")
messages = [{"role": "user", "content": prompt}]
hf_formatted = hf_processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"HF Formatted Prompt:\n{hf_formatted}\n")
print(f"Length: {len(hf_formatted)} chars")

# Comparison
print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"Formats are {'IDENTICAL' if mlx_formatted == hf_formatted else 'DIFFERENT'}")

if mlx_formatted != hf_formatted:
    print("\n⚠️  FOUND ISSUE: Prompt formats differ!")
    print("\nDifference Analysis:")
    print(f"  MLX includes: {set(mlx_formatted) - set(hf_formatted)}")
    print(f"  HF includes: {set(hf_formatted) - set(mlx_formatted)}")
    
    # Show side by side
    print("\n--- MLX Format (first 200 chars) ---")
    print(repr(mlx_formatted[:200]))
    print("\n--- HF Format (first 200 chars) ---")
    print(repr(hf_formatted[:200]))

# Test actual inference with both formats
print("\n" + "="*80)
print("RUNNING INFERENCE TEST (max_tokens=50 for speed)")
print("="*80)

print("\n[TEST 1] Using MLX format...")
try:
    output_mlx = generate(
        model, mlx_processor, 
        prompt=mlx_formatted, 
        image=TEST_IMAGE, 
        max_tokens=50,
        verbose=False
    )
    print(f"✅ SUCCESS")
    print(f"Output: {output_mlx[:100]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n[TEST 2] Using HF format...")
try:
    output_hf = generate(
        model, mlx_processor, 
        prompt=hf_formatted, 
        image=TEST_IMAGE, 
        max_tokens=50,
        verbose=False
    )
    print(f"✅ SUCCESS")
    print(f"Output: {output_hf[:100]}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
