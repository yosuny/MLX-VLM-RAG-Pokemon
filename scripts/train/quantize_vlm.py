import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load
import os
import json
import shutil
import glob

def quantize_model():
    parser = argparse.ArgumentParser(description="Quantize fused Float16 Qwen2-VL to 4-bit")
    parser.add_argument("--model", type=str, default="models/fused_qwen2_vl_4bit", help="Path to fused float16 model")
    parser.add_argument("--save-path", type=str, default="models/fused_qwen2_vl_4bit_quantized", help="Output path")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits")
    
    args = parser.parse_args()
    
    print(f"Loading Float16 model from {args.model}...")
    # Load model using mlx_vlm
    model, processor = load(args.model)
    
    print(f"Quantizing to {args.q_bits}-bit (Group Size: {args.q_group_size})...")
    # Apply quantization
    # We quantize Linear layers. Qwen2-VL has QuantizedLinear support in MLX.
    # mlx_vlm.utils.quantize_model is commonly used if available, or nn.quantize
    
    # Using nn.quantize to modify the model in-place (model, group_size, bits)
    nn.quantize(model, args.q_group_size, args.q_bits)
    
    print(f"Saving quantized model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    
    # Save weights
    model.save_weights(os.path.join(args.save_path, "model.safetensors"))
    
    # Save config and processor files
    print("Copying config files...")
    # Update config to reflect quantization
    config_dict = model.config.as_dict() if hasattr(model.config, "as_dict") else model.config.__dict__
    
    # Add quantization config manually
    config_dict["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits
    }
    
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.save_path)
    else:
        # Manual copy of unrelated files if save_pretrained misses them
        for filename in ["preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "*.jpg", "*.png"]:
             for file in glob.glob(os.path.join(args.model, filename)):
                 shutil.copy(file, args.save_path)

    print(f"✅ Quantization Complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    quantize_model()
