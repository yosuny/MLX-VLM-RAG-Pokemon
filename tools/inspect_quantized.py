from mlx_vlm import load
import mlx.nn as nn

print("Loading model...")
model, _ = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")

# Find a quantized layer
for name, mod in model.named_modules():
    if hasattr(mod, "original_layer"): # wrapped
        mod = mod.original_layer
        
    if "QuantizedLinear" in str(type(mod)):
        print(f"Found QuantizedLinear: {name}")
        print(f"Type: {type(mod)}")
        print(f"Attributes: {vars(mod).keys()}")
        if hasattr(mod, "weight"):
            print(f"Weight shape: {mod.weight.shape}")
        if hasattr(mod, "scales"):
            print(f"Scales shape: {mod.scales.shape}")
        if hasattr(mod, "biases"):
            print(f"Biases shape: {mod.biases.shape}")
        if hasattr(mod, "group_size"):
             print(f"Group size: {mod.group_size}")
        if hasattr(mod, "bits"):
             print(f"Bits: {mod.bits}")
        break
