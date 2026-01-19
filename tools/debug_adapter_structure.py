import mlx_vlm
from mlx_vlm import load
import mlx.nn as nn

def print_structure():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    adapter_path = "adapters"
    
    print(f"Loading {model_path} with {adapter_path}...")
    model, _ = load(model_path, adapter_path=adapter_path, processor_config={"trust_remote_code": True})
    
    print("\n--- Inspecting First Layer ---")
    # Traverse to find a linear layer
    # Usually language_model.model.layers[0].self_attn.q_proj
    
    # Check top level
    print("Top level keys:", model.__dict__.keys())
    
    found = False
    for name, module in model.named_modules():
        if "q_proj" in name:
            print(f"\nFound Layer: {name}")
            print(f"Type: {type(module)}")
            print(f"Attributes: {dir(module)}")
            
            if hasattr(module, "lora_a"):
                print("✅ Found lora_a!")
            else:
                 print("❌ No lora_a found on this module.")
                 
            # Check if it has 'weight'
            if hasattr(module, "weight"):
                print(f"Has weight: {module.weight.shape}, type: {type(module.weight)}")
            
            found = True
            break
            
    if not found:
        print("Could not find any q_proj layer.")

if __name__ == "__main__":
    print_structure()
