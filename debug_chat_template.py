
import json
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

def debug_template():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    print(f"Loading processor from {model_path}...")
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    # Load 1 sample
    with open("data_pokemon/train.jsonl", "r") as f:
        line = f.readline()
        entry = json.loads(line)
        
    print("\n--- Original Entry ---")
    print(entry["messages"])
    
    # Apply Template
    print("\n--- Applied Template ---")
    formatted = apply_chat_template(
        processor=processor,
        config=model.config,
        prompt=entry["messages"],
        return_messages=False # Get string to see tokens
    )
    
    print(formatted)
    
    # Check for keys
    if "<|image_pad|>" in formatted or "<|vision_start|>" in formatted:
        print("\n✅ Success: Image tokens detected.")
    else:
        print("\n❌ Failure: NO image tokens found! The model is blind.")

if __name__ == "__main__":
    debug_template()
