
import json
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

def debug_template():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    print(f"Loading processor from {model_path}...")
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    # 1. Current String Format
    messages_str = [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "It is a pokemon."}
    ]
    
    print("\n--- Test 1: String Content ---")
    out1 = apply_chat_template(
        processor=processor,
        config=model.config,
        prompt=messages_str,
        return_messages=False 
    )
    print(out1)
    
    # Check Token Count of the PAD
    # We can't see IDs here, but we see the string. 
    # "<|image_pad|>" appears once.
    
    # 2. List of Dicts Format (Qwen2VL Standard)
    messages_dict = [
        {"role": "user", "content": [
            {"type": "image", "image": "file:///tmp/dummy.jpg"},
            {"type": "text", "text": "What is this?"}
        ]},
        {"role": "assistant", "content": "It is a pokemon."}
    ]
    
    print("\n--- Test 2: List[Dict] Content ---")
    try:
        out2 = apply_chat_template(
            processor=processor,
            config=model.config,
            prompt=messages_dict,
            return_messages=False
        )
        print(out2)
    except Exception as e:
        print(f"Error in Test 2: {e}")

if __name__ == "__main__":
    debug_template()
