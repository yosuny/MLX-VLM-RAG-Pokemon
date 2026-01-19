import argparse
import logging
import os
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm import load

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fuse_vlm")

def dequantize_and_fuse(child, name="unknown"):
    """
    Process a layer:
    1. If QuantizedLinear -> Dequantize to Float16.
    2. If LoRA -> Fuse update.
    3. Return new nn.Linear (or None if no change needed).
    """
    
    # 1. Identify Base Weights and Dequantize if needed
    w_base = None
    bias = None
    is_quantized = False
    
    # Target could be the child itself, or wrapped in original_layer/linear
    target = child
    if hasattr(child, "original_layer"):
        target = child.original_layer
    elif hasattr(child, "linear"):
        target = child.linear

    # Check for QuantizedLinear attributes
    if hasattr(target, "weight") and hasattr(target, "scales") and hasattr(target, "biases"):
        try:
            w_base = mx.dequantize(
                target.weight, 
                target.scales, 
                target.biases, 
                target.group_size, 
                target.bits
            )
            is_quantized = True
        except Exception as e:
            logger.error(f"[{name}] Dequantization failed: {e}")
            return None
    elif hasattr(target, "weight"):
        # Not quantized, just use weight
        w_base = target.weight

    if hasattr(target, "bias"):
        bias = target.bias

    # 2. Check for LoRA Attributes
    lora_a = None
    lora_b = None
    scale = 1.0

    if hasattr(child, "lora_a") and hasattr(child, "lora_b"):
        lora_a = child.lora_a
        lora_b = child.lora_b
        if hasattr(child, "scale"): scale = child.scale
    elif hasattr(child, "A") and hasattr(child, "B"):
        lora_a = child.A
        lora_b = child.B
        if hasattr(child, "scale"): scale = child.scale
        elif hasattr(child, "scaling"): scale = child.scaling

    is_lora = (lora_a is not None and lora_b is not None)

    # Decisions
    if not is_quantized and not is_lora:
        # Nothing to do (already dense, no adapter)
        # UNLESS we want to ensure everything is Float16? 
        # But if it's already dense, we leave it.
        return None

    if w_base is None:
        if is_lora:
            logger.warning(f"[{name}] LoRA adapter found but no base weights? Skipping.")
        return None

    # 3. Calculate Update (if LoRA)
    update = None
    if is_lora:
        # Try B @ A 
        try:
            cand = (lora_b @ lora_a) * scale
            if cand.shape == w_base.shape:
                update = cand
        except: pass
        
        # Try A @ B with Transpose check
        if update is None:
            try:
                cand = (lora_a @ lora_b) * scale
                if cand.shape == w_base.shape:
                    update = cand
                elif cand.T.shape == w_base.shape:
                    update = cand.T
            except: pass
        
        if update is None:
             logger.error(f"[{name}] Shape mismatch for LoRA fusion. Base: {w_base.shape}")
             # If fusion fails, we could potentially still return dequantized base? 
             # But likely better to fail or skip.
             # We will proceed with just w_base if update fails? No, that loses adapter info.
             return None

    # 4. Create New Weights
    w_new = w_base.astype(mx.float16)
    if update is not None:
        w_new = w_new + update.astype(mx.float16)

    # 5. Create new nn.Linear
    out_features, in_features = w_new.shape
    
    if bias is not None:
        bias = bias.astype(mx.float16)

    new_linear = nn.Linear(in_features, out_features, bias=(bias is not None))
    new_linear.weight = w_new
    if bias is not None:
        new_linear.bias = bias
        
    action = "Fused & Dequantized" if (is_quantized and is_lora) else ("Fused" if is_lora else "Dequantized")
    print(f"{action}: {name}", flush=True)
    
    return new_linear


def fuse_model(model):
    logger.info("Processing model for Dequantization and Fusion...")
    
    def _process_module(module, path="model"):
        # 1. Handle List
        if isinstance(module, list):
            for i, child in enumerate(module):
                child_path = f"{path}.{i}"
                new_child = dequantize_and_fuse(child, name=child_path)
                if new_child is not None:
                    module[i] = new_child
                else:
                    _process_module(child, path=child_path)
            return

        # 2. Handle Dict
        if isinstance(module, dict):
            for name, child in module.items():
                child_path = f"{path}.{name}"
                new_child = dequantize_and_fuse(child, name=child_path)
                if new_child is not None:
                    module[name] = new_child
                else:
                    _process_module(child, path=child_path)
            return

        # 3. Explicit Attributes for deep recursion
        explicit_attrs = ["language_model", "vision_tower", "model", "layers", "self_attn", "mlp", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "merger", "blocks", "patch_embed", "attn", "fc1", "fc2", "proj", "qkv"] 
        # Added vision tower specific attributes (blocks, patch_embed, etc assuming structure)
        
        for attr in explicit_attrs:
            if hasattr(module, attr):
                child = getattr(module, attr)
                child_path = f"{path}.{attr}"
                new_child = dequantize_and_fuse(child, name=child_path)
                if new_child is not None:
                    setattr(module, attr, new_child)
                else:
                    _process_module(child, path=child_path)

        # 4. Generic Children
        if hasattr(module, "children"):
            children = module.children()
            for name, child in children.items():
                child_path = f"{path}.{name}"
                if name in explicit_attrs: continue 

                new_child = dequantize_and_fuse(child, name=child_path)
                if new_child is not None:
                    setattr(module, name, new_child)
                else:
                    _process_module(child, path=child_path)

    _process_module(model)
    return model

def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into Qwen2-VL")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2-VL-7B-Instruct-4bit", help="Base model path")
    parser.add_argument("--adapter-path", type=str, default="adapters", help="Adapter path")
    parser.add_argument("--save-path", type=str, default="models/fused_qwen2_vl_4bit", help="Output path")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model} with adapter: {args.adapter_path}", flush=True)
    model, processor = load(args.model, adapter_path=args.adapter_path, processor_config={"trust_remote_code": True})
    print("Model loaded successfully.", flush=True)
    
    model = fuse_model(model)
    
    logger.info(f"Saving fused model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    model.save_weights(os.path.join(args.save_path, "model.safetensors"))
    
    # Save config
    def make_serializable(obj):
        if hasattr(obj, "to_dict"): return obj.to_dict()
        if hasattr(obj, "__dict__"): return obj.__dict__
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        if isinstance(obj, list): return [make_serializable(x) for x in obj]
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        return str(obj)

    config_dict = model.config.as_dict() if hasattr(model.config, "as_dict") else model.config.__dict__
    if "quantization" in config_dict: del config_dict["quantization"]
    
    if 'text_config' in config_dict:
        text_config = config_dict['text_config']
        if hasattr(text_config, "to_dict"): text_config = text_config.to_dict()
        elif hasattr(text_config, "__dict__"): text_config = text_config.__dict__
        
        if isinstance(text_config, dict):
            for k, v in text_config.items():
                if k not in config_dict: config_dict[k] = v
            
    config_dict = make_serializable(config_dict)
    
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.save_path)
    
    logger.info("Done! Fusion Complete.")

if __name__ == "__main__":
    main()
