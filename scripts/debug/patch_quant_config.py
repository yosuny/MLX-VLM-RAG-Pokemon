import json

path = "models/fused_qwen2_vl_4bit_quantized/config.json"

with open(path, 'r') as f:
    config = json.load(f)

config["quantization"] = {
    "group_size": 64,
    "bits": 4
}

with open(path, 'w') as f:
    json.dump(config, f, indent=4)

print("Config updated.")
