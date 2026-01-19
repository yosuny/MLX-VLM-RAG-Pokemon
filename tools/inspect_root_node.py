from mlx_vlm import load

model, _ = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")
print(f"Model Type: {type(model)}")
print(f"Dir: {dir(model)}")
print(f"Vars Keys: {vars(model).keys()}")

if hasattr(model, "language_model"):
    print("Found 'language_model'")
    lm = model.language_model
    print(f"LM keys: {vars(lm).keys()}")
    if hasattr(lm, "model"):
         print(f"LM.model keys: {vars(lm.model).keys()}")
