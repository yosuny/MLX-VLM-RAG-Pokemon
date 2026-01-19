import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import numpy as np

# Monkey Patch for Qwen2-VL padding bug
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_input_ids_with_image_features_patched(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    
    if pad_size < 0:
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
        
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched

class RobustImageProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor
        if hasattr(processor, "image_processor"):
            self.processor = processor.image_processor
        for attr in dir(self.processor):
            if not attr.startswith("__"):
                try:
                    setattr(self, attr, getattr(self.processor, attr))
                except:
                    pass

    def __call__(self, images=None, text=None, **kwargs):
        if "return_tensors" in kwargs:
            kwargs["return_tensors"] = "pt"
        out = self.processor(images, text, **kwargs)
        for k, v in out.items():
            if hasattr(v, "numpy"):
                out[k] = v.numpy()
            elif isinstance(v, list) and hasattr(v[0], "numpy"):
                out[k] = [x.numpy() for x in v]
        return out
        
    def preprocess(self, images, **kwargs):
        if "return_tensors" in kwargs:
            kwargs["return_tensors"] = "pt"
        out = self.processor.preprocess(images, **kwargs)
        if hasattr(out, "pixel_values"):
           if hasattr(out["pixel_values"], "numpy"):
               out["pixel_values"] = out["pixel_values"].numpy()
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(v, "numpy"):
                    out[k] = v.numpy()
        return out
        
    def __getattr__(self, name):
         return getattr(self.processor, name)

def debug_tuned():
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    adapter_path = "adapters"
    image_path = "data_pokemon/images/pokemon_001.jpg" # Bulbasaur
    
    print("="*80)
    print("DEBUGGING TUNED MODEL GENERATION")
    print("="*80)

    # ... (Weights check code remains)

    # 2. Load Model
    print("\n[2] Loading Model with Adapters...")
    model, processor = load(model_path, adapter_path=adapter_path, processor_config={"trust_remote_code": True})
    
    if hasattr(processor, "image_processor"):
        print("Applying Robust Processor Wrapper...")
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)

    
    # 3. Test Prompts
    prompts = [
        "Describe this character. What is it called?", # Neutral
        "What is the name of this Pokemon?", # Original (Overfit check)
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n[3-{i+1}] Testing Prompt: '{prompt}'")
        
        formatted_prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=prompt,
            num_images=1
        )
        print(f"Formatted Prompt Length: {len(formatted_prompt)}")
        # print(f"Formatted Prompt:\n{formatted_prompt}") 
        
        print("Generating (verbose=True)...")
        output = generate(
            model, processor,
            prompt=formatted_prompt,
            image=image_path,
            max_tokens=100,
            temperature=0.7, # Higher temp to encourage diversity
            verbose=True
        )
        print(f"\nResult: '{output}'")
        print(f"Length: {len(output)}")

if __name__ == "__main__":
    debug_tuned()
