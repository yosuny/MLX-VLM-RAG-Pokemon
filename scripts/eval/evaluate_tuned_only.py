import json
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

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


def run_tuned_only():
    dataset_path = "data/eval_v2/dataset.json"
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    adapter_path = "adapters"
    
    # Load existing dataset with Vanilla/RAG results
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # NEUTRAL PROMPT (same as used in Phase 1)
    neutral_prompt = "Describe this character. What is it called?"
    
    print("="*80)
    print("TUNED MODEL EVALUATION (LoRA Adapter)")
    print("="*80)
    print(f"\nUsing neutral prompt: '{neutral_prompt}'")
    
    # Load Model with LoRA Adapter
    print("\nLoading Model with LoRA Adapter...")
    try:
        model_tuned, processor_tuned = load(
            model_path, 
            adapter_path=adapter_path,
            processor_config={"trust_remote_code": True}
        )
        
        if hasattr(processor_tuned, "image_processor"):
            print("Applying Robust Processor Wrapper...")
            processor_tuned.image_processor = RobustImageProcessorWrapper(processor_tuned.image_processor)
        
        print("Adapter loaded successfully!\n")
        adapter_loaded = True
        
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Falling back to base model...\n")
        model_tuned, processor_tuned = load(model_path, processor_config={"trust_remote_code": True})
        if hasattr(processor_tuned, "image_processor"):
            processor_tuned.image_processor = RobustImageProcessorWrapper(processor_tuned.image_processor)
        adapter_loaded = False
    
    # Run Tuned Inference
    for i, item in enumerate(dataset):
        print(f"Tuned Inference {i+1}/{len(dataset)}...")
        image_path = item['image_path_original']
        
        formatted_prompt = apply_chat_template(
            processor_tuned,
            config=model_tuned.config,
            prompt=neutral_prompt,
            num_images=1
        )
        
        try:
            tuned_out = generate(
                model_tuned, processor_tuned,
                prompt=formatted_prompt,
                image=image_path,
                max_tokens=100,
                temperature=0.1,
                verbose=False
            )
        except Exception as e:
            print(f"Error generating for {item['id']}: {e}")
            tuned_out = f"Error: {str(e)}"

        item["tuned_result"] = tuned_out
    
    # Save updated dataset
    print("\nSaving updated dataset...")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Generate Report
    report_path = "EVALUATION_REPORT_v2.md"
    print(f"Generating report: {report_path}")
    
    with open(report_path, "w") as f:
        f.write("# Pokemon VLM Evaluation Report (v2)\n\n")
        f.write(f"**Evaluation Prompt**: `{neutral_prompt}`\n")
        f.write(f"**LoRA Adapter Status**: {'✅ Loaded' if adapter_loaded else '❌ Failed (using base model)'}\n\n")
        f.write("| Image | Type | Vanilla (Base) | RAG (Context) | Tuned (Gen 1-2) |\n")
        f.write("| :---: | :---: | --- | --- | --- |\n")
        
        for item in dataset:
            thumb_path = os.path.relpath(item['image_path_thumb'], start=os.getcwd())
            
            vanilla_text = item.get('vanilla_result', 'N/A').strip().replace("\n", " ")
            rag_text = item.get('rag_result', 'N/A').strip().replace("\n", " ")
            tuned_text = item.get('tuned_result', 'N/A').strip().replace("\n", " ")
            
            row = f"| ![]({thumb_path})<br><sub>{item['id']}</sub> | **{item['split'].upper()}** | {vanilla_text} | {rag_text} | **{tuned_text}** |\n"
            f.write(row)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"Report saved to: {report_path}")
    print(f"Dataset updated: {dataset_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_tuned_only()
