import json
import os
import sys
import gc

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from src.rag_engine import RAGEngine

# Monkey Patch for Qwen2-VL padding bug
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_input_ids_with_image_features_patched(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    # FIX: Calculate pad size. If negative, truncate image features.
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    
    if pad_size < 0:
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
        
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

# Apply patch
Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched


class RobustImageProcessorWrapper:
    """
    Wraps a generic ImageProcessor to force return_tensors='pt' and convert to numpy.
    This bypasses the 'Only returning PyTorch tensors is currently supported' error.
    """
    def __init__(self, processor):
        self.processor = processor
        # Copy attributes
        if hasattr(processor, "image_processor"):
            self.processor = processor.image_processor
        for attr in dir(self.processor):
            if not attr.startswith("__"):
                try:
                    setattr(self, attr, getattr(self.processor, attr))
                except:
                    pass

    def __call__(self, images=None, text=None, **kwargs):
        # Force PT, then convert
        if "return_tensors" in kwargs:
            kwargs["return_tensors"] = "pt"
        
        out = self.processor(images, text, **kwargs)
        
        # Convert all tensors to numpy
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
        
        # Convert
        if hasattr(out, "pixel_values"):
           if hasattr(out["pixel_values"], "numpy"):
               out["pixel_values"] = out["pixel_values"].numpy()
        
        # Generic dict conversion
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(v, "numpy"):
                    out[k] = v.numpy()
                    
        return out
        
    def __getattr__(self, name):
         return getattr(self.processor, name)


def cleanup_memory():
    """Force garbage collection and clear MLX cache."""
    gc.collect()
    try:
        mx.metal.clear_cache()
    except:
        pass


def evaluate_v2():
    dataset_path = "data/eval_v2/dataset.json"
    model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    adapter_path = "adapters"
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f) # Processing all 10 samples
        
    def resize_for_speed(image_path, max_side=512):
        from PIL import Image
        try:
            img = Image.open(image_path)
            ratio = max_side / max(img.size)
            if ratio < 1:
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                # Save to temp
                temp_path = image_path.replace(".jpg", "_sized.jpg")
                img.save(temp_path)
                return temp_path
            return image_path
        except Exception as e:
            print(f"Resize failed: {e}")
            return image_path
        
    print("="*80)
    print("PHASE 1: VANILLA + RAG EVALUATION (Base Model)")
    print("="*80)
    
    print("\nLoading Base Model...")
    model, processor = load(model_path, processor_config={"trust_remote_code": True})
    
    # Apply Robust Processor Wrapper
    if hasattr(processor, "image_processor"):
        print("Applying Robust Processor Wrapper...")
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
    
    print("Loading RAG Engine...")
    rag = RAGEngine()
    
    # NEUTRAL PROMPT (modified for Korean)
    neutral_prompt = "Describe this character. What is it called in English and Korean?"
    
    print(f"\nStarting Evaluation on {len(dataset)} samples...")
    print(f"Using neutral prompt: '{neutral_prompt}'\n")
    
    # Phase 1: Vanilla + RAG Inference
    for i, item in enumerate(dataset):
        print(f"Processing {i+1}/{len(dataset)}: {item['id']}")
        
        # Resize for speed
        image_path = resize_for_speed(item['image_path_original'])
        
        # 1. Vanilla Inference
        print("  - Running Vanilla...")
        formatted_prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=neutral_prompt,
            num_images=1
        )
        
        vanilla_out = generate(
            model, processor, 
            prompt=formatted_prompt, 
            image=image_path, 
            max_tokens=100,
            temperature=0.1,
            verbose=False
        )
        
        # 2. RAG Inference
        print("  - Running RAG...")
        retrieved = rag.search(query=image_path, top_k=1)
        rag_context = ""
        rag_image_path = None
        
        if retrieved and retrieved.get('documents') and len(retrieved['documents']) > 0:
            docs = retrieved['documents'][0]
            metas = retrieved['metadatas'][0]
            
            if len(docs) > 0:
                top_doc = docs[0]
                top_meta = metas[0]
                
                rag_context = f"\n\nHint: Similar character info - {top_doc}"
                rag_image_path = top_meta.get('path', 'Unknown')
                
                # RAG Prompt (with context)
                rag_prompt_text = f"{neutral_prompt}{rag_context}"
                rag_formatted = apply_chat_template(
                    processor,
                    config=model.config,
                    prompt=rag_prompt_text,
                    num_images=1
                )
                
                rag_out = generate(
                    model, processor,
                    prompt=rag_formatted,
                    image=image_path,
                    max_tokens=100,
                    temperature=0.1,
                    verbose=False
                )
        else:
            rag_out = "RAG Failed"
            
        # Cleanup temp resized image if created
        if "_sized.jpg" in image_path:
            try:
                os.remove(image_path)
            except:
                pass
            
        item["vanilla_result"] = vanilla_out
        item["rag_result"] = rag_out
        item["rag_retrieved_image"] = rag_image_path
    
    # Save Phase 1 results
    print("\nSaving Phase 1 results to dataset...")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Clean up base model
    print("\n" + "="*80)
    print("PHASE 2: TUNED EVALUATION (LoRA Adapter)")
    # Phase 2: Tuned Inference (SKIPPED per user request)
    # print("\n" + "="*80)
    # print("PHASE 2: TUNED EVALUATION (LoRA Adapter)")
    # ... (Skipped)
        
    # Generate Report (Vanilla + RAG Only)
    report_path = "EVALUATION_REPORT_v2.md"
    with open(report_path, "w") as f:
        f.write("# Pokemon VLM Evaluation Report (v2)\n\n")
        f.write(f"**Evaluation Prompt**: `{neutral_prompt}`\n")
        f.write(f"**Comparison**: Vanilla (Base) vs RAG (Context)\n\n")
        f.write("| Image | Type | Vanilla (Base) | RAG (Context) |\n")
        f.write("| :---: | :---: | --- | --- |\n")
        
        for item in dataset:
            thumb_path = os.path.relpath(item['image_path_thumb'], start=os.getcwd())
            
            vanilla_text = item.get('vanilla_result', 'N/A').strip().replace("\n", " ")
            rag_text = item.get('rag_result', 'N/A').strip().replace("\n", " ")
            
            row = f"| ![]({thumb_path})<br><sub>{item['id']}</sub> | **{item['split'].upper()}** | {vanilla_text} | **{rag_text}** |\n"
            f.write(row)
            
    print(f"\n{'='*80}")
    print(f"Evaluation Complete. Report saved to {report_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    evaluate_v2()
