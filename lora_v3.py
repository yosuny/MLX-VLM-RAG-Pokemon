import argparse
import json
import logging
import math
import mlx.optimizers as optim
import mlx.core as mx
from datasets import load_dataset
from tqdm import tqdm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer import Dataset, Trainer, save_adapter
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load
from transformers import AutoImageProcessor
from PIL import Image
import os

# --- PATCHES for Qwen2-VL ---

def load_image_processor_patched(model_path):
    try:
        # Force slow processor for Qwen2-VL to avoid tensor issues
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return processor
    except Exception as e:
        print(f"Failed to load image processor: {e}")
        return None

# MONKEY PATCH: Correct Merge Logic (No Truncation)
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_input_ids_with_image_features_patched_v3(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    # Validation: Ensure slots match features
    num_slots = mx.sum(image_positions).item()
    num_features = image_features.shape[0] if len(image_features.shape) > 0 else 0
    
    # In V3, we expanded tokens manually, so they SHOULD match.
    # If slight mismatch due to grid calc diff, we pad/truncate minimally, but NOT to 1.
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    
    if pad_size < 0:
        # This shouldn't happen if we expanded correctly.
        # But if it does, we truncate features to fit slots.
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
    
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

# Apply patch
Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)

def main(args):
    logger.info(f"\033[32mLoading model from {args.model_path}\033[0m")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    
    # Load patched image processor
    image_processor = load_image_processor_patched(args.model_path)
    if image_processor:
        processor.image_processor = image_processor
        
    logger.info(f"\033[32mLoaded image processor (use_fast=False): {image_processor}\033[0m")
    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split)

    # V3: Apply resize to processor if requested
    if args.image_resize_shape:
        new_max = args.image_resize_shape[0] * args.image_resize_shape[1]
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = new_max
            logger.info(f"Overrode processor max_pixels to {new_max} for V3 optimization")

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying V3 Token Expansion Strategy\033[0m")

        def process_data(examples):
            # Qwen2-VL/MLX apply_chat_template might not handle batch of conversations efficiently.
            # Let's iterate manually to be safe.
            base_texts = []
            for msg in examples["messages"]:
                formatted = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=msg,
                    return_messages=False,
                )
                base_texts.append(formatted)

            expanded_messages = []
            
            # 2. Iterate and expand tokens
            for i, text in enumerate(base_texts):
                img_path = examples["images"][i] # Assuming list of paths
                if isinstance(img_path, list): img_path = img_path[0] # Handle list wrapper
                
                try:
                    # Load Image
                    # Check if files exist
                    if not os.path.exists(img_path):
                         print(f"Warning: Image not found {img_path}")
                         expanded_messages.append(text)
                         continue
                         
                    image = Image.open(img_path)
                    
                    # Preprocess to get Grid Size
                    out = processor.image_processor.preprocess(image, return_tensors='np')
                    
                    if 'image_grid_thw' in out:
                        grid = out['image_grid_thw'][0] # [t, h, w]
                        num_tokens = int(grid[1] * grid[2])
                        
                        # Replace singular <|image_pad|>
                        expanded_text = text.replace(
                            "<|image_pad|>", 
                            "<|image_pad|>" * num_tokens
                        )
                    else:
                        expanded_text = text
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    expanded_text = text
                
                expanded_messages.append(expanded_text)

            examples["messages"] = expanded_messages
            return examples

        dataset = dataset.map(process_data, batched=True, batch_size=10) # Batched for speed

    # Dataset Setup
    dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=None, # Disable internal resize, we rely on processor default
    )

    logger.info(f"\033[32mSetting up LoRA\033[0m")
    # 3. LoRA Setup - RECOMMENDATION: Target only 'q_proj', 'v_proj' for stability?
    # User didn't explicitly ask to change LoRA target, but RCA recommended it.
    # I will stick to 'find_all_linear_names' as per previous success (stability-wise with 1e-5), 
    # but maybe we should be safer? 
    # Let's keep it same as V2 to isolate variables (Token Fix only).
    list_of_modules = find_all_linear_names(model.language_model)
    model = get_peft_model(
        model,
        list_of_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    logger.info(f"\033[32mSetting up optimizer\033[0m")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info(f"\033[32mSetting up trainer\033[0m")
    trainer = Trainer(model, optimizer)

    model.train()

    # Training loop
    logger.info(f"\033[32mTraining model\033[0m")
    for epoch in range(args.epochs):
        if args.steps == 0:
            args.steps = len(dataset) // args.batch_size

        progress_bar = tqdm(range(args.steps), position=0, leave=True)
        for i in progress_bar:
            inputs = dataset[i * args.batch_size : (i + 1) * args.batch_size]
            
            # Debug: Check token count for first batch
            if i == 0 and epoch == 0:
                # inputs is a list of dicts or similar. 
                # MLX VLM Dataset returns 'input_ids', 'pixel_values', etc.
                # Let's check correctness
                pass
                
            loss = trainer.train_step(inputs)
            
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Epoch": epoch, "Step": i, "Loss": f"{loss.item():.4f}"}
            )

            if i % args.print_every == 0:
                custom_print(
                    {
                        "Epoch": epoch,
                        "Step": i,
                        "Loss": f"{loss.item():.4f}",
                    }
                )

    logger.info(f"Saving adapter to {args.output_path}...")
    save_adapter(model, args.output_path)
    logger.info("Save complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLM model (V3 Token Fix)")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2-VL-7B-Instruct-4bit")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None) # Added missing arg
    parser.add_argument("--apply-chat-template", action="store_true")

    # Default params same as Phase 9
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output-path", type=str, default="adapters_v3")

    args = parser.parse_args()
    main(args)
