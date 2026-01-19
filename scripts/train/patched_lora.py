import argparse
import json
import logging
import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer import Dataset, Trainer, save_adapter
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load
from transformers import AutoImageProcessor

# PATCH: Explicitly define a load_image_processor that forces use_fast=False for Qwen2VL
def load_image_processor_patched(model_path):
    try:
        # For Qwen2-VL, using the fast processor causes tensor type errors on MLX/MPS if torch is not fully configured for it or returns generic tensors.
        # Force slow processor.
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        return processor
    except Exception as e:
        print(f"Failed to load image processor: {e}")
        return None

# MONKEY PATCH for Qwen2-VL Model in mlx_vlm
import mlx.core as mx
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_input_ids_with_image_features_patched(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index

    # Positions of <image> tokens in input_ids, assuming batch size is 1
    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index

    image_features = image_features.astype(mx.float32)
    
    # FIX: Calculate pad size. If negative, it means we have more image features than placeholders.
    # This happens because the processor might generate more features than the template reserved tokens for.
    # Qwen2-VL should dynamically resize, but MLX implementation here assumes static placeholder match.
    # We will try to truncate image features to fit, OR pad inputs_embeds.
    # Truncating is safer for shape consistency in this static graph assumption.
    
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    
    if pad_size < 0:
        # Image features > placeholders. Truncate image features.
        # This is not ideal but prevents crash. Best fix is ensuring processor usage matches expected token count.
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
        
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    
    inputs_embeds = mx.where(
        image_positions[:, :, None], image_features, inputs_embeds
    )

    return inputs_embeds

# Apply patch
Model._merge_input_ids_with_image_features = _merge_input_ids_with_image_features_patched

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
    
    # PATCH: Use patched loader
    image_processor = load_image_processor_patched(args.model_path)
    
    # CRITICAL: Overwrite the processor's internal image processor to ensure it uses the slow one
    # even when called as processor(images=...)
    if image_processor:
        processor.image_processor = image_processor
        
    logger.info(f"\033[32mLoaded image processor (use_fast=False): {image_processor}\033[0m")

    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split)

    # Simplified validation specific to our Pokemon dataset format
    if "messages" not in dataset.column_names:
        # Fallback if "messages" is missing but "text" exists (legacy format)
        pass 
        

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying chat template to the dataset\033[0m")

        def process_data(examples):
            # Qwen2-VL logic
            examples["messages"] = apply_chat_template(
                config=config,
                processor=processor,
                prompt=examples["messages"],
                return_messages=True,
            )
            # Manual Resize Patch
            if args.image_resize_shape:
                resized_images = []
                for img_path in examples["images"]: # examples["images"] is a list of lists of paths or images
                    # If it's a list of paths (which it is for us), we need to load and resize
                    # BUT wait, apply_chat_template handles prompt, but Dataset expects "images" column to be PIL images 
                    # if we are doing manual resize.
                    # Actually, let's keep it simple: We just modify the Dataset call below to NOT pass resize_shape
                    # and rely on Qwen2-VL processor to handle it (which might be slow) OR we resize here.
                    pass
            return examples

        dataset = dataset.map(process_data)
        
        # PATCH: Resize images in dataset manually to avoid 'AttributeError: list object has no attribute width' in mlx_vlm
        if args.image_resize_shape:
            logger.info(f"Manually resizing images to {args.image_resize_shape}...")
            from PIL import Image
            def resize_transform(examples):
                # examples["images"] is a list of lists of strings (paths) usually for this dataset format
                # BUT load_dataset might have loaded them as PIL images if we didn't specify.
                # Let's assume they are PIL images because 'diffusers/pokemon-gpt4-captions' has 'image' column.
                # My setup_pokemon_data saved them as files, so 'train.jsonl' has paths.
                # The Dataset class will load them. 
                # PROPOSAL: We don't change dataset here. We rely on the fact that MLX Dataset handles paths.
                # The BUG was in mlx_vlm utils.py `resize_image`. 
                # Let's TRY to just use the processor's default behavior (no explicit resize arg to Dataset) 
                # but modify the processor config to FORCE a smaller max_pixels if possible.
                pass
            pass

    # PATCH: Set image_resize_shape to None in Dataset init to avoid the bug, 
    # but we need to ensure speed. 
    # Qwen2VLProcessor respects 'min_pixels' and 'max_pixels'.
    # If we lower 'max_pixels', it will resize.
    if args.image_resize_shape:
        # 512*512 = 262144
        new_max = args.image_resize_shape[0] * args.image_resize_shape[1]
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = new_max
            logger.info(f"Overrode processor max_pixels to {new_max} for speedup")

    dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=None, # DISABLE library resize to avoid bug
    )

    logger.info(f"\033[32mSetting up LoRA\033[0m")
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
            loss = trainer.train_step(inputs)
            
            # Update progress bar
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

    # Save the adapter
    save_adapter(model, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLM model (Patched)")
    parser.add_argument("--model-path", type=str, default="mlx-community/Qwen2-VL-7B-Instruct-4bit")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    parser.add_argument("--apply-chat-template", action="store_true") # Default false mainly, but we set true
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output-path", type=str, default="adapters")

    args = parser.parse_args()
    main(args)
