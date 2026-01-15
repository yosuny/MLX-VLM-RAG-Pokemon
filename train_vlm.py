import argparse
import subprocess
import sys
import os

def train(args):
    print(f"Starting LoRA training for {args.model}...")
    
    # We use subprocess to call mlx_vlm.lora to ensure isolation and use official CLI entry point logic
    # basic command structure: python -m mlx_vlm.lora --model <name> --train --data <dir> --iters <N>
    
    # Ensure data directory exists
    if not os.path.exists(args.data):
        print(f"Error: Data path {args.data} does not exist.")
        return

    # Construct command
    # Updated based on actual usage output:
    # usage: lora.py [--model-path] --dataset [--steps] [--output-path] ...
    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", args.model,
        "--dataset", args.data,
        "--batch-size", str(args.batch_size),
        "--steps", str(args.iters),
        "--output-path", args.adapter_path
    ]
    
    # Check if a chat template is needed or handled by --apply-chat-template
    # For now, we omit it unless needed, but Qwen-VL often needs one.
    # We will assume JSONL has keys "image" and "text" or "messages".
    # If standard MLX VLM, it likely needs default handling.
    
    # Optional: Add --apply-chat-template if the dataset is in chat format
    # cmd.append("--apply-chat-template")
    
    if args.qlora:
        # Some versions might default to qlora or have a flag. 
        # For 4bit model, it usually automatically handles it.
        pass

    print("Running command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nTraining completed! Adapters saved to {args.adapter_path}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2-VL-7B-Instruct-4bit")
    # Note: mlx_vlm.lora expects a DIRECTORY containing train.jsonl for --data usually, 
    # but let's point to the directory of the file if a file is passed.
    parser.add_argument("--data", type=str, default="data", help="Directory containing train.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (implied by 4bit model usually)")
    
    args = parser.parse_args()
    
    # Adjust data path if user provided a file instead of dir
    if args.data.endswith(".jsonl"):
        args.data = os.path.dirname(args.data)
        
    train(args)
