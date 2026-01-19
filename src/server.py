import os
import sys

# Ensure project root is in path when running from src/
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Change to project root for relative paths

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.rag_engine import RAGEngine
import shutil
import json
import gc
import mlx.core as mx
from transformers import AutoProcessor
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import requests
from typing import Optional

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="data/pokemon/images"), name="images")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
rag_engine = None
allowed_rag_ids = None  # Gen 1-2 only for fair comparison
# We do NOT keep VLM models globally anymore to save RAM

MODEL_PATH = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
FUSED_MODEL_PATH = "models/fused_qwen2_vl_4bit_quantized"

@app.on_event("startup")
async def startup_event():
    global rag_engine, allowed_rag_ids
    print("🚀 Starting Server (Sequential Mode)...")
    
    # Load RAG Engine (Small enough to keep resident)
    print("Loading RAG Engine...")
    rag_engine = RAGEngine(db_path="./chroma_db")
    print("RAG Engine Loaded.")
    
    # Load allowed IDs (Gen 1-2 only from train.jsonl)
    print("Loading Gen 1-2 IDs for RAG filtering...")
    allowed_rag_ids = set()
    train_path = "data/pokemon/train.jsonl"
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "images" in entry:
                        img_path = entry["images"][0]
                        filename = os.path.basename(img_path)
                        allowed_rag_ids.add(filename)
                except:
                    continue
    print(f"Loaded {len(allowed_rag_ids)} Gen 1-2 Pokemon IDs for RAG.")

def cleanup_memory():
    """Force garbage collection and clear MLX cache."""
    gc.collect()
    try:
        mx.metal.clear_cache()
    except:
        pass

class RobustImageProcessorWrapper:
    """
    Wraps a generic ImageProcessor to force return_tensors='pt' and convert to numpy.
    This bypasses the 'Only returning PyTorch tensors is currently supported' error
    when libraries request 'np' but the processor refuses.
    """
    def __init__(self, processor):
        self.processor = processor
        # Copy attributes
        if hasattr(processor, "image_processor"): # If it's a wrapper itself
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
            elif isinstance(v, list) and hasattr(v[0], "numpy"): # List of tensors
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

def run_inference_task(image_path, model_path, adapter_path=None, task_name="Inference"):
    print(f"[{task_name}] Loading Model...")
    try:
        # Load with use_fast=False attempt
        model, processor = load(model_path, adapter_path=adapter_path, processor_config={"trust_remote_code": True, "use_fast": False})
        
        # *** ROBUST FIX: Wrap the image processor ***
        if hasattr(processor, "image_processor"):
            print(f"[{task_name}] Applying Robust Processor Wrapper...")
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)

        # Patch for speed if available
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
             processor.image_processor.max_pixels = 512 * 512 

        prompt = "What is the name of this Pokemon? Answer in English and Korean."
        formatted_prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=prompt,
            num_images=1
        )
        
        print(f"[{task_name}] Generating...")
        output = generate(
            model, 
            processor, 
            prompt=formatted_prompt,
            image=image_path,
            max_tokens=100, 
            temperature=0.1
        )
        print(f"[{task_name}] Done.")
        
        # Cleanup
        del model
        del processor
        cleanup_memory()
        return output
    except Exception as e:
        print(f"[{task_name}] Error: {e}")
        return f"Error: {str(e)}"
    finally:
        cleanup_memory()

@app.post("/analyze")
async def analyze(file: UploadFile = File(None), url: str = Form(None)):
    cleanup_memory() # Start fresh
    os.makedirs("uploads", exist_ok=True)
    
    if url:
        try:
            filename = url.split("/")[-1].split("?")[0]
            if not filename: filename = "downloaded_image.jpg"
            file_path = f"uploads/{filename}"
            print(f"Downloading image from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
        except Exception as e:
            return {"error": f"Failed to download image from URL: {str(e)}"}
    elif file:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        return {"error": "No file or URL provided"}
        
    print(f"Analyzing image: {file_path}")
    
    response_data = {
        "vanilla": "Waiting...",
        "rag": {},
        "tuned": "Waiting..."
    }

    # 1. RAG Search (Fastest)
    print("[RAG] Searching...")
    try:
        results = rag_engine.search(file_path, top_k=1, allowed_ids=allowed_rag_ids)
        if results['ids']:
            best_id = results['ids'][0][0]
            best_dist = results['distances'][0][0]
            best_meta = results['metadatas'][0][0]
            response_data["rag"] = {
                "id": best_id,
                "distance": float(best_dist),
                "caption": best_meta.get("caption", "No caption"),
                "image_url": f"/images/{os.path.basename(best_meta.get('path', ''))}"
            }
        else:
            response_data["rag"] = {"error": "No results found"}
    except Exception as e:
        response_data["rag"] = {"error": str(e)}

    # 2. Vanilla Inference
    response_data["vanilla"] = run_inference_task(
        file_path, 
        MODEL_PATH, 
        adapter_path=None, 
        task_name="Vanilla"
    )

    # 3. Tuned Inference (Fused Model - No Adapter Needed)
    response_data["tuned"] = run_inference_task(
        file_path, 
        FUSED_MODEL_PATH, 
        adapter_path=None, 
        task_name="Tuned (Fused)"
    )
    
    return response_data
