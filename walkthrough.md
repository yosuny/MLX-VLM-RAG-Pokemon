# VLM Image Tuning & RAG Walkthrough

## 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation
We use the updated `setup_pokemon_data.py` which downloads a subset of Pokemon images and adds Korean name mappings.

```bash
# Generate Pokemon dataset (English + Korean labels)
python setup_pokemon_data.py
# Creates: data_pokemon/images/ and data_pokemon/train.jsonl
```

## 3. Image RAG (Retrieval)
The `demo_rag.py` script indexes the images and performs search.

```bash
# Index and Search Test
python demo_rag.py
```
- Indexes images in `data_pokemon/images` using SigLIP.
- Stores vectors in `chroma_db`.
- Performs text-to-image and image-to-image search.

## 4. Image Tuning (LoRA)
We use a patched training script `patched_lora.py` to fix MLX/Torch tensor compatibility issues with Qwen2-VL.

```bash
# Fine-tune Model (30 Steps)
python patched_lora.py --dataset data_pokemon --steps 30
```
- **Model**: `mlx-community/Qwen2-VL-7B-Instruct-4bit`
- **Adapters**: Saved to root directory (`adapters.safetensors`).

## 5. Evaluation
The `evaluate_models.py` script compares the Baseline, RAG, and Tuned models and generates a report.

```bash
# Run 3-Way Evaluation
python evaluate_models.py
```

**Output**: Open `EVALUATION_REPORT.md` to see the visual comparison results.

### Expected Results
- **Vanilla Model**: Good at English names, poor at Korean.
- **RAG Model**: Retrieves exact matches if indexed.
- **Tuned Model**: Should improve Korean naming (Note: Requires careful hyperparameter tuning to avoid generation collapse like `!!!!`).
