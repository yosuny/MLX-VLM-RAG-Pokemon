# MLX-VLM-RAG-Pokemon

A local Vision-Language Model (VLM) tuning and RAG (Retrieval-Augmented Generation) project for Pokemon identification, built with [Apple MLX](https://github.com/ml-explore/mlx) on macOS.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Non--Commercial-red)

## 📌 Project Overview
The goal of this project is to create an AI that can identify and describe Pokemon, including their Korean names, using two approaches:
1.  **Fine-tuning (LoRA)**: Teaching the model specific knowledge (English/Korean names) via Qwen2-VL.
2.  **RAG (Retrieval)**: Retrieving visual matches from a vector database (ChromaDB + SigLIP) to assist the model.

## 🚀 Features
*   **Data Pipeline**: Downloads Pokemon image-text pairs and augments them with Korean names.
*   **VLM RAG**: Image-to-Image retrieval using SigLIP embeddings.
*   **LoRA Fine-tuning**: Patched training script for Qwen2-VL on MLX.
*   **Evaluation**: Automated comparison of Vanilla vs. RAG vs. Tuned models against Ground Truth.

## ⚠️ Disclaimer & License
*   **Unofficial Project**: This is a fan-made educational project and is not affiliated with, endorsed, sponsored, or specifically approved by Nintendo, Game Freak, or The Pokémon Company.
*   **Trademark**:  Pokemon and character names are trademarks of Nintendo.
*   **Dataset**: The dataset used (`diffusers/pokemon-gpt4-captions`) requires adherence to its non-commercial license terms. **Do not distribute the image files commercially.**

## 🛠️ Usage
### 1. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Downloads data and creates Korean name mappings
python setup_pokemon_data.py
```
> **Note**: This script downloads data locally. The images are NOT included in this repository to respect copyright.

### 3. Run RAG Demo
```bash
python demo_rag.py
```

### 4. Fine-Tune (LoRA)
```bash
python patched_lora.py --dataset data_pokemon --steps 30
```

### 5. Evaluate
```bash
python evaluate_models.py
```
Check `EVALUATION_REPORT.md` for visual results.

## 📁 Repository Structure
*   `setup_pokemon_data.py`: Data downloader & preprocessor (adding Korean labels).
*   `data_pokemon/`: (Excluded) Local folder for images and JSONL.
*   `rag_engine.py`: SigLIP-based image retrieval engine.
*   `patched_lora.py`: Customized MLX-VLM training script.
*   `evaluate_models.py`: Comparative evaluation script.
*   `EVALUATION_REPORT.md`: Generated report.

## 🤝 Acknowledgements
*   [Apple MLX](https://github.com/ml-explore/mlx)
*   [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
*   [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
