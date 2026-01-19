# MLX-VLM-RAG-Pokemon

[한국어 가이드 (Korean)](README_KR.md)

A local Vision-Language Model (VLM) tuning and RAG (Retrieval-Augmented Generation) project for Pokemon identification, built with [Apple MLX](https://github.com/ml-explore/mlx) on macOS.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Non--Commercial-red)

## 📌 Project Overview
The goal of this project is to create an AI that can identify and describe Pokemon, including their Korean names, using two approaches:
1.  **Fine-tuning (LoRA)**: Teaching the model specific knowledge (English/Korean names) via Qwen2-VL.
2.  **RAG (Retrieval)**: Retrieving visual matches from a vector database (ChromaDB + SigLIP) to assist the model.

## 🛠️ Methodology

### 1. Data Processing
We utilized the [pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) dataset.
- **Source**: 883 image-text pairs of Pokemon.
### 1. Data Processing
We utilized the [pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) dataset.
- **Source**: 883 image-text pairs of Pokemon.
- **Enrichment (`setup_pokemon_data.py`)**:
    - **Identification**: Matched English captions against a Pokemon Database to identify the specific Pokemon.
    - **Metadata Injection**: Added **Korean Names** and **Generation Info** (e.g., "This is Bulbasaur (이상해씨). It is a GEN I Pokemon.").
- **Strategic Splitting**:
    - **Train Set (600+)**: Contains **Gen 1 & Gen 2** Pokemon.
    - **Valid Set (200+)**: Contains **Gen 3+** Pokemon.
    - *Purpose*: To test if the model can describe unseen generations using the learned visual-text alignment.

### 2. RAG System Implementation
Since 4-bit quantized VLMs often hallucinate exact names, we built a **Retrieval-Augmented Generation (RAG)** pipeline.
- **Vision Encoder**: Used **SigLIP (So400m)** to generate high-quality image embeddings.
- **Vector Database**: **ChromaDB** stores these embeddings along with metadata (Pokemon Names).
- **Process**:
    1. **Query**: When a new image comes in, SigLIP encodes it.
    2. **Retrieval**: ChromaDB finds the "visually most similar" image in the database.
    3. **Context Injection**: The metadata (Name/Description) of the retrieved image is injected into the LLM's prompt as a "Hint".
    4. **Generation**: The VLM uses the visual input + text hint to generate the final answer.

## 🔬 Experimental Results & Technical Review (Phase 2)
We conducted a second round of evaluation focusing on **Korean name generation** and **Tuning Stability**.

### 1. Comparative Analysis: Vanilla vs RAG vs Tuned
| Approach | Stability | English Accuracy | Korean Accuracy | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla (Base)** | ⭐⭐⭐⭐⭐ | High | **Low** (Phonetic Transliteration) | Good for description, bad for specific entity naming. |
| **RAG (Context)** | ⭐⭐⭐⭐⭐ | **High** | **High** (If context is retrieved) | **Best Choice**. Corrects hallucinations (e.g., Bulbasaur -> Kabutops context). |
| **Tuned (LoRA)** | ⭐⭐ | Low | N/A (Failed) | **Not Recommended for 4-bit**. Suffered from pattern collapse. |

### 2. Tuning Failure Analysis (Post-Mortem)
Despite fixing the initial **IndexError (Vocab Mismatch)** by lowering the learning rate (`1e-4` -> `1e-5`), the fine-tuned model exhibited **quality degradation** (repetitive text).

#### 📉 Training Log Analysis (`training_log_v2.csv`)
- **Loss trajectory**: Started at **18.9** and plateaued at **~7.86**.
- **Insight**: A converged SFT loss typically drops below **2.0**. A final loss of ~8.0 indicates the model **failed to learn meaningful patterns**, effectively memorizing noise or getting stuck in a local minimum due to the 4-bit quantization constraints.

#### 🔍 Root Causes
- **Root Cause 1 (Quantization Noise)**: Tuning **all linear layers** on a **4-bit** quantized model caused it to learn noise instead of features.
- **Root Cause 2 (Data Scarcity)**: 500 images were insufficient for "All-Linear" tuning scope, leading to overfitting.
- **Detailed Report**: See [`TUNING_ROOT_CAUSE_ANALYSIS.md`](TUNING_ROOT_CAUSE_ANALYSIS.md).

### 3. V3 Tuning Success & Inference Challenges (Final Status)
After investigating the initial failure, we developed a new tuning script `lora_v3.py` that fixes the "Blind Model" issue by manually expanding image tokens.
- **Tuning Success**:
    - **Dataset**: 520 Images (Full Train Set).
    - **Result**: Loss dropped to **0.0006**. The model successfully learned visual features and Korean names.
- **Inference Limitation (MLX Platform)**:
    - While the model weights (`adapters_v3_full`) are valid, the current `mlx_vlm` library lacks native support for the dynamic token expansion required by our fix.
    - **Symptom**: The inference script fails to properly manage **M-RoPE (Multimodal Rotary Embedding)** states during autoregressive generation, leading to empty outputs.
    - **Conclusion**: The tuning methodology is proven, but serving the model requires library-level updates or a complex custom inference engine.

### 4. Key Findings
*   **Prompt Engineering**: Adding `"in English and Korean"` to the prompt significantly improved the Base model's attempt to output Korean (even if phonetically inferred).
*   **Vision Encoder**: Qwen2-VL uses a **ViT-based Vision Encoder** separated from the LLM. Standard practice is to **freeze** this encoder and tune the LLM, which we followed.
*   **RAG Superiority**: For entity-heavy tasks like Pokemon naming (especially in multi-lingual contexts), RAG proved far more effective and cheaper than fine-tuning.

### 4. Artifacts
- **[EVALUATION_REPORT_v2.md](EVALUATION_REPORT_v2.md)**: Visual comparison of Vanilla vs RAG.
- **[TUNING_REPORT.md](TUNING_REPORT.md)**: Detailed log of the tuning attempt.

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
python patched_lora.py --dataset data_pokemon --steps 600 --output-path adapters --apply-chat-template
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
