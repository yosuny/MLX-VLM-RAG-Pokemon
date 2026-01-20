# MLX-VLM-RAG-Pokemon

[한국어 가이드 (Korean)](README_KR.md)

A local Vision-Language Model (VLM) tuning and RAG (Retrieval-Augmented Generation) project for Pokemon identification, built with [Apple MLX](https://github.com/ml-explore/mlx) on macOS.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Non--Commercial-red)

## 📌 Project Overview
An AI system that identifies and describes 800+ Pokemon with Korean names using:
1. **RAG (Retrieval)**: Visual similarity search via ChromaDB + SigLIP for accurate naming.
2. **Fine-tuning (LoRA)**: Custom LoRA adapter fused into a 4-bit Qwen2-VL model.

## 🛠️ Methodology

### 1. Data Processing
- **Source**: [pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) (883 image-text pairs)
- **Enrichment**: Added **Korean Names** and **Generation Info** (e.g., "This is Bulbasaur (이상해씨). GEN I.")
- **Split**: Train (Gen 1-2, 520) / Valid (Gen 3+, 313) to test generalization.

### 2. RAG System
| Component | Technology |
| :--- | :--- |
| **Vision Encoder** | SigLIP (So400m) |
| **Vector Database** | ChromaDB |
| **Process** | Query Image → SigLIP Embedding → Retrieve Similar → Inject Hint → VLM Generate |
| **Filtering** | Gen 1-2 only (for fair comparison with Tuned model) |

### 3. LoRA Tuning & Model Fusion

#### Why 16-bit Adapter on 4-bit Base?
- LoRA requires **precise gradients**; 4-bit weights cause gradient loss → training instability.
- Solution: Train adapter in **16-bit (Float16)**, then fuse.

#### Fusion Strategy
```
[4-bit Base] → Dequantize → [16-bit] + [16-bit LoRA] → Fuse → Re-Quantize → [4-bit Fused]
```
- **Final Model**: `models/fused_qwen2_vl_4bit_quantized` (**4.3GB**)

### 4. Multi-Stage Evaluation
We conducted 4 evaluation phases to rigorously test all approaches:

| Version | Test Type | Purpose |
| :--- | :--- | :--- |
| v3 | Generic Prompt | Baseline comparison without hints |
| v4 | Hinted Prompt | Test with "Pokemon" keyword hint |
| v5 | OOD Trap Test | Test on **unseen Pokemon species** (Gen 3+) |
| v6 | Generalization | Test on **unseen images of known Pokemon** |

## 📊 Evaluation Results

### Quick Comparison
| Metric | Vanilla | RAG | Tuned |
| :--- | :---: | :---: | :---: |
| **Gen 1-2 Accuracy (v5)** | 23.5% | **70.6%** | 17.6% |
| **OOD (Gen 3+) Accuracy (v5)** | 16.7% | 16.7%* | 16.7% |
| **Generalization (v6)** | 86.7% | **100%** | 80.0% |

> *RAG is restricted to Gen 1-2 DB for fair comparison, so Gen 3+ matches are intentionally limited.

### Key Findings

1. **RAG dominates accuracy**: 100% on generalization test, 70.6% on trained data
2. **Tuned model overfits**: Performs worse than Vanilla on new images of the same Pokemon (80% vs 86.7%)
3. **Tuned model hallucinates on OOD**: Confidently names unseen Pokemon incorrectly (e.g., Lickilicky → "Gastly")

### Sample Results (v3/v4)

| Image | Ground Truth | Vanilla | RAG | Tuned |
| :---: | :---: | :--- | :--- | :--- |
| <img src="docs/reports/assets/images/pokemon_117.jpg" width="80"><br>**Umbreon** | Umbreon<br>(블래키) | ✅ Correct | ✅ Correct + Korean | ✅ Correct |
| <img src="docs/reports/assets/images/pokemon_025.jpg" width="80"><br>**Staryu** | Staryu<br>(별가사리) | ⚠️ "Staraptor" | ✅ Correct + Korean | ❌ "Star-shaped object" |

### Conclusion
| Approach | Best For |
| :--- | :--- |
| **RAG** | ✅ **Production** (800+ entities, highest accuracy, best generalization) |
| **Fused** | Response formatting, Korean output style |
| **Vanilla** | Quick prototyping |

> **Recommendation**: Use **RAG** for accuracy. Fine-tuning is useful for **style/format control** only, not for knowledge injection.

## 📚 Lessons Learned

### 1. MLX LoRA Adapter Inference Issue (M-RoPE)
- **Problem**: After successful LoRA training (Loss 0.0006), the adapter failed to generate output during inference.
- **Root Cause**: `mlx_vlm` library's `generate()` function couldn't properly manage **M-RoPE (Multimodal Rotary Embedding)** states when dynamically expanded image tokens were used.
- **Solution**: **Model Fusion** - Permanently merge LoRA weights into the base model.

### 2. EOS Token Learning Failure (Underfitting)
- **Problem**: Early tuning produced repetitive garbage output (`!!!!`).
- **Root Cause**: Training stopped too early (20-30 steps) while loss was still high (~8.0).
- **Solution**: Increase training to **600+ steps** until loss converges below 1.0.

### 3. Why Use 16-bit Instead of 4-bit?
- **Training**: 4-bit precision is too coarse to capture the **subtle weight updates (gradients)** needed for learning. These updates simply vanish to zero. Thus, training **must** use 16-bit.
- **Fusion**: You cannot directly add 16-bit LoRA weights to a 4-bit model. You must **dequantize** the base model to 16-bit, merge weights, and then re-quantize to 4-bit.

### 4. RAG vs Fine-tuning Trade-off
- **Finding**: For 800+ entity identification, RAG outperformed fine-tuning in both accuracy and cost.
- **Lesson**: For **large-scale entity recognition**, prioritize RAG over fine-tuning.

### 5. Substring Match Bug in RAG Metadata
- **Problem**: "signature" matched "natu" → Piplup was labeled as Natu.
- **Solution**: Use **Regex with Word Boundaries** (`\b{name}\b`).

### 6. Trade-off: Fine-tuning vs. Visual Generalization (Regression)
- **Evidence**: In v6 evaluation, the Vanilla model correctly identified **Dragonite**, but the Tuned model misclassified it as **Charizard**.
- **Insight**: Training on a small dataset (~500 images) constrained the model to the training distribution, **suggesting a suspected regression** in pre-trained visual discrimination capabilities for unseen image styles.
- **Lesson**: If generalization is critical, relying solely on fine-tuning is risky. **RAG** or **Ensemble** methods are safer.

## 🚀 Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare Data
python scripts/setup/setup_pokemon_data.py

# 3. Run Web UI
uvicorn src.server:app --reload --port 8000
# Open http://localhost:8000/static/index.html
```

## 📁 Key Files
| File | Description |
| :--- | :--- |
| `src/rag_engine.py` | SigLIP + ChromaDB retrieval engine |
| `src/server.py` | FastAPI inference server |
| `scripts/train/lora_v3.py` | LoRA training script |
| `scripts/train/fuse_vlm.py` | Model fusion script |
| `models/fused_qwen2_vl_4bit_quantized/` | Final fused model (4.3GB) |

## 📄 Detailed Reports
| Report | Description |
| :--- | :--- |
| [v3 - Generic Prompt](docs/reports/EVALUATION_REPORT_v3.md) | Baseline without "Pokemon" hint |
| [v4 - Hinted Prompt](docs/reports/EVALUATION_REPORT_v4.md) | With "What Pokemon is this?" |
| [v5 - OOD Trap Test](docs/reports/EVALUATION_REPORT_v5_OOD_TRAP_TEST.md) | Gen 3+ unseen species test |
| [v6 - Generalization](docs/reports/EVALUATION_REPORT_v6_GENERALIZATION.md) | Unseen images of known Pokemon |

## ⚠️ Disclaimer
- **Unofficial Project**: Not affiliated with Nintendo, Game Freak, or The Pokémon Company.
- **Dataset**: Non-commercial use only per [diffusers/pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) license.

## 🤝 Acknowledgements
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
