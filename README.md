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
- **Split**: Train (Gen 1-2, 600+) / Valid (Gen 3+, 200+) to test generalization.

### 2. RAG System
| Component | Technology |
| :--- | :--- |
| **Vision Encoder** | SigLIP (So400m) |
| **Vector Database** | ChromaDB |
| **Process** | Query Image → SigLIP Embedding → Retrieve Similar → Inject Hint → VLM Generate |

### 3. LoRA Tuning & Model Fusion

#### Why 16-bit Adapter on 4-bit Base?
- LoRA requires **precise gradients**; 4-bit weights cause gradient loss → training instability.
- Solution: Train adapter in **16-bit (Float16)**, then fuse.

#### Fusion Strategy
```
[4-bit Base] → Dequantize → [16-bit] + [16-bit LoRA] → Fuse → Re-Quantize → [4-bit Fused]
```
- **Final Model**: `models/fused_qwen2_vl_4bit_quantized` (**4.3GB**)

## 📊 Final Evaluation Results

| Image | Ground Truth | Vanilla Model | RAG System | Tuned Model |
| :---: | :---: | :--- | :--- | :--- |
| <img src="data/pokemon/images/pokemon_117.jpg" width="100"><br>**Umbreon**<br>(Gen 2) | Umbreon<br>(블래키) | ✅ **Correct**<br>"This is Umbreon." | ✅ **Correct**<br>"This is Umbreon (블래키)." | ✅ **Correct**<br>"This is Umbreon." |
| <img src="data/pokemon/images/pokemon_025.jpg" width="100"><br>**Staryu**<br>(Gen 1) | Staryu<br>(별가사리) | ⚠️ **Hallucination**<br>"Staraptor" | ✅ **Correct**<br>"This is Staryu (별가사리)."<br>*(Fixed via metadata update)* | ❌ **Literal**<br>"Star-shaped object" |
| <img src="data/pokemon/images/pokemon_440.jpg" width="100"><br>**Riolu**<br>(Gen 4) | Riolu<br>(리오르) | ❌ **Hallucination**<br>"Umbreon" | ⚠️ **Gen 1-2 Similar Match**<br>"Umbreon/Glaceon"<br>*(Restricted to Gen 1-2 DB)* | ❌ **Hallucination**<br>"Glaceon" |
| <img src="data/pokemon/images/pokemon_411.jpg" width="100"><br>**Gastrodon**<br>(Gen 4) | Gastrodon<br>(트리토돈) | ❌ **Hallucination**<br>"Gallade" | ⚠️ **Gen 1-2 Similar Match**<br>"Mawile/Slug-like"<br>*(Restricted to Gen 1-2 DB)* | ❌ **Hallucination**<br>"Gallade" |

> **Note on RAG Fairness**: For a fair comparison with the Tuned Model (trained only on Gen 1-2), **RAG retrieval was restricted to Gen 1-2 Pokemon**.
> Therefore, for Gen 3+ queries like Riolu, the RAG system is *expected* to retrieve the most visually similar Gen 1-2 Pokemon rather than the exact match. This confirms the system correctly follows constraints.

### Conclusion
| Approach | Best For |
| :--- | :--- |
| **RAG** | ✅ **Production** (800+ entities, highest accuracy) |
| **Fused** | Response formatting, Korean output style |
| **Vanilla** | Quick prototyping |

> **Recommendation**: Combine **RAG** (for accuracy) + **Fused Model** (for style) for optimal results.

## 📚 Lessons Learned

### 1. MLX LoRA Adapter Inference Issue (M-RoPE)
- **Problem**: After successful LoRA training (Loss 0.0006), the adapter failed to generate output during inference.
- **Root Cause**: `mlx_vlm` library's `generate()` function couldn't properly manage **M-RoPE (Multimodal Rotary Embedding)** states when dynamically expanded image tokens were used.
- **Solution**: **Model Fusion** - Permanently merge LoRA weights into the base model, eliminating runtime adapter loading.
- **Lesson**: When facing library-level limitations, consider **weight fusion** as an alternative to runtime adapter injection.

### 2. EOS Token Learning Failure (Underfitting)
- **Problem**: Phase 1-2 tuning produced repetitive garbage output (`!!!!`).
- **Root Cause**: Training stopped too early (20-30 steps) while loss was still high (~8.0). Model never learned to generate the end-of-sequence token (`<|im_end|>`).
- **Solution**: Increase training to **600+ steps** until loss converges below 1.0.
- **Lesson**: For VLM fine-tuning, **sufficient training steps** are critical. Early stopping before EOS learning leads to infinite generation loops.

### 3. 4-bit Quantization & Gradient Precision
- **Problem**: Training LoRA adapters in 4-bit caused training instability and poor convergence (Loss stuck at ~8.0).
- **Insight**: 4-bit quantized weights lose gradient precision during backpropagation.
- **Lesson**: Always train adapters in **16-bit (Float16)**, then re-quantize after fusion for deployment.

### 4. RAG vs Fine-tuning Trade-off
- **Finding**: For 800+ entity identification, RAG outperformed fine-tuning in both accuracy and cost.
- **Why**: Fine-tuning requires massive data per entity; RAG only needs one indexed image per entity.
- **Lesson**: For **large-scale entity recognition**, prioritize RAG over fine-tuning. Use fine-tuning for **style/format control** only.

### 5. Prompt Engineering Impact
- **Observation**: Adding "Pokemon" hint to prompt triggered hallucinations (Staryu → Staraptor).
- **Insight**: Domain-specific keywords can bias the model toward linguistically similar (but visually incorrect) answers.
- **Lesson**: Test both generic and hinted prompts; sometimes **less context is better**.

### 6. Substring Match Bug in RAG Metadata
- **Problem**: Piplup was incorrectly identified as **Natu (네이티)** in the RAG UI.
- **Root Cause**: The automated labeling script used `if name in caption.lower()`. The word **"signature"** (frequent in captions) contains **"natu"**, triggering a false positive for Natu.
- **Solution**: Updated the script to use **Regex with Word Boundaries** (`\b{name}\b`) to ensure only full Pokemon names are matched.
- **Lesson**: When performing keyword-based automated labeling, always use **word boundaries** to prevent substring-induced mislabeling.

### 7. Code-Level Debugging Insights

| Issue | Symptom | Root Cause | Fix |
| :--- | :--- | :--- | :--- |
| **RAG returning empty hints** | RAG mode same as Vanilla | Script read `documents` (empty) instead of `metadatas['caption']` | Access `results['metadatas'][0][0]['caption']` |
| **PyTorch tensor error** | `ValueError: Only PyTorch tensors supported` | HuggingFace fast image processor incompatible with MLX | Set `use_fast=False` + wrap with numpy converter |
| **Image path mismatch** | `FileNotFoundError` | JSONL had `data_pokemon/` but actual path was `data/pokemon/` | String replace in data loader |
| **Quantize argument order** | Silent wrong quantization | `nn.quantize(model, bits, group_size)` → actually `(model, group_size, bits)` | Check MLX API docs |
| **Config missing quantization info** | Model loads as Float16 | Fused model config.json lacked `quantization` block | Patch config post-save |

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

## 📄 Reports
- [EVALUATION_REPORT_v3.md](docs/reports/EVALUATION_REPORT_v3.md) - Generic prompt evaluation
- [EVALUATION_REPORT_v4.md](docs/reports/EVALUATION_REPORT_v4.md) - Hinted prompt evaluation

## ⚠️ Disclaimer
- **Unofficial Project**: Not affiliated with Nintendo, Game Freak, or The Pokémon Company.
- **Dataset**: Non-commercial use only per [diffusers/pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) license.

## 🤝 Acknowledgements
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
