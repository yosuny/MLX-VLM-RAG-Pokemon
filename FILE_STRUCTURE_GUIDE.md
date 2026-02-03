# File Structure Guide

This document maps key files to their purpose and corresponding experimental phase.

## 📂 Folder Structure

```
mlx-vlm-rag-pokemon/
├── README.md / README_KR.md    # Project documentation (with Lessons Learned)
├── FILE_STRUCTURE_GUIDE.md     # This file
├── requirements.txt
│
├── src/                        # Core application code
│   ├── rag_engine.py           # SigLIP-based image retrieval
│   ├── server.py               # FastAPI backend (Web UI, uses Fused Model)
│   ├── demo_rag.py             # CLI RAG demo
│   └── pokemon_info.py         # Pokemon metadata
│
├── scripts/                    # Training & Evaluation
│   ├── train/
│   │   ├── lora_v3.py          # [Phase 3] V3 Tuning Script (Success!)
│   │   ├── fuse_vlm.py         # [Phase 4] Dequantize + Fuse LoRA script
│   │   └── quantize_vlm.py     # [Phase 4] Re-quantize fused model to 4-bit
│   ├── eval/
│   │   ├── evaluate_models_v3.py  # Final Eval (Generic Prompt)
│   │   └── evaluate_models_v4.py  # Final Eval (Hinted Prompt)
│   └── setup/
│       └── setup_pokemon_data.py  # Downloads and prepares data
│   │
│   ├── debug/                  # Debug & Utility Scripts
│   │   ├── test_fused_manual.py    # Manual verification of fused model
│   │   ├── patch_quant_config.py   # Injects quantization config into JSON
│   │   └── debug_rag_retrieval.py  # Debugging RAG metadata extraction
│
├── docs/                       # Reports & Logs
│   ├── reports/
│   │   ├── EVALUATION_REPORT_v3.md   # Final Results (Generic)
│   │   ├── EVALUATION_REPORT_v4.md   # Final Results (Hinted)
│   │   └── PHASE_2_TUNING_FAILURE_ANALYSIS.md
│   └── logs/
│       └── eval/               # Evaluation logs (if preserved)
│
├── models/                     # Model weights (gitignored/partially tracked)
│   └── fused_qwen2_vl_4bit_quantized/ # Final standalone version
│
├── data/                       # Datasets (gitignored)
│   └── pokemon/                # Images & JSONL
│
└── static/                     # Web UI assets
    └── index.html
```

## 🔑 Key File Mapping

| File | Phase | Description |
| :--- | :--- | :--- |
| `scripts/train/lora_v3.py` | **Phase 3** | Manual token expansion tuning. Loss → 0.0006. |
| `scripts/train/fuse_vlm.py` | **Phase 4** | Fusing LoRA adapters into base model weights. |
| `scripts/train/quantize_vlm.py` | **Phase 4** | Re-quantizing 16-bit fused model back to 4-bit. |
| `docs/reports/EVALUATION_REPORT_v4.md` | **Final** | Performance comparison after RAG fix. |
| `src/server.py` | Final | Production Web UI using the Fused Model. |

## 🚀 Usage

```bash
# Setup
python scripts/setup/setup_pokemon_data.py

# Start Web UI (Production mode with Fused Model)
uvicorn src.server:app --reload --port 8000

# Evaluation (Final Comparison)
python scripts/eval/evaluate_models_v4.py
```

## 🚫 Excluded from Git

- `data/` - Datasets
- `adapters/` - Intermediate LoRA weights
- `models/` - Large safetensors files (unless using Git LFS)
- `*.safetensors`, `*.log`
