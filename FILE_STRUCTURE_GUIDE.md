# File Structure Guide

This document maps key files to their purpose and corresponding experimental phase.

## 📂 Folder Structure

```
mlx-vlm-rag-pokemon/
├── README.md / README_KR.md    # Project documentation
├── FILE_STRUCTURE_GUIDE.md     # This file
├── requirements.txt
│
├── src/                        # Core application code
│   ├── rag_engine.py           # SigLIP-based image retrieval
│   ├── server.py               # FastAPI backend (Web UI)
│   ├── demo_rag.py             # CLI RAG demo
│   └── pokemon_info.py         # Pokemon metadata (308k+ entries)
│
├── scripts/                    # Training & Evaluation
│   ├── train/
│   │   ├── lora_v3.py          # [Phase 3] V3 Tuning Script (Success!)
│   │   ├── patched_lora.py     # [Phase 2] Legacy (Blind Model Issue)
│   │   └── train_vlm.py        # [Phase 1] Original script
│   ├── eval/
│   │   ├── evaluate_models.py
│   │   ├── evaluate_models_v2.py
│   │   └── evaluate_tuned_only.py
│   └── setup/
│       ├── setup_pokemon_data.py  # Downloads and prepares data
│       └── setup_eval_v2.py
│
├── tools/                      # Debug & Utility Scripts (Reference)
│   ├── debug_chat_template*.py # Token expansion debugging
│   ├── debug_v3_tokens.py      # V3 token verification
│   ├── inference_v3_custom.py  # Custom inference (M-RoPE issue)
│   └── ...
│
├── docs/                       # Reports & Logs
│   ├── reports/
│   │   ├── PHASE_2_TUNING_FAILURE_ANALYSIS.md
│   │   └── PHASE_2_VS_RAG_EVALUATION.md
│   └── logs/
│       ├── phase_2_training_log.csv
│       └── phase_3_training_log.txt
│
├── data/                       # Datasets (gitignored)
│   ├── pokemon/                # Main Pokemon dataset (images, train.jsonl, validation.jsonl)
│   ├── eval_v2/                # Evaluation dataset
│   └── pilot/                  # Pilot training data
│
├── static/                     # Web UI assets
│   └── index.html
│
└── archive/                    # Obsolete files (gitignored)
    ├── LORA_ADAPTER_FIX_PLAN.md
    └── RETRAINING_PLAN.md
```

## 🔑 Key File Mapping

| File | Phase | Description |
| :--- | :--- | :--- |
| `scripts/train/lora_v3.py` | **Phase 3** | Manual token expansion tuning. Loss → 0.0006. |
| `scripts/train/patched_lora.py` | Phase 2 | Had truncation bug causing "Blind Model". |
| `src/rag_engine.py` | Phase 5 | SigLIP + ChromaDB retrieval. |
| `src/server.py` | Phase 4 | FastAPI Web UI backend. |
| `docs/reports/PHASE_2_TUNING_FAILURE_ANALYSIS.md` | Phase 2 | Consolidated failure analysis. |
| `docs/reports/PHASE_2_VS_RAG_EVALUATION.md` | Phase 2 | Vanilla vs RAG comparison. |

## 🚀 Usage

```bash
# Setup
python scripts/setup/setup_pokemon_data.py

# Run RAG Demo
python src/demo_rag.py

# Start Web UI
uvicorn src.server:app --reload

# Train (V3)
python scripts/train/lora_v3.py --dataset data_pokemon --steps 600

# Evaluate
python scripts/eval/evaluate_models_v2.py
```

## 🚫 Excluded from Git

- `archive/` - Old plans
- `data*/` - Datasets
- `adapters*/` - Model weights
- `*.safetensors`, `*.log`, `*.csv`
