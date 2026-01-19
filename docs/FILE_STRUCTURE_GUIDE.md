# File Structure Guide & Task Mapping

This document provides a map of the key files in this repository, their purpose, and the corresponding Experimental Phase/Task relative to the `README.md`.

## 📂 Documentation (`docs/`)

### Reports (Analysis)
| File | Phase | Description |
| :--- | :--- | :--- |
| `docs/reports/PHASE_2_TUNING_FAILURE_ANALYSIS.md` | **Phase 2 (Failed)** | Consolidated analysis of why the 2nd tuning attempt failed (includes Root Cause: "Blind Model" & Truncation). |
| `docs/reports/PHASE_2_VS_RAG_EVALUATION.md` | **Phase 2 (Eval)** | Detailed comparison of Vanilla vs RAG vs Tuned (Phase 2) models. |

### Logs (Raw Data)
| File | Phase | Description |
| :--- | :--- | :--- |
| `docs/logs/phase_2_training_log.csv` | **Phase 2** | Training metrics for the failed tuning run (showing high loss at ~7.8). |
| `docs/logs/phase_3_training_log.txt` | **Phase 3** | Raw log of the successful V3 tuning (showing loss dropping to 0.0006). |

### Legacy (Archive)
- `docs/reports/legacy/`: Contains old, fragmented reports (`TUNING_REPORT.md`, `INDEX_ERROR_ANALYSIS.md`) that have been consolidated into the files above.

## 🛠️ Key Scripts (Root)

### Tuning & Training
| File | Phase | Description |
| :--- | :--- | :--- |
| `lora_v3.py` | **Phase 3 (Success)** | **Current Best Script**. Implements Manual Token Expansion fix. |
| `patched_lora.py` | **Phase 2 (Fail)** | Old script with the "Truncation Patch" that caused model blindness. Kept for reference. |

### Inference & RAG
| File | Status | Description |
| :--- | :--- | :--- |
| `rag_engine.py` | ✅ Active | SigLIP-based image retrieval engine. |
| `demo_rag.py` | ✅ Active | CLI demo for RAG. |
| `evaluate_models_v2.py` | ✅ Active | Evaluation script used to generate Phase 2 reports. |
| `inference_v3_custom.py` | ⚠️ Alpha | Custom inference loop attempt (Failed due to M-RoPE complexity). |

### Data Setup
| File | Status | Description |
| :--- | :--- | :--- |
| `setup_pokemon_data.py` | ✅ Phase 1 | Downloads data and adds Korean/Gen metadata. |
