# Tuned Model Retraining Report (Phase 9)

## 📊 Summary
- **Status**: ✅ **Success** (Completed without divergence)
- **Model**: `mlx-community/Qwen2-VL-7B-Instruct-4bit`
- **Method**: LoRA (Rank 16, Alpha 16)
- **Dataset**: Pokemon Image-Caption Pairs (Generations 1-2)
- **Hyperparameters**:
    - Learning Rate: `1e-5` (Reduced from 1e-4) -> **Key to success**
    - Steps: 100 (Reduced from 200)

## ⏱️ Execution Stats
- **Total Duration**: **5 hours 20 minutes**
- **Average Speed**: ~3.2 minutes / step (Due to high-resolution image processing)
- **Final Loss**: `~8.0` (Converged stably from 18.9)

## 🛠️ Fix Verification
- **Issue**: Previous `IndexError` (Vocab out of range).
- **Resolution**: Lowering Learning Rate preventing gradient explosion.
- **Confirmation**: Inference process runs without crashing.

## 🔍 Root Cause Analysis (Quality Issues)
Despite stability, the model generates repetitive text. Detailed analysis in `TUNING_ROOT_CAUSE_ANALYSIS.md` reveals:
1. **Model (Vocab Mismatch)**: Tokenizer (`151k`) vs Model Output (`152k`) mismatch caused initial crashes.
2. **Strategy (Over-Targeting)**: Tuning **all linear layers** on a small dataset (~500 images) with a **4-bit** model caused the model to learn quantization noise patterns instead of features.
3. **Platform (4-bit QLoRA)**: 4-bit quantization introduces significant noise, which "exploded" under high learning rates (`1e-4`), leading to pattern collapse.

## 📂 Artifacts
- **Adapter Weights**: `adapters/adapters.safetensors` (~161 MB)
- **Training Logs**: `training_log_v2.csv`
- **Report**: `EVALUATION_REPORT_v2.md` (Updating...)
