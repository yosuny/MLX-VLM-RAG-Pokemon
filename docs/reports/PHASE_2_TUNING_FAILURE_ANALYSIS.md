# Phase 2 Tuning Failure & Root Cause Analysis

## 1. Executive Summary
This document consolidates the analysis of the **failed** 2nd tuning attempt (Phase 2).
Despite resolving the initial `IndexError` crash by lowering the learning rate (Phase 1 fix), the fine-tuned model exhibited **quality degradation** (repetitive text like "helpful helpful").
The ultimate root cause was identified as **"Model Blindness"** due to incorrect token expansion, which was later fixed in Phase 3.

## 2. Technical Symptoms
### Phase 1: The Crash (`IndexError`)
- **Symptom**: Inference crashed with `list index out of range`.
- **Cause**: Tokenizer vocab (`151,643`) < Model Output (`152,064`). High Learning Rate (`1e-4`) caused weights to drift into the "unknown" token range, selecting tokens the tokenizer couldn't handle.
- **Temporary Fix**: Reduced Learning Rate to `1e-5`. Stopped the crash, but led to Phase 2 issues.

### Phase 2: The "Broken Record" (Repetition)
- **Symptom**: Model generated repetitive text (e.g., `helpful helpful helpful`) or empty strings.
- **Log Analysis**: Loss converged to `~7.8` (high) instead of `< 2.0`.
- **Diagnosis**: The model failed to learn any meaningful patterns and collapsed into a degenerate state.

## 3. Root Cause Investigation
We analyzed three potential causes:

### Root Cause 1: Quantization Noise (Contributor)
- **Factor**: Tuning **all linear layers** on a **4-bit** quantized model.
- **Analysis**: Backpropagating through 4-bit weights introduces noise. The "All Linear" strategy amplified this, causing the model to memorize noise patterns.

### Root Cause 2: Data Scarcity (Contributor)
- **Factor**: 500 images for a broad parameter update.
- **Analysis**: Insufficient data for the scope of parameters being tuned.

### Root Cause 3: The "Blind Model" (Definitive Cause)
- **Discovery**: `debug_chat_template_v2.py` revealed that the `apply_chat_template` function inserted **only 1 `<|image_pad|>` token** for each image.
- **Reality**: The Qwen2-VL Vision Encoder generates ~600 visual tokens (for 224x224).
- **Impact**: The model received **1 token** of visual info but was expected to process 600.
- **Fatal Patch**: To prevent a dimension mismatch crash, we truncated the visual features to 1. **This effectively blinded the model**, forcing it to guess text from nothing.

## 4. Resolution (Phase 3)
The insights from this analysis led to the **V3 Tuning Strategy**:
1.  **Manual Token Expansion**: Created `lora_v3.py` to calculate the exact grid size and insert correct `<|image_pad|>` tokens.
2.  **Removal of Truncation**: Allowed the full visual features to flow into the LLM.
3.  **Result**: Loss dropped to **0.0006**, achieving perfect alignment.
