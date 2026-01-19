# Tuning Failure Root Cause Analysis

## Executive Summary
Despite resolving the critical `IndexError` crash by lowering the learning rate, the fine-tuned model exhibits low-quality outputs (repetitive text patterns). This analysis explores the failure from Model, Quantization, and ML Platform perspectives.

## 1. Model Perspective (Qwen2-VL)
### 🚫 Vocab & Architecture Mismatch
- **Observation**: We identified a mismatch between the Tokenizer's vocab size (`151,643`) and the Model's output layer size (`152,064`).
- **Impact**: The base model uses these extra ~400 tokens internally (likely for vision encoding), but the tokenizer is unaware of them. When LoRA training destabilizes (high LR), the model assigns high probability to these "unknown" tokens, causing the `IndexError` crash.
- **LoRA Scope**: The script targeted **ALL** linear layers (`find_all_linear_names`). For a complex multi-modal model like Qwen2-VL, blindly tuning all layers (including MLP and Output heads) on a small dataset (~500 images) rapidly leads to **Overfitting** and **Pattern Collapse** (e.g., repeating "helpful").

## 2. Quantization Perspective (4-bit)
### 📉 Signal-to-Noise Ratio
- **Context**: We used `Qwen2-VL-7B-Instruct-4bit`.
- **Issue**: 4-bit quantization introduces quantization error. When backpropagating gradients through these low-precision weights to update the high-precision LoRA adapters, the **gradient noise** is significantly higher than in 16-bit models.
- **Result**: "All Linear Layers" strategy amplified this noise. The model likely learned to memorize the quantization noise pattern rather than the actual Pokemon features, leading to the "helpful helpful" nonsensical generation.

## 3. ML Platform Perspective (MLX)
### 🧪 Experimental Stability
- **Qwen2-VL Support**: Support for this specific model in `mlx-vlm` is relatively new and required manual patching (`pad_size` fix, `RobustImageProcessorWrapper`).
- **Optimization**: The default `Adam` optimizer settings in standard MLX examples might not be tuned for **4-bit QLoRA** stability (e.g., epsilon values, weight decay).
- **Inference Pipeline**: The need to "force" slow image processors and monkey-patch tensor handling indicates the platform pipeline for this specific model is not yet mature enough for seamless "out-of-the-box" fine-tuning.

### 4. Definitive Root Cause: Model Blindness (Token Expansion Failure)
**Verification**:
- `debug_chat_template_v2.py` confirmed that `apply_chat_template` inserts **only 1 `<|image_pad|>` token**.
- **V3 Pilot Run**: Implemented manually token expansion in `lora_v3.py`.
- **Result**: Loss dropped from **3.83** to **0.09** in just 50 steps. (Previous failure stayed at 8.0).
- **Conclusion**: The model is no longer blind and is learning effectively.
- **Problem**: Qwen2-VL requires dynamic token expansion (e.g., 256 tokens for a 224x224 image).
- **Consequence**: The model received 1 placeholder token for an image that generated ~600 visual features.
- **The Fatal Patch**: To prevent the resulting dimension mismatch crash (`Features > Embeds`), our `patched_lora.py` implemented a truncation logic:
    ```python
    # Logic in patched_lora.py
    if pad_size < 0: # Features (600) > Placeholders (1)
        image_features = image_features[:, :inputs_embeds.shape[1], :] # Truncate to 1
    ```
- **Result**: **We fed the model only 1 pixel worth of data (0.2% of the image).** The model was effectively blind, leading to the high loss (7.86) and repetitive text.

### 5. Ruled Out Factors (Verified)
- **Data Format / Image Token Placement**:
    - `apply_chat_template` correctly identifies the string content and adds the *start/end* tokens. The issue is strictly the *quantity* of the inner pad tokens.

## 6. Recommendations for Future
1.  **Correct Tokenization**: The training script must manually calculate the image grid size and insert `N` image tokens (`<|image_pad|>`) into the prompt string *before* tokenization, matching the Vision Encoder's output.
2.  **Remove Truncation Patch**: Once tokens are expanded correctly, the truncation patch should be removed.
3. **Reduce Scope**: Only target attention layers (`q_proj`, `v_proj`) instead of all linear layers. This reduces parameters and noise.
4. **Use Higher Precision**: Move to **8-bit** or **BF16** base model to reduce quantization noise. This requires ~16GB+ VRAM/RAM). 4-bit is too unstable for rigorous fine-tuning on small datasets.
5. **Data Quality**: 500 images might be too few for "All Linear" tuning. Increase dataset size or use data augmentation locally.
