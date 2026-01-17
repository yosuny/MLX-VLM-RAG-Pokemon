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

## Recommendations for Future Attempts
1. **Reduce Scope**: Only target attention layers (`q_proj`, `v_proj`) instead of all linear layers. This reduces parameters and noise.
2. **Increase Precision**: Use an **8-bit** or **BF16** base model if hardware permits (requires ~16GB+ VRAM/RAM). 4-bit is too unstable for rigorous fine-tuning on small datasets.
3. **Data Quality**: 500 images might be too few for "All Linear" tuning. Increase dataset size or use data augmentation locally.
