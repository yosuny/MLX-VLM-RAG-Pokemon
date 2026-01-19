# Tuned Model Failure Analysis

## Issue
The LoRA fine-tuned model fails during inference with `IndexError: list index out of range` in the detokenizer.

## Symptoms
- Inference crashes or returns empty strings.
- Debugging reveals the model generates token IDs that exceed the tokenizer's vocabulary size.
- Some generated text fragments (viewed in debug logs) are incoherent (e.g., random foreign characters).

## Root Cause Analysis
This behavior typically indicates **Model Collapse** or **Gradient Explosion** during training, often caused by:
1. **Learning Rate too high**: The default `1e-4` might be too aggressive for Qwen2-VL with LoRA, causing weights to drift into unstable regions (NaNs or extreme values).
2. **Overfitting**: 200 steps on a small dataset (~10-20 images) might be excessive, leading the model to memorize noise.
3. **Data Quality**: If any image/caption pair was corrupt, it could destabilize the gradients.

## Conclusion
The current `adapters.safetensors` file is unusable. Retraining is required with more conservative hyperparameters.
