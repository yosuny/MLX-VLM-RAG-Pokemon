# Retraining Plan: Qwen2-VL Pokemon Tuner

## Objective
Re-train the LoRA adapter to fix the "IndexError" (Vocab collapse) issue by stabilizing the training process.

## Revised Hyperparameters

| Parameter | Old Value (Failed) | New Value (Proposed) | Rationale |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | `1e-4` | **`1e-5`** | Reduce by 10x to prevent gradient explosion and unstable weight updates. |
| **Max Steps** | `200` | **`100`** | Reduce duration to prevent overfitting on the small dataset. |
| **Batch Size** | `1` | `1` | Keep same (limited by VRAM). |
| **LoRA Rank** | `16` | `16` | Keep same. |
| **LoRA Alpha** | `16` | `16` | Keep same. |

## Execution Steps

### 1. Clean Up
- Remove old `adapters/` directory to prevent mixing weights.
- `rm -rf adapters`

### 2. Run Training
Execute `patched_lora.py` with new flags:

```bash
python patched_lora.py \
    --model-path mlx-community/Qwen2-VL-7B-Instruct-4bit \
    --dataset data_pokemon \
    --learning-rate 1e-5 \
    --steps 100 \
    --output-path adapters_retrain \
    --apply-chat-template
```

### 3. Validation
- Run `evaluate_tuned_only.py` immediately after training.
- Check for `IndexError` or gibberish output.
- **[NEW] Save Training Logs**: Save step/loss history to `training_log_v2.csv` for analysis.

## Timeline
- **Training**: ~20-30 minutes (100 steps)
- **Evaluation**: ~5 minutes
