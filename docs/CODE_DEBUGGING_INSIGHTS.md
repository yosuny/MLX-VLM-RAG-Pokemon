# Code-Level Debugging Insights

This document captures detailed debugging insights encountered during the MLX-VLM-RAG-Pokemon project development.

## Common Issues & Fixes

| Issue | Symptom | Root Cause | Fix |
| :--- | :--- | :--- | :--- |
| **RAG returning empty hints** | RAG mode same as Vanilla | Script read `documents` (empty) instead of `metadatas['caption']` | Access `results['metadatas'][0][0]['caption']` |
| **PyTorch tensor error** | `ValueError: Only PyTorch tensors supported` | HuggingFace fast image processor incompatible with MLX | Set `use_fast=False` + wrap with numpy converter |
| **Image path mismatch** | `FileNotFoundError` | JSONL had `data_pokemon/` but actual path was `data/pokemon/` | String replace in data loader |
| **Quantize argument order** | Silent wrong quantization | `nn.quantize(model, bits, group_size)` → actually `(model, group_size, bits)` | Check MLX API docs |
| **Config missing quantization info** | Model loads as Float16 | Fused model config.json lacked `quantization` block | Patch config post-save |

## Detailed Explanations

### 1. RAG Empty Hints Issue
When using ChromaDB with SigLIP embeddings, the `query()` function returns multiple fields. Initially, we incorrectly accessed `results['documents']` which was empty because we stored data in metadata, not documents.

**Incorrect:**
```python
hint = results['documents'][0][0]  # Empty string
```

**Correct:**
```python
hint = results['metadatas'][0][0]['caption']
```

### 2. PyTorch Tensor Error
MLX and PyTorch tensors are not interchangeable. When HuggingFace's image processor returns PyTorch tensors, MLX operations fail.

**Solution:**
```python
class RobustImageProcessorWrapper:
    def __call__(self, images=None, text=None, **kwargs):
        if "return_tensors" in kwargs:
            kwargs["return_tensors"] = "pt"
        out = self.processor(images, text, **kwargs)
        for k, v in out.items():
            if hasattr(v, "numpy"):
                out[k] = v.numpy()  # Convert to numpy for MLX compatibility
        return out
```

### 3. MLX Quantization API
The MLX `nn.quantize()` function has a non-intuitive argument order:

**Incorrect (intuitive but wrong):**
```python
nn.quantize(model, bits=4, group_size=64)
```

**Correct:**
```python
nn.quantize(model, group_size=64, bits=4)
```

Always check the [MLX documentation](https://ml-explore.github.io/mlx/) for correct API usage.

### 4. Config.json Quantization Block
After fusing and quantizing a model, the `config.json` must include quantization metadata for proper loading:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4
  }
}
```

If this is missing, the model will attempt to load as Float16 and fail.

## Related Files
- `src/rag_engine.py` - RAG implementation
- `src/server.py` - FastAPI server with robust image processor
- `scripts/train/fuse_vlm.py` - Model fusion script
- `scripts/train/quantize_vlm.py` - Quantization script
