# Implementation Plan - VLM Tuning & Image RAG

## Goal Description
Implement a system for **Image Tuning** (Fine-tuning) and **Image RAG** (Retrieval-Augmented Generation) using the **Qwen-VL-7B** model.
Given the MacOS environment, we will utilize **MLX** (`mlx-vlm`) for efficient local inference and training of the quantized model.

## User Review Required
> [!IMPORTANT]
> This plan assumes the use of **Apple MLX** framework for hardware acceleration on Mac. If you prefer PyTorch/MPS or pure CPU (slower), please let me know.
> I will use `mlx-community/Qwen-VL-Chat-4bit` (or similar) from Hugging Face as the base model to save memory.

## Proposed Changes

### Setup & Infrastructure
#### [NEW] `requirements.txt`
- `mlx`
- `mlx-vlm`
- `transformers`
- `huggingface_hub`
- `pillow`
- `torch` (for CLIP/embeddings if needed, or use mlx counterpart)
- `chromadb` or `faiss-cpu` (for RAG vector store)

### Image Tuning (Fine-tuning)
#### [NEW] `train_vlm.py`
- Script to fine-tune Qwen-VL using LoRA/QLoRA.
- Will use `mlx.optimizers` and `mlx.nn`.
- Supports loading a JSONL dataset (image_path, conversation).

### Image RAG
#### [NEW] `rag_engine.py`
- **Embedding**: Use a CLIP-like model to embed images.
- **Retrieval**: Store embeddings in a local vector store.
- **Pipeline**:
    1. Query (Image or Text) -> Embedding.
    2. Retrieve top-k relevant images/context.
    3. Feed retrieved images as context to Qwen-VL.

#### [NEW] `demo_rag.py`
- A script to demonstrate the RAG flow: "Here is a query image, retrieve similar ones, and answer a question based on them."

## Verification Plan

### Automated Tests
- Run `train_vlm.py` with a dummy dataset (1-2 entries) to ensure the training loop runs and loss changes.
- Run `rag_engine.py` to index ~5 images and perform 1 retrieval query.

### Manual Verification
- Execute `demo_rag.py` and observe the console output.
- Check if Qwen-VL answers questions using the retrieved context.

## Refined Technical Approach (Based on Research)

### 1. Model Selection & Environment
- **Model**: `mlx-community/Qwen-VL-Chat-4bit` (or conversion from `Qwen/Qwen-VL-Chat`).
- **Library**: `mlx-vlm` is the core. It supports direct generation and has experimental LoRA support.

### 2. Image Tuning Strategy
- **Method**: LoRA (Low-Rank Adaptation) is confirmed as the viable path on MLX.
- **Script**: We will adapt the official `mlx-examples/vlm/lora` approach.
    - **Data Format**: `train.jsonl` with `{"image": "path/to/img", "text": "conversation..."}`.
    - **Optimization**: Use `mlx.optimizers.AdamW`.

### 3. Image RAG Architecture
- **Embedding Model**: **SigLIP** (`google/siglip-so400m-patch14-384`) is recommended over CLIP for better image-text retrieval performance. We can run this via `transformers` (PyTorch CPU/MPS) or port to MLX if needed.
- **Vector Store**: **ChromaDB** for local persistence.
- **Flow**:
    1. **Ingest**: Images -> SigLIP -> Embeddings -> ChromaDB.
    2. **Query**: User Text/Image -> SigLIP -> Vector Search.
    3. **Generate**: Top-k Images + Query -> Qwen-VL Prompt.

## 한국어 계획 설명 (Korean Plan) - Refined

### 목표 및 기술 스택 업데이트
**Qwen-VL-7B**의 MLX 버전(`mlx-vlm`)을 기반으로, **LoRA 튜닝**과 **SigLIP 기반 RAG**를 구현합니다.

### 상세 구현 방안

#### 설정 및 인프라
- **`requirements.txt`**: `mlx-vlm`, `chromadb`, `transformers` (SigLIP용).

#### 이미지 튜닝 (Fine-tuning)
- **`train_vlm.py`**:
    - GitHub 리서치 결과, `mlx-vlm`의 LoRA 예제 코드를 기반으로 커스텀 데이터셋(이미지+텍스트)을 학습하도록 작성합니다.
    - 데이터 형식: `{"image": "./img.jpg", "text": "사용자: 이 사진 설명해.\nAI: 고양이입니다."}` 형태의 JSONL.

#### 이미지 RAG (검색)
- **`rag_engine.py`**:
    - **임베딩 모델**: 기존 CLIP보다 성능이 우수한 **SigLIP** 모델을 사용합니다.
    - **저장소**: **ChromaDB**를 사용하여 로컬에 벡터를 저장하고 검색합니다.
    - **검색 로직**: 텍스트나 이미지를 입력받아 가장 유사한 이미지를 찾고, 이를 Qwen-VL의 프롬프트에 '컨텍스트'로 넣어줍니다.

### 검증 계획
- **튜닝**: 손실(Loss) 값이 떨어지는지 그래프/로그로 확인.
- **RAG**: "비슷한 분위기의 사진 찾아줘" 같은 쿼리에 대해 정확한 이미지를 가져오는지 1차 확인 후, Qwen-VL이 그 이미지를 보고 대답하는지 2차 확인.

