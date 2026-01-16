# MLX-VLM-RAG-Pokemon (한국어 가이드)

[English README](README.md)

Apple MLX 프레임워크를 사용하여 macOS에서 포켓몬 식별을 위한 **VLM (Vision-Language Model)** 미세조정 및 **RAG (검색 증강 생성)**를 수행한 프로젝트입니다.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 실험 결과 및 기술적 회고 (Technical Review)

이미지 인식 튜닝 과정과 결과, 기술적 분석 내용 

### 1. 학습 방식: SFT (Supervised Fine-Tuning)
우리는 **Qwen2-VL** 모델을 사용하여 **LoRA (Low-Rank Adaptation)** 방식으로 SFT를 진행했습니다.

*   **구조 (Architecture)**:
    *   **이미지 벡터화 (Multimodal Embedding)**: Vision Encoder가 이미지를 시각적 토큰(Visual Tokens)으로 변환합니다. 변환된 벡터는 텍스트 임베딩과 결합되어 LLM에 입력됩니다. (Cross-Modal Instruction Tuning)
    *   **접근법**: 이미지 벡터를 학습하는 것이 아니라, **이미지 벡터와 텍스트(정답) 사이의 연결 고리(Alignment)**를 LLM 파트에서 학습했습니다.

### 2. 실험 과정 (Experiments)

| 시도 | 설정 (Configuration) | 결과 | 분석 |
| :--- | :--- | :--- | :--- |
| **1차** | 30 Steps, `apply_chat_template` 미적용 | **실패** (`!!!!` 반복) | 텍스트와 이미지를 구분하는 특수 토큰 누락으로 모델이 환각 증세를 보임. |
| **2차** | 20 Steps, `apply_chat_template` 적용 | **실패** (`!!!!` 반복) | 포맷팅은 수정했으나 학습량(Step)이 절대적으로 부족(Underfitting)하여 EOS 토큰을 학습하지 못함. |

### 2-1. RAG의 강력함: 유사 이미지 검색 (Semantic Search)
우리는 RAG 시스템이 "똑같지 않은" 유사 이미지도 잘 찾는지 검증했습니다.
*   **실험**: 학습 데이터에 없는 "이상해씨 인형(Plush Toy)" 사진으로 검색
*   **결과**:
    *   1순위: **이상해씨 (공식 일러스트)** (유사도 0.70) ✅
    *   2순위: 캐터피 (다른 포켓몬) (유사도 0.97 - 멀어짐)
*   **의의**: SigLIP 임베딩은 픽셀 일치 여부가 아니라, **"초록색", "등에 씨앗이 있음" 같은 시각적 의미(Semantics)**를 인식하므로 스타일이 달라도 정확히 검색합니다.

### 3. 실패 원인 심층 분석 (Root Cause Analysis)

**Q. 학습 방법에 문제(벡터 결합 등)가 있었는가?**
> **아니요, 아키텍처는 올바랐습니다.**
> 우리는 현대적인 VLM의 표준 방식(이미지 패치 벡터화 + 텍스트 인스트럭션 결합)을 따랐습니다. 문제는 **데이터 포맷팅(Template)**과 **학습량(Volume)**에 있었습니다.

**Q. EOS Token (문장 종료) 학습 실패 원인은?**
> **조기 종료(Early Stopping)에 의한 학습 미달(Underfitting)입니다.**
> *   Loss(손실 함수)가 `24.0` -> `11.0`으로 계속 떨어지는 도중에 학습을 중단했습니다.
> *   모델이 정답("피카츄")을 생성하기도 전에 학습이 끊겨버려, 문장을 끝맺는 법(`<|im_end|>`)을 배울 기회를 갖지 못했습니다. 최소 600 Step 이상의 충분한 학습이 필요합니다.

### 4. 결론 (Conclusion)
*   **RAG (검색) 승리**: 특정 데이터(한국어/영어 이름 매칭)를 찾는 데에는 학습보다 검색(RAG)이 월등히 빠르고 정확했습니다.
*   **생성 모델의 한계**: 적은 데이터와 짧은 학습으로는 기존 모델이 가진 강력한 표현력을 재조정하기 어렵습니다.

### 5. 향후 개선 및 해결 방안 (Future Work)

실패 원인을 바탕으로 성능을 개선하기 위한 다음 단계 제안입니다:

1.  **학습량 증대 (Increase Training Steps)**:
    *   현재 20~30 Step은 턱없이 부족합니다. 최소 **600~1,000 Step** 이상 학습하여 Loss가 충분히 수렴(Converge)하도록 기다려야 합니다.
2.  **데이터 품질 강화 (Data Quality)**:
    *   EOS 토큰(`<|im_end|>`) 학습 강화를 위해 답변 문장의 길이를 다양화하고, 명시적인 종료 패턴을 학습 데이터에 추가합니다.
3.  **LoRA Rank 조정**:
    *   현재의 미세조정 강도가 부족할 수 있으므로, LoRA의 Rank(r) 값을 16 또는 32로 높여 더 많은 파라미터를 학습에 참여시킵니다.
4.  **하이브리드 접근 (Hybrid RAG-Tuning)**:
    *   미세조정 모델이 RAG에서 검색된 정보를 문장으로 자연스럽게 다듬는 역할만 수행하도록 역할을 분담시킵니다.

---

## 🚀 주요 기능
*   **Data Pipeline**: 포켓몬 데이터 다운로드 및 한국어 이름 매핑 추가.
*   **VLM RAG**: SigLIP 임베딩을 활용한 이미지-이미지 검색.
*   **LoRA Fine-tuning**: MLX용 Qwen2-VL 학습 패치 스크립트.
*   **Evaluation**: 바닐라 모델 vs RAG vs 튜닝 모델 성능 자동 비교.

## 🛠️ 사용 방법

### 1. 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 데이터 다운로드 및 한국어 매핑 생성
python setup_pokemon_data.py
```

### 3. RAG 데모 실행
```bash
python demo_rag.py
```

### 4. 미세조정 (Fine-Tune) 실행
```bash
# 주의: 충분한 학습을 위해 steps를 높게 설정하는 것을 권장합니다 (600+)
python patched_lora.py --dataset data_pokemon --steps 600 --output-path adapters --apply-chat-template
```

### 5. 평가 (Evaluation)
```bash
python evaluate_models.py
```

## ⚠️ 라이선스 및 고지사항
이 프로젝트는 교육 목적으로 제작되었으며 닌텐도나 포켓몬 컴퍼니와 무관합니다. 데이터셋의 라이선스 규정을 준수해 주십시오.
