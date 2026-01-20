# MLX-VLM-RAG-Pokemon (한국어 가이드)

[English README](README.md)

Apple MLX 프레임워크를 활용한 포켓몬 식별 **VLM 미세조정 + RAG** 프로젝트입니다.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Non--Commercial-red)

## 📌 프로젝트 개요
800종 이상의 포켓몬을 한국어 이름과 함께 식별하는 AI 시스템:
1. **RAG (검색)**: ChromaDB + SigLIP을 통한 시각적 유사도 검색
2. **Fine-tuning (LoRA)**: Qwen2-VL에 커스텀 어댑터를 퓨전한 4-bit 모델

## 🛠️ 방법론

### 1. 데이터 처리
- **소스**: [pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) (883장)
- **보강**: 한국어 이름 + 세대 정보 추가 (예: "이상해씨 (Bulbasaur). GEN I.")
- **분할**: Train (Gen 1-2, 520장) / Valid (Gen 3+, 313장)

### 2. RAG 시스템
| 구성요소 | 기술 |
| :--- | :--- |
| **비전 인코더** | SigLIP (So400m) |
| **벡터 DB** | ChromaDB |
| **프로세스** | 쿼리 이미지 → 임베딩 → 유사 이미지 검색 → 힌트 주입 → VLM 생성 |
| **필터링** | 1-2세대만 (튜닝 모델과 공정 비교를 위해) |

### 3. LoRA 튜닝 & 모델 퓨전

#### 왜 4-bit 베이스에 16-bit 어댑터를 사용하는가?
- LoRA는 **정밀한 그래디언트**가 필요함; 4-bit 가중치는 그래디언트 손실 유발
- 해결: **16-bit로 학습** 후 퓨전

#### 퓨전 전략
```
[4-bit 베이스] → 역양자화 → [16-bit] + [16-bit LoRA] → 퓨전 → 재양자화 → [4-bit 퓨전 모델]
```
- **최종 모델**: `models/fused_qwen2_vl_4bit_quantized` (**4.3GB**)

### 4. 다단계 평가
모든 접근법을 철저히 테스트하기 위해 4단계 평가를 진행했습니다:

| 버전 | 테스트 유형 | 목적 |
| :--- | :--- | :--- |
| v3 | 일반 프롬프트 | 힌트 없이 기본 비교 |
| v4 | 힌트 프롬프트 | "Pokemon" 키워드 힌트 포함 |
| v5 | OOD 함정 테스트 | **미학습 포켓몬 종** (3세대+) 테스트 |
| v6 | 일반화 테스트 | **학습된 포켓몬의 다른 이미지** 테스트 |

## 📊 평가 결과

### 빠른 비교
| 지표 | Vanilla | RAG | Tuned |
| :--- | :---: | :---: | :---: |
| **1-2세대 정확도 (v5)** | 23.5% | **70.6%** | 17.6% |
| **OOD (3세대+) 정확도 (v5)** | 16.7% | 16.7%* | 16.7% |
| **일반화 정확도 (v6)** | 86.7% | **100%** | 80.0% |

> *RAG는 공정한 비교를 위해 1-2세대 DB로 제한되어 3세대+ 매칭이 의도적으로 제한됩니다.

### 핵심 발견

1. **RAG가 정확도를 지배**: 일반화 테스트 100%, 학습 데이터 70.6%
2. **튜닝 모델은 과적합**: 같은 포켓몬의 새 이미지에서 Vanilla보다 낮은 성능 (80% vs 86.7%)
3. **튜닝 모델은 OOD에서 환각**: 미학습 포켓몬을 자신있게 틀리게 명명 (예: 내룸벨트 → "고오스")

### 샘플 결과 (v3/v4)

| 이미지 | 정답 | Vanilla | RAG | Tuned |
| :---: | :---: | :--- | :--- | :--- |
| <img src="docs/reports/assets/images/pokemon_117.jpg" width="80"><br>**블래키** | Umbreon<br>(블래키) | ✅ 정답 | ✅ 정답 + 한국어 | ✅ 정답 |
| <img src="docs/reports/assets/images/pokemon_025.jpg" width="80"><br>**별가사리** | Staryu<br>(별가사리) | ⚠️ "Staraptor" | ✅ 정답 + 한국어 | ❌ "별 모양 물체" |

### 결론
| 방식 | 최적 용도 |
| :--- | :--- |
| **RAG** | ✅ **프로덕션** (800+ 엔티티, 최고 정확도, 최고 일반화) |
| **Fused** | 응답 스타일/한국어 출력 제어 |
| **Vanilla** | 빠른 프로토타이핑 |

> **권장**: **RAG**를 정확도를 위해 사용하세요. 미세조정은 **스타일/형식 제어**에만 유용하며, 지식 주입에는 적합하지 않습니다.

## 📚 레슨런 (Lessons Learned)

### 1. MLX LoRA 어댑터 인퍼런스 이슈 (M-RoPE)
- **문제**: LoRA 학습 성공 후에도 인퍼런스 시 출력이 생성되지 않음
- **원인**: `mlx_vlm`의 `generate()` 함수가 M-RoPE 상태 관리 실패
- **해결**: **모델 퓨전** - LoRA 가중치를 베이스 모델에 영구 병합

### 2. EOS 토큰 학습 실패 (과소적합)
- **문제**: 초기 튜닝 시 반복적인 쓰레기 출력 (`!!!!`)
- **원인**: 학습이 너무 일찍 중단됨 (20-30 steps)
- **해결**: Loss가 1.0 이하로 수렴할 때까지 **600+ steps** 학습

### 3. 왜 16-bit로 학습해야 하는가? (4-bit의 한계)
- **학습 시**: 4-bit는 표현 단계가 너무 거칠어, 학습 중 발생하는 **미세한 가중치 조정값(Gradient)이 0으로 사라집니다.** 따라서 학습은 반드시 **16-bit**로 해야 합니다.
- **퓨전 시**: 16-bit로 학습된 LoRA를 4-bit 모델에 바로 더할 수 없습니다. 베이스 모델을 **16-bit로 복원(Dequantize)**하여 합친 후, 다시 4-bit로 압축해야 합니다.

### 4. RAG vs Fine-tuning 트레이드오프
- **발견**: 800+ 엔티티 식별에서 RAG가 정확도와 비용 모두에서 미세조정을 능가
- **교훈**: **대규모 엔티티 인식**에는 RAG를 우선시

### 5. RAG 메타데이터 서브스트링 버그
- **문제**: "signature"가 "natu"를 매칭 → 팽도리가 네이티로 표시
- **해결**: **정규식 단어 경계** (`\b{name}\b`) 사용

### 6. 미세조정과 시각적 일반화의 트레이드오프 (Regression)
- **증거**: v6 평가에서 Vanilla 모델은 망나뇽(Dragonite)을 정답으로 맞췄으나, Tuned 모델은 이를 리자몽(Charizard)으로 오판함.
- **인사이트**: 500장 규모의 소규모 데이터셋 학습은 모델이 학습 데이터의 분포에 과적합되게 하여, 기존에 가지고 있던 범용적인 시각적 식별 능력(Pre-trained Knowledge)의 **퇴행(Regression)이 의심됨**.
- **교훈**: 일반화 성능이 중요한 경우, 미세조정 모델 단독 사용보다는 **RAG**나 **앙상블** 접근이 안전함.

## 🚀 빠른 시작

```bash
# 1. 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 데이터 준비
python scripts/setup/setup_pokemon_data.py

# 3. 웹 UI 실행
uvicorn src.server:app --reload --port 8000
# http://localhost:8000/static/index.html 접속
```

## 📁 주요 파일
| 파일 | 설명 |
| :--- | :--- |
| `src/rag_engine.py` | SigLIP + ChromaDB 검색 엔진 |
| `src/server.py` | FastAPI 인퍼런스 서버 |
| `scripts/train/lora_v3.py` | LoRA 학습 스크립트 |
| `scripts/train/fuse_vlm.py` | 모델 퓨전 스크립트 |
| `models/fused_qwen2_vl_4bit_quantized/` | 최종 퓨전 모델 (4.3GB) |

## 📄 상세 리포트
| 리포트 | 설명 |
| :--- | :--- |
| [v3 - 일반 프롬프트](docs/reports/EVALUATION_REPORT_v3.md) | "Pokemon" 힌트 없이 |
| [v4 - 힌트 프롬프트](docs/reports/EVALUATION_REPORT_v4.md) | "What Pokemon is this?" 포함 |
| [v5 - OOD 함정 테스트](docs/reports/EVALUATION_REPORT_v5_OOD_TRAP_TEST.md) | 3세대+ 미학습 종 테스트 |
| [v6 - 일반화](docs/reports/EVALUATION_REPORT_v6_GENERALIZATION.md) | 학습된 포켓몬의 다른 이미지 |

## ⚠️ 면책 조항
- **비공식 프로젝트**: Nintendo, Game Freak, The Pokémon Company와 무관합니다.
- **데이터셋**: [diffusers/pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) 라이선스에 따라 비상업적 용도로만 사용 가능.

## 🤝 감사의 말
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
