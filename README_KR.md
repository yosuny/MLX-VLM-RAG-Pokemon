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
- **분할**: Train (Gen 1-2, 600+) / Valid (Gen 3+, 200+)

### 2. RAG 시스템
| 구성요소 | 기술 |
| :--- | :--- |
| **비전 인코더** | SigLIP (So400m) |
| **벡터 DB** | ChromaDB |
| **프로세스** | 쿼리 이미지 → 임베딩 → 유사 이미지 검색 → 힌트 주입 → VLM 생성 |

### 3. LoRA 튜닝 & 모델 퓨전

#### 왜 4-bit 베이스에 16-bit 어댑터를 사용하는가?
- LoRA는 **정밀한 그래디언트**가 필요함; 4-bit 가중치는 그래디언트 손실 유발
- 해결: **16-bit로 학습** 후 퓨전

#### 퓨전 전략
```
[4-bit 베이스] → 역양자화 → [16-bit] + [16-bit LoRA] → 퓨전 → 재양자화 → [4-bit 퓨전 모델]
```
- **최종 모델**: `models/fused_qwen2_vl_4bit_quantized` (**4.3GB**)

## 📊 최종 평가 결과

| 이미지 | 정답 | Vanilla 모델 | RAG 시스템 | Tuned 모델 |
| :---: | :---: | :--- | :--- | :--- |
| <img src="data/pokemon/images/pokemon_117.jpg" width="100"><br>**블래키**<br>(2세대) | Umbreon<br>(블래키) | ✅ **정답**<br>"This is Umbreon." | ✅ **정답**<br>"This is Umbreon (블래키)." | ✅ **정답**<br>"This is Umbreon." |
| <img src="data/pokemon/images/pokemon_025.jpg" width="100"><br>**별가사리**<br>(1세대) | Staryu<br>(별가사리) | ⚠️ **환각 (Hallucination)**<br>"Staraptor (찌르호크)" | ✅ **정답**<br>"This is Staryu..."<br>*(메타데이터 갱신 완료)* | ❌ **단순 묘사**<br>"Star-shaped object" |
| <img src="data/pokemon/images/pokemon_440.jpg" width="100"><br>**리오르**<br>(4세대) | Riolu<br>(리오르) | ❌ **환각**<br>"Umbreon" | ⚠️ **1-2세대 유사 매칭**<br>"Umbreon/Glaceon"<br>*(1-2세대 DB 제한)* | ❌ **환각**<br>"Glaceon" |
| <img src="data/pokemon/images/pokemon_411.jpg" width="100"><br>**트리토돈**<br>(4세대) | Gastrodon<br>(트리토돈) | ❌ **환각**<br>"Gallade" | ⚠️ **1-2세대 유사 매칭**<br>"Mawile/Slug-like"<br>*(1-2세대 DB 제한)* | ❌ **환각**<br>"Gallade" |

> **RAG 공정성 참고**: 튜닝된 모델(1-2세대만 학습)과의 공정한 비교를 위해, **RAG 검색 범위를 1-2세대 포켓몬으로 제한**했습니다.
> 따라서 Riolu와 같은 3세대 이후 포켓몬 검색 시, 정확한 정답 대신 시각적으로 가장 유사한 1-2세대 포켓몬을 찾아내는 것이 시스템의 정상적인 동작입니다.

### 결론
| 방식 | 최적 용도 |
| :--- | :--- |
| **RAG** | ✅ **프로덕션** (800+ 엔티티, 최고 정확도) |
| **Fused** | 응답 스타일/한국어 출력 제어 |
| **Vanilla** | 빠른 프로토타이핑 |

> **권장**: **RAG** (정확도) + **Fused Model** (스타일)을 조합하면 최적의 결과를 얻을 수 있습니다.

## 📚 레슨런 (Lessons Learned)

### 1. MLX LoRA 어댑터 인퍼런스 이슈 (M-RoPE)
- **문제**: LoRA 학습 성공 (Loss 0.0006) 후에도 인퍼런스 시 출력이 생성되지 않음
- **원인**: `mlx_vlm`의 `generate()` 함수가 동적 토큰 확장 시 **M-RoPE (멀티모달 위치 인코딩)** 상태 관리를 제대로 처리하지 못함
- **해결**: **모델 퓨전** - LoRA 가중치를 베이스 모델에 영구 병합하여 런타임 어댑터 로딩 제거
- **교훈**: 라이브러리 한계에 직면하면 **가중치 퓨전**을 런타임 어댑터 주입의 대안으로 고려

### 2. EOS 토큰 학습 실패 (과소적합/Underfitting)
- **문제**: Phase 1-2 튜닝 시 반복적인 쓰레기 출력 (`!!!!`) 발생
- **원인**: 학습이 너무 일찍 중단됨 (20-30 steps), Loss가 여전히 높은 상태 (~8.0). 모델이 문장 종료 토큰 (`<|im_end|>`)을 학습하지 못함
- **해결**: Loss가 1.0 이하로 수렴할 때까지 **600+ steps** 학습
- **교훈**: VLM 미세조정에서 **충분한 학습 스텝**이 필수. EOS 학습 전 조기 중단은 무한 생성 루프 유발

### 3. 4-bit 양자화 & 그래디언트 정밀도
- **문제**: 4-bit로 LoRA 학습 시 불안정하고 수렴 실패 (Loss ~8.0에서 고착)
- **인사이트**: 4-bit 양자화된 가중치는 역전파 시 그래디언트 정밀도 손실
- **교훈**: 어댑터는 항상 **16-bit (Float16)**로 학습 후 퓨전하여 배포

### 4. RAG vs Fine-tuning 트레이드오프
- **발견**: 800+ 엔티티 식별에서 RAG가 정확도와 비용 모두에서 미세조정보다 우수
- **이유**: 미세조정은 엔티티당 대량 데이터 필요; RAG는 엔티티당 1장 이미지만 인덱싱
- **교훈**: **대규모 엔티티 인식**에는 RAG 우선. 미세조정은 **스타일/형식 제어**에만 사용

### 5. 프롬프트 엔지니어링의 영향
- **관찰**: "Pokemon" 힌트를 프롬프트에 추가하면 오히려 환각 유발 (Staryu → Staraptor)
- **인사이트**: 도메인 키워드가 모델을 언어적으로 유사하지만 시각적으로 틀린 답변으로 편향시킴
- **교훈**: 일반/힌트 프롬프트 모두 테스트; 때로는 **적은 컨텍스트가 더 나음**

### 6. RAG 메타데이터 내 부분 문자열 매칭 버그 (Substring Match)
- **문제**: 펭도리(Piplup) 이미지가 RAG UI에서 **네이티(Natu)**로 식별됨
- **원인**: 자동 라벨링 스크립트가 `if name in caption.lower()`를 사용함. 펭도리 설명에 포함된 **"signature"**라는 단어 안에 **"natu"**가 포함되어 있어 네이티로 오판독됨
- **해결**: **정규표현식 단어 경계**(`\b{name}\b`)를 사용하여 정확히 대응하는 이름만 매칭되도록 수정
- **교훈**: 키워드 기반 자동 라벨링 시, **단어 단위 매칭**을 수행하지 않으면 부분 문자열에 의한 대규모 오답 데이터가 생성될 수 있음

### 7. 코드 레벨 디버깅 인사이트

| 이슈 | 증상 | 원인 | 해결 |
| :--- | :--- | :--- | :--- |
| **RAG 힌트 누락** | RAG가 Vanilla와 동일 결과 | `documents` 대신 `metadatas['caption']` 읽어야 함 | 올바른 필드 접근 |
| **PyTorch 텐서 오류** | `ValueError` | HF Fast Processor가 MLX 비호환 | `use_fast=False` + numpy 래퍼 |
| **이미지 경로 불일치** | `FileNotFoundError` | `data_pokemon/` vs `data/pokemon/` | 문자열 치환 |
| **양자화 인자 순서** | 잘못된 양자화 | `nn.quantize(model, group_size, bits)` 순서 | MLX API 문서 확인 |
| **Config 누락** | Float16으로 로드됨 | `quantization` 블록 미포함 | config.json 패치 |

## 🚀 빠른 시작

```bash
# 1. 설치
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

## 📄 리포트
- [EVALUATION_REPORT_v3.md](docs/reports/EVALUATION_REPORT_v3.md) - 일반 프롬프트 평가
- [EVALUATION_REPORT_v4.md](docs/reports/EVALUATION_REPORT_v4.md) - 힌트 프롬프트 평가

## ⚠️ 라이선스 및 고지사항
- **비공식 프로젝트**: 닌텐도, 게임프리크, 포켓몬 컴퍼니와 제휴되지 않음
- **데이터셋**: [diffusers/pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) 라이선스에 따라 비상업적 용도로만 사용

## 🤝 감사의 글
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
