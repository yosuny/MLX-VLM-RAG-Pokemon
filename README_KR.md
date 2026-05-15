# MLX-VLM-RAG-Pokemon (한국어 가이드)

[English README](README.md)

Apple MLX 프레임워크를 활용한 포켓몬 식별 **VLM 미세조정 + RAG** 프로젝트입니다.

![MLX](https://img.shields.io/badge/MLX-Compatible-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-Non--Commercial-red)
![Phase](https://img.shields.io/badge/Phase-2%20완료-brightgreen)

## 📌 프로젝트 개요
800종 이상의 포켓몬을 한국어 이름과 함께 식별하는 AI 시스템:
1. **RAG (검색)**: ChromaDB + SigLIP을 통한 시각적 유사도 검색
2. **Fine-tuning (LoRA)**: Qwen2-VL에 커스텀 어댑터를 퓨전한 4-bit 모델

> **Phase 2 안내**: Phase 1 평가에는 구조적 결함이 있었습니다 (데이터 누수, 느슨한 정확도 기준, 소규모 n). Phase 2는 이를 교정하여 신뢰할 수 있는 벤치마크를 제시합니다. 자세한 내용은 [비판적 분석 보고서](docs/reports/CRITICAL_ANALYSIS.md)를 참고하세요.

## 🛠️ 방법론

### 1. 데이터 처리
- **소스**: [pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) (883장)
- **보강**: 한국어 이름 + 세대 정보 추가 (예: "이상해씨 (Bulbasaur). GEN I.")
- **분할**: Train (Gen 1-2, 520장) / Valid (Gen 3+, 313장)
- **주의**: 학습 데이터 520개 중 323개(62.1%)의 캡션에 **포켓몬 이름이 없습니다** (GPT-4가 시각적 묘사만 생성한 경우). 이 항목들은 실제 세대 확인 없이 train에 배정되었습니다.

### 2. RAG 시스템
| 구성요소 | 기술 |
| :--- | :--- |
| **비전 인코더** | SigLIP (So400m) |
| **벡터 DB** | ChromaDB |
| **프로세스** | 쿼리 이미지 → 임베딩 → 유사 이미지 검색 → 힌트 주입 → VLM 생성 |
| **공정 DB** | `chroma_db_fair/` — Phase 2에서 테스트 이미지 제외한 DB |

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

| 페이즈 | 버전 | 테스트 유형 | n | 비고 |
| :--- | :--- | :--- | :---: | :--- |
| **Phase 1** | v3 | 일반 프롬프트 | 4 | 정성 평가만 |
| **Phase 1** | v4 | 힌트 프롬프트 | 4 | 정성 평가만 |
| **Phase 1** | v5 | OOD 함정 테스트 | 30 | RAG 데이터 누수 있음 |
| **Phase 1** | v6 | 일반화 테스트 | 15 | 서브스트링 매칭 |
| **Phase 2** | v7 | 공정한 벤치마크 | **50** | 누수 없음, word-boundary |
| **Phase 2** | v8 | Tuned 모델 진단 | **41** | 학습/스프라이트 이미지 비교 |
| **Phase 2** | v9 | OOD 강화 테스트 | **30** | 3카테고리 분석 |

## 📊 평가 결과

### Phase 2 — 공정한 벤치마크 (v7, n=50)

> **평가 조건**: PokeAPI 공식 스프라이트(학습 이미지와 다른 소스), RAG DB는 테스트 이미지 제외(최소 거리 0.018), 첫 문장 word-boundary 정확도 측정.

| 모델 | 영어 이름 정확도 | 한국어 이름 정확도 |
| :--- | :---: | :---: |
| **Vanilla** | 24/50 (48.0%) | 1/50 (2.0%) |
| **RAG** | **45/50 (90.0%)** | **36/50 (72.0%)** |
| **Tuned** | 24/50 (48.0%) | 2/50 (4.0%) |

### Phase 1 vs Phase 2 비교

| 모델 | Phase 1 v6 *(결함 있음)* | Phase 2 v7 *(교정됨)* | 차이 |
| :--- | :---: | :---: | :---: |
| Vanilla | 86.7% | 48.0% | -38.7%p |
| RAG | 100% | **90.0%** | -10.0%p |
| Tuned | 80.0% | 48.0% | -32.0%p |

> Phase 1 수치가 높았던 이유: ① 테스트 이미지와 DB 이미지 동일 (Dist=0.00, 데이터 누수), ② 서브스트링 매칭, ③ n=15 소규모.

### OOD 성능 (v9, n=30, Gen3-9)

| 카테고리 | Vanilla | RAG |
| :--- | :---: | :---: |
| 진화계열 (예: 에레키블) | **30.0%** | 10.0% |
| 시각적 유사 | 10.0% | **20.0%** |
| 완전 이질적 | 10.0% | 10.0% |
| **전체** | **16.7%** | **13.3%** |

### 핵심 발견

1. **RAG 우위는 실재하지만 Phase 1보다 덜 극적** (90% vs 48%, Phase 1의 100% vs 87%가 아님)
2. **Tuned 모델은 과적합이 아니라 학습 미흡** — 학습 이미지와 새 이미지 모두 48%로 동일 (F6 진단, n=41, 차이 0.0%p)
3. **RAG가 OOD 진화계열에서 오히려 역효과** — "에레브" 힌트를 받으면 모델이 "에레브"를 그대로 출력
4. **한국어 이름 정확도 차이가 가장 명확한 지표**: RAG 72% vs Vanilla 2% vs Tuned 4%

### Phase 2 샘플 결과 (v7, n=50 — 공정한 평가)

이미지: PokeAPI 공식 아트워크 · RAG DB: 학습 이미지만 (테스트 이미지 미포함) · 정확도: word-boundary, 첫 문장

| 이미지 | 정답 | Vanilla | RAG | Tuned |
| :---: | :---: | :--- | :--- | :--- |
| <img src="docs/reports/assets/images/phase2/machamp.png" width="80"><br>**괴력몬** | Machamp<br>(괴력몬) | ❌ "Groudon" | ✅ Machamp **(괴력몬)** | ❌ "Groudon" |
| <img src="docs/reports/assets/images/phase2/alakazam.png" width="80"><br>**후딘** | Alakazam<br>(후딘) | ❌ "Gallade" | ✅ Alakazam **(후딘)** | ❌ "Gallade" |
| <img src="docs/reports/assets/images/phase2/gyarados.png" width="80"><br>**갸라도스** | Gyarados<br>(갸라도스) | ❌ "Dragonair" | ✅ Gyarados *(기라도스)* | ❌ "Dragonair" |
| <img src="docs/reports/assets/images/phase2/charizard.png" width="80"><br>**리자몽** | Charizard<br>(리자몽) | ✅ Charizard *(챌리조드)* | ✅ Charizard **(리자몽)** | ✅ Charizard *(챌리조드)* |
| <img src="docs/reports/assets/images/phase2/meowth.png" width="80"><br>**나옹** | Meowth<br>(나옹) | ✅ Meowth *(미우스)* | ✅ Meowth **(나옹)** | ✅ Meowth *(미우스)* |
| <img src="docs/reports/assets/images/phase2/dragonite.png" width="80"><br>**망나뇽** | Dragonite<br>(망나뇽) | ✅ Dragonite | ✅ Dragonite **(망나뇽)** | ✅ Dragonite |

> **굵은 한국어** = 공식 이름 정답 · *기울임 한국어* = 오답/음역 · 거리 범위: 0.018–0.191 (Dist=0.00 누수 없음)

**패턴 해설:**
- 1–2행: Vanilla·Tuned 완전 실패 구간에서 RAG가 구출
- 3행: RAG 영어 이름 정답, 한국어는 DB에 *기라도스*로 저장되어 오답 (공식명: 갸라도스)
- 4–5행: Tuned가 Vanilla와 동일한 오답 한국어 출력 → **과적합이 아닌 학습 미흡 증거**
- 6행: 모두 영어 정답, 한국어는 RAG만 정확 (망나뇽)

### Phase 1 샘플 결과 (v3/v4 — 정성 평가)

| 이미지 | 정답 | Vanilla | RAG | Tuned |
| :---: | :---: | :--- | :--- | :--- |
| <img src="docs/reports/assets/images/pokemon_117.jpg" width="80"><br>**블래키** | Umbreon<br>(블래키) | ✅ 정답 | ✅ 정답 + 한국어 | ✅ 정답 |
| <img src="docs/reports/assets/images/pokemon_025.jpg" width="80"><br>**별가사리** | Staryu<br>(별가사리) | ⚠️ "Staraptor" | ✅ 정답 + 한국어 | ❌ "별 모양 물체" |

### 결론 (Phase 2 업데이트)

| 방식 | 최적 용도 | 한계 |
| :--- | :--- | :--- |
| **RAG** | ✅ **프로덕션** — 높은 정확도, 한국어 이름 제공 | OOD 진화계열에서 힌트가 혼란 유발 |
| **Tuned** | 현재 설정으로는 명확한 이점 없음 | 학습 미흡: 학습 데이터 62%에 이름 레이블 없음 |
| **Vanilla** | 빠른 프로토타이핑, Gen1-2 연관 없는 OOD | 한국어 이름 불가 |

> **권장**: 정확도·한국어 이름을 위해 **RAG** 사용. 현재 데이터 품질로는 미세조정이 포켓몬 이름 인식을 개선하지 못함 — 학습 데이터 레이블 보완이 선행되어야 함 (Phase 3 방향).

## 📚 레슨런 (Lessons Learned)

### 1. MLX LoRA 어댑터 인퍼런스 이슈 (M-RoPE)
- **문제**: LoRA 학습 성공(Loss 0.0006) 후에도 인퍼런스 시 출력이 생성되지 않음
- **원인**: `mlx_vlm`의 `generate()` 함수가 M-RoPE 상태 관리 실패
- **해결**: **모델 퓨전** — LoRA 가중치를 베이스 모델에 영구 병합

### 2. EOS 토큰 학습 실패 (과소적합)
- **문제**: 초기 튜닝 시 반복적인 쓰레기 출력 (`!!!!`)
- **원인**: 학습이 너무 일찍 중단됨 (20-30 steps)
- **해결**: Loss가 1.0 이하로 수렴할 때까지 **600+ steps** 학습

### 3. 왜 16-bit로 학습해야 하는가?
- **학습 시**: 4-bit 정밀도로는 그래디언트가 0으로 사라져 학습 불가
- **퓨전 시**: 베이스 모델 역양자화 → 16-bit LoRA 병합 → 4-bit 재양자화

### 4. RAG vs Fine-tuning 트레이드오프 (Phase 2 수정)
- **Phase 1 결론**: RAG가 정확도·비용 모두에서 미세조정 능가
- **Phase 2 수정**: 격차는 실재하나(90% vs 48%), 미세조정 실패의 원인은 아키텍처가 아닌 **데이터 품질 문제**

### 5. RAG 메타데이터 서브스트링 버그
- **문제**: "signature"가 "natu"를 매칭 → 팽도리가 네이티로 표시
- **해결**: **정규식 단어 경계** (`\b{name}\b`) 사용

### 6. 학습 데이터 62%에 이름 레이블 없음 (Phase 2 발견)
- **문제**: 학습 데이터 323/520개의 캡션이 `"A blue crab-like Pokemon with claws…"` 형식 — 포켓몬 이름 없음
- **원인**: `setup_pokemon_data.py`에서 이름 미매칭 시 원본 GPT-4 캡션 그대로 train에 배정
- **영향**: 이름 레이블 없는 데이터로 학습하면 이름 매핑 자체를 학습할 수 없음 → 학습 미흡(H2) 원인
- **해결 방향**: PokeAPI로 미레이블 샘플 재식별하거나 필터링

### 7. RAG 힌트가 OOD 진화계열에서 역효과 (Phase 2 발견)
- **문제**: 진화형 포켓몬 입력 시 RAG가 진화 전 포켓몬을 힌트로 제공 → 모델이 힌트를 그대로 답변
  - 에레키블 입력 → 힌트: "This is Electabuzz" → 출력: "Electabuzz"
- **영향**: evolution 카테고리 OOD 정확도: RAG 10% < Vanilla 30%
- **교훈**: Gen1-2 DB만으로는 Gen3+ 진화형에 RAG가 역효과. DB를 전 세대로 확장해야 함.

### 8. 평가 설계가 모델 성능보다 더 중요할 수 있다 (Phase 2 발견)
- **Phase 1 결함**: 테스트 이미지 = DB 이미지 (Dist=0.00) → RAG 점수 10~40%p 과장
- **교훈**: 검색 거리를 항상 확인할 것. Dist=0.00이면 "답안지를 보고 답하는 것"과 같음.

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
| `src/eval_utils.py` | Phase 2 평가 유틸리티 (word-boundary 정확도) |
| `src/server.py` | FastAPI 인퍼런스 서버 |
| `scripts/train/lora_v3.py` | LoRA 학습 스크립트 |
| `scripts/train/fuse_vlm.py` | 모델 퓨전 스크립트 |
| `scripts/setup/build_rag_db_fair.py` | 공정한 RAG DB 구축 스크립트 |
| `scripts/setup/download_eval_sprites.py` | PokeAPI 스프라이트 수집 (50종) |
| `models/fused_qwen2_vl_4bit_quantized/` | 최종 퓨전 모델 (4.3GB) |

## 📄 평가 보고서

### Phase 2 (교정된 평가)
| 리포트 | 설명 |
| :--- | :--- |
| [비판적 분석](docs/reports/CRITICAL_ANALYSIS.md) | Phase 1의 7가지 구조적 결함 분석 |
| [v7 — 공정한 벤치마크](docs/reports/EVALUATION_REPORT_v7_FAIR_BENCHMARK.md) | n=50, 누수 없음, word-boundary 정확도 |
| [v8 — Tuned 모델 진단](docs/reports/EVALUATION_REPORT_v8_TUNED_DIAGNOSIS.md) | 학습 미흡 vs 과적합 판정 (n=41) |
| [v9 — OOD 강화 테스트](docs/reports/EVALUATION_REPORT_v9_OOD_ENHANCED.md) | Gen3-9, 30종, 3카테고리 분석 |
| [Phase 2 최종 보고서](docs/reports/PHASE_2_FINAL_REPORT.md) | 수정된 결론 및 권장사항 종합 |

### Phase 1 (원본, 알려진 결함 있음)
| 리포트 | 설명 |
| :--- | :--- |
| [v3 — 일반 프롬프트](docs/reports/EVALUATION_REPORT_v3.md) | 정성 평가 기준선 |
| [v4 — 힌트 프롬프트](docs/reports/EVALUATION_REPORT_v4.md) | "What Pokemon is this?" 포함 |
| [v5 — OOD 함정 테스트](docs/reports/EVALUATION_REPORT_v5_OOD_TRAP_TEST.md) | 3세대+ 미학습 종 (RAG 수치 과장됨) |
| [v6 — 일반화](docs/reports/EVALUATION_REPORT_v6_GENERALIZATION.md) | 학습된 포켓몬의 다른 이미지 (수치 과장됨) |
| [학습 세대 분리 검증](docs/reports/TRAIN_GEN_SPLIT_VERIFICATION.md) | 62.1% 미레이블 학습 데이터 발견 |

## ⚠️ 면책 조항
- **비공식 프로젝트**: Nintendo, Game Freak, The Pokémon Company와 무관합니다.
- **데이터셋**: [diffusers/pokemon-gpt4-captions](https://huggingface.co/datasets/diffusers/pokemon-gpt4-captions) 라이선스에 따라 비상업적 용도로만 사용 가능.

## 🤝 감사의 말
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Hugging Face Diffusers](https://huggingface.co/diffusers/pokemon-gpt4-captions)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
