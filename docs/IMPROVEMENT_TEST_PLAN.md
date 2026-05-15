# 개선 테스트 계획 (Improvement Test Plan)

**작성일**: 2026-05-15  
**전제**: `docs/reports/CRITICAL_ANALYSIS.md`에서 식별된 7가지 결함을 해소하기 위한 재실험 계획  
**환경**: MacBook M-series, 48GB 메모리

---

## 우선순위 요약

| 우선순위 | 개선 항목 | 해소 결함 | 예상 공수 |
|---|---|---|---|
| P0 | 공정한 RAG 평가 재설계 | 결함 1 | 중 |
| P1 | 정확도 판정 기준 강화 | 결함 2, 3 | 소 |
| P2 | 평가 셋 확대 + 재실험 | 결함 4, 5 | 대 |
| P3 | Tuned 모델 실패 원인 재진단 | 결함 6 | 중 |
| P4 | 학습 데이터 구성 검증 | 결함 7 | 소 |

---

## P0. 공정한 RAG 평가 재설계

### 문제
현재 RAG DB = 학습 이미지 전체 → 테스트 시 동일 이미지 Dist:0.00으로 검색됨.

### 개선 방안 (2가지 옵션)

**옵션 A: Holdout RAG DB** (권장)
- RAG DB를 학습 이미지로만 구성하되, **테스트 이미지는 DB에서 제외**
- 즉, 테스트 이미지 `pokemon_025.jpg`는 DB에 인덱싱 안 됨
- 동일 포켓몬의 **다른 이미지**가 DB에 있어야 의미 있는 테스트
- 현재 데이터셋에 같은 포켓몬의 다수 이미지가 있는지 확인 필요

**옵션 B: 완전 분리 (v6 방식 확장)**
- 학습 데이터(Gen1-2 이미지)를 RAG DB로 사용
- 테스트 데이터는 **PokeAPI / 공식 artwork** 등 완전히 다른 소스
- v6 방식을 체계화하여 모든 평가에 적용

**권장**: 옵션 B — v6에서 이미 검증된 방식을 표준화

### 구현 계획
```
scripts/eval/
  evaluate_fair_rag.py      # RAG DB와 테스트셋 완전 분리 보장
  build_rag_db_train_only.py # 학습 이미지만 DB 구축 (테스트 이미지 제외 명시)
```

---

## P1. 정확도 판정 기준 강화

### 문제
- v6: `target.lower() in output.lower()` — false positive 가능
- v5: 정량 수치 재현 불가

### 개선 방안

**정확도 판정 함수 v2** (엄격한 기준):
```python
import re

def check_accuracy_v2(output: str, target_name: str) -> bool:
    """
    단어 경계(word boundary) 기반 이름 매칭.
    'Pikachu'가 포함된 경우만 정답, 'PikachuExtra' 등은 제외.
    """
    pattern = rf'\b{re.escape(target_name)}\b'
    return bool(re.search(pattern, output, re.IGNORECASE))

def check_accuracy_strict(output: str, target_name: str) -> bool:
    """
    첫 번째 문장에서 이름 매칭 (전체 응답에서 우연히 포함되는 경우 제외).
    """
    first_sentence = output.split('.')[0]
    pattern = rf'\b{re.escape(target_name)}\b'
    return bool(re.search(pattern, first_sentence, re.IGNORECASE))
```

**추가 지표 (선택)**:
- Korean name accuracy (한국어 이름 별도 측정)
- "모른다/알 수 없다" 응답 비율 (hallucination vs abstention 구분)

### 구현 계획
```
src/eval_utils.py   # 공통 정확도 판정 함수 모듈화
```

---

## P2. 평가 셋 확대 및 재실험 (핵심)

### 문제
- v3/v4: n=4 (통계 불가)
- v5: n=17 (불명확)
- v6: n=15

### 목표 샘플 수

| 평가 유형 | 현재 | 목표 | 비고 |
|---|---|---|---|
| Trained (Gen1-2) | ~17 | **50** | 전체의 약 10% |
| OOD Gen3+ | 6 | **30** | 진화계열/유사외형 포함 |
| Generalization | 15 | **50** | PokeAPI 공식 스프라이트 |

### 재실험 구성

#### Experiment A: 공정한 RAG 벤치마크 (P0 + P2 통합)

```
테스트 세트: 학습에 사용되지 않은 Gen1-2 이미지 50장 (PokeAPI 스프라이트)
RAG DB:      학습 이미지 (GPT-4 캡션 기반) — 테스트 이미지와 겹치지 않음
비교 모델:   Vanilla / RAG / Tuned
측정 지표:   check_accuracy_strict() 기준
```

이것이 **가장 중요한 실험** — 현재 RAG 70.6%가 재현되는지, 아니면 훨씬 낮아지는지 확인

#### Experiment B: OOD 강화 테스트 (Gen3-9, n=30)

```
테스트 세트: Gen3~9 포켓몬 30종 (PokeAPI 스프라이트)
              - 진화 관련 10종 (Electivire, Togekiss 류)
              - 시각적 유사 10종 (색/형태 유사)
              - 완전 이질적 10종
RAG DB:      Gen1-2 이미지 (현행 유지)
측정:         RAG가 진화계열을 잡아내는 비율 별도 측정
```

#### Experiment C: 파인튜닝 효과 재진단 (P3)

```
목적:    Tuned 모델이 "과적합"인지 "학습 미흡"인지 구분
방법:    학습 이미지 자체 + 동일 포켓몬 다른 이미지 + 완전 다른 포켓몬 3가지 비교
측정:    학습 이미지 정확도 vs 다른 이미지 정확도 차이
기대:    만약 학습 이미지 정확도 >> 다른 이미지 → 과적합
         만약 학습 이미지 정확도 ≈ 다른 이미지 (둘 다 낮음) → 학습 미흡
```

---

## P3. Tuned 모델 실패 원인 재진단

### 가설 정의

| 가설 | 예측 패턴 | 확인 방법 |
|---|---|---|
| H1: 과적합 | 학습 이미지 ✅, 다른 이미지 ❌ | 동일 포켓몬 2가지 이미지 비교 |
| H2: 학습 미흡 | 학습 이미지도 ❌, 다른 이미지도 ❌ | 위와 동일 |
| H3: 텍스트 패턴만 학습 | 이름은 맞지만 설명이 훈련 캡션 복사 | 출력 다양성 분석 |

### 추가 분석

현재 Tuned 모델 응답 패턴 관찰:
- `pokemon_025.jpg`(Staryu): "Staraptor" — 학습 전과 동일한 오답
- `pokemon_323.jpg`(Clefable): "Gardevoir" — Vanilla와 동일한 오답
- `pokemon_674.jpg`(Kadabra): "Groudon" — Vanilla와 동일한 오답

이 패턴은 Tuned 모델이 사실상 **Vanilla와 동일하게 동작**하고 있음을 시사합니다.
즉, "학습 미흡" 가설(H2)이 더 가능성 있습니다.

---

## P4. 학습 데이터 구성 검증

### 목적
Gen1-2 분리가 실제로 지켜졌는지 확인.

### 검증 스크립트
```python
# scripts/debug/verify_train_gen_split.py
# JSONL의 각 이미지에 대해 포켓몬 이름 → 도감번호 → 세대 매핑 검증
```

확인 항목:
- 학습 JSONL에 Gen3+ 이미지가 포함되어 있는지
- 검증 JSONL에 Gen1-2 이미지가 포함되어 있는지
- `pokemon_684.jpg`, `pokemon_830.jpg` 등 이름 없는 샘플의 실제 포켓몬 확인

---

## 실험 실행 순서 (권장)

```
Phase 1 (빠른 검증, 1-2일):
  ① P4: 학습 데이터 구성 검증 (가장 빠름, 기반 확인)
  ② P1: eval_utils.py 모듈 작성 (정확도 함수 통일)

Phase 2 (핵심 재실험, 3-5일):
  ③ P0: 공정한 RAG DB 구축 (테스트 이미지 제외)
  ④ P2-A: 공정한 RAG 벤치마크 (n=50, PokeAPI 스프라이트)
  ⑤ P3: Tuned 모델 과적합 vs 미흡 진단

Phase 3 (강화 테스트, 필요시):
  ⑥ P2-B: OOD 강화 테스트 (n=30, Gen3-9)
  ⑦ P2-C: 한국어 이름 정확도 별도 측정
```

---

## 48GB 메모리 활용 방안

현재 환경에서 가능한 추가 실험:

| 실험 | 메모리 요구 | 비고 |
|---|---|---|
| 16-bit 퓨전 모델 추론 | ~16GB | `models/fused_qwen2_vl_4bit` (15GB) 이미 존재 |
| Qwen2-VL-7B 16-bit | ~16GB | 4-bit 대비 품질 비교 |
| 더 큰 배치 재학습 | ~20GB | LoRA rank 증가, batch_size 4→8 |
| Gen1-9 전체 RAG DB | ~메모리 무관 | 디스크/인덱싱 시간 문제 |

**가장 가치 있는 활용**: Phase 2 재실험에서 **16-bit 모델과 4-bit 모델 비교** 추가
- 4-bit 양자화가 평가 결과에 영향을 주는지 확인 가능

---

## 예상 결과 시나리오

### 시나리오 A (낙관): RAG가 실제로 강함이 확인되는 경우
- 공정한 평가(P0)에서도 RAG > Vanilla 유의하게 나옴
- OOD에서 진화계열 매칭 효과 통계적으로 확인
- → "RAG 권장" 결론 유지, 단 근거가 탄탄해짐

### 시나리오 B (현실적): RAG 성능이 크게 하락하는 경우
- 공정한 평가에서 RAG ≈ Vanilla 또는 근소한 차이
- → "RAG가 도움이 되지만 현재 수치는 과장됨" 으로 결론 수정
- → RAG DB 품질(캡션 정확도, 이미지 다양성) 개선 방향 탐색

### 시나리오 C: Tuned 모델이 사실 학습 미흡이었던 경우
- H2 가설 확인: Vanilla와 거의 동일한 오답 패턴
- → LoRA 학습 설정 재검토 필요 (rank 증가, 학습 데이터 확대, 더 많은 steps)
- 48GB 환경에서 더 강력한 재학습 시도 가능

---

*이 계획은 결함 발견 시 업데이트 예정.*  
*관련 문서: `docs/reports/CRITICAL_ANALYSIS.md`*
