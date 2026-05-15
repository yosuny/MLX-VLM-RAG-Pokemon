# 실험 비판적 분석 (Critical Analysis)

**작성일**: 2026-05-15  
**분석 범위**: v3, v4, v5 (OOD Trap), v6 (Generalization) 평가 전체  
**분석 대상**: 결과 보고서 + 평가 스크립트 + raw JSON 데이터 + 학습 로그

---

## 요약

현재 실험에서 도출된 주요 결론("RAG > Tuned > Vanilla", "RAG가 프로덕션 권장")은
아래 7가지 구조적 결함으로 인해 신뢰하기 어렵다.
**신뢰 가능한 결론은 단 하나**: "현재 LoRA 파인튜닝 설정으로는 포켓몬 이름을 정확히 학습시키기 어렵다."

---

## 결함 1 🔴 RAG 평가의 구조적 데이터 누수 (가장 심각)

### 근거
`docs/reports/raw_data/ood_trap_test_results.json`의 trained 샘플 전체:

| 이미지 | 검색 거리 |
|---|---|
| pokemon_117.jpg (Umbreon) | Dist: **0.00** |
| pokemon_025.jpg (Staryu) | Dist: **0.00** |
| pokemon_344.jpg | Dist: **0.00** |
| pokemon_033.jpg (Pinsir) | Dist: **0.00** |
| pokemon_674.jpg (Kadabra) | Dist: **0.00** |
| ... 모든 trained 샘플 | Dist: **0.00** |

### 문제
- RAG ChromaDB는 **학습용 이미지**로 인덱싱됨
- v5 "Trained" 평가 세트는 **동일한 학습 이미지**를 사용
- 결과적으로 RAG는 테스트 이미지를 DB에서 그대로 찾는 구조 (오픈북 시험)

### 영향
- v5 Gen1-2 "RAG 70.6%" 수치는 검색 능력이 아닌 **데이터 누수**의 산물
- RAG가 100%가 안 된 이유는 실패가 아니라, 일부 ground truth 캡션에 이름이 없기 때문
  - 예: `pokemon_344.jpg` GT = "A blue, crab-like Pokémon..." (이름 없음)
- **v3/v4/v5 trained에서의 RAG 수치는 모두 무효**

---

## 결함 2 🔴 v5 보고서 수치 오류 및 n=17 근거 불명

### 근거
`docs/reports/EVALUATION_REPORT_v5_OOD_TRAP_TEST.md` 헤더:
```
Total Samples: 30
Trained (Gen 1-2): 30 (80%)   ← 오기. 30 × 80% = 24여야 함
Untrained Val (Gen 3+): 6 (20%)
```

정확도 표의 `n=17` 근거:
- 코드(`evaluate_ood_trap_test.py`)에 accuracy 계산 로직이 없음
- 보고서 Note: "Excluded 13 samples from Trained set due to missing ground truth names"
- 그러나 24 - 13 = **11**인데 보고서는 **n=17**

### 문제
- 수치 17의 산출 방식이 코드에 없어 재현 불가
- 헤더의 "30 (80%)" 오기로 전체 보고서 신뢰도 하락
- Accuracy 계산이 수동으로 이루어졌을 가능성이 높고, 그 기준 미명시

---

## 결함 3 🔴 정확도 판정 기준 오류 (v6)

### 근거
`scripts/eval/evaluate_models_v6.py:243`:
```python
def check_accuracy(output, target):
    return target.lower() in output.lower()
```

### 문제
- `target = "Pikachu"` 일 때, `"This is not Pikachu, this is Raichu"` → **정답 처리**
- `target = "Eevee"` 일 때, 부분 문자열이 다른 단어에 포함되는 경우도 정답
- v5에는 이 함수 자체가 없음 — 정량 수치 재현 불가

### 영향
- v6 정확도 수치(Vanilla 86.7%, RAG 100%, Tuned 80.0%)의 신뢰도 의문
- 특히 RAG 100%는 "정답 이름이 출력 어딘가에 포함"된 것이지 "정확한 식별"이 아닐 수 있음

---

## 결함 4 🟠 통계적 유의성 없는 소규모 샘플

| 평가 | 실제 샘플 수 | 1개 차이의 영향 | 문제 |
|---|---|---|---|
| v3/v4 | **4개** | 25%p | 통계 불가 |
| v5 Gen1-2 | **n=17** (불명확) | 5.9%p | 기준 불명확 |
| v6 | **15개** | 6.7%p | 통계 검증 없음 |

v6의 "Tuned underperforms Vanilla" 결론은 12/15 vs 13/15 차이(1개 차이)에 근거합니다.
신뢰구간 95%에서 이 차이는 통계적으로 유의하지 않습니다.

---

## 결함 5 🟠 평가 조건 불일치 — 비교 자체가 무의미

| 평가 | RAG DB 구성 | 테스트 이미지 출처 | 실질적 의미 |
|---|---|---|---|
| v3/v4/v5 trained | 학습 이미지 | **동일한** 학습 이미지 | RAG는 자신을 찾음 |
| v5 OOD (trap) | 학습 이미지 | Gen3+ 미학습 이미지 | 공정한 RAG 테스트 |
| v6 | 학습 이미지 | **다른** 이미지 (PokeAPI) | 그나마 공정 |

README의 한 표에서 v5 Gen1-2 수치와 v6 수치를 나란히 비교하고 있는데,
두 실험의 RAG 평가 조건이 근본적으로 다릅니다.

---

## 결함 6 🟠 Tuned 모델 실패 원인 진단 불명확

v5에서 Tuned 모델이 `pokemon_025.jpg`(Staryu)를 여전히 "Staraptor"로 답합니다.
이는 두 가지 해석이 가능합니다:

- **해석 A (채택된 결론)**: 파인튜닝이 되었지만 과적합(overfitting)
- **해석 B (배제된 가능성)**: 파인튜닝 자체가 VLM의 시각적 인식 레이어에 효과적으로 작용하지 않음

현재 실험 설계로는 두 해석을 구분할 수 없습니다.
학습 loss가 0.0006까지 수렴했음에도 동일 이미지에서 틀린다면,
이는 **loss가 텍스트 패턴에만 수렴하고 이미지-이름 매핑 학습은 미흡**했을 가능성을 시사합니다.

---

## 결함 7 🟡 학습/검증 세대 분리 미검증

훈련 데이터가 "Gen1-2"라고 명시했지만, 학습 JSONL 파일의 실제 구성이 Gen1-2만으로
이루어졌는지 검증한 기록이 없습니다.

실제로 다음 이미지들이 "TRAINED" 세트에 포함되어 있음:
- `pokemon_684.jpg`: "bipedal, mechanical dragon-like creature" (파일명으로 세대 불명)
- `pokemon_830.jpg`: "bipedal yellow feline Pokémon brandishes a silver pendulum" (파일명으로 세대 불명)

파일명 번호는 포켓몬 도감 번호가 아닌 임의 번호입니다. Gen1-2 분리 근거가 코드에 없습니다.

---

## 신뢰할 수 있는 결론 (재정리)

| 기존 결론 | 신뢰 수준 | 실제 상황 |
|---|---|---|
| "RAG Gen1-2 70.6%" | ❌ 무효 | 데이터 누수(Dist:0.00) — 재실험 필요 |
| "RAG 100% 일반화" | ⚠️ 낮음 | n=15, substring match — 기준 재정의 필요 |
| "Tuned 모델 과적합" | ⚠️ 낮음 | 파인튜닝 효과 자체가 없을 가능성 미배제 |
| "Tuned 17.6% < Vanilla 23.5%" | ⚠️ 낮음 | n=17 산출 기준 불명확 |
| "RAG가 프로덕션 권장" | ❌ 근거 미흡 | 위 결함들 해소 전 결론 불가 |
| **"LoRA 파인튜닝 단독으로는 이름 학습이 어렵다"** | ✅ 신뢰 | 다수 사례에서 일관되게 관찰됨 |

---

## 참고: 신뢰할 수 있는 데이터 — v5 OOD (Trap) 케이스

v5에서 Gen3+ trap 케이스는 RAG DB에 해당 이미지가 없으므로 (Dist: 0.15~0.28)
유일하게 공정한 RAG 평가입니다.

| 포켓몬 | Dist | RAG 반환 | 평가 |
|---|---|---|---|
| Electivire | 0.20 | Electabuzz (관련 진화) | 합리적 |
| Munchlax | 0.28 | Snorlax (진화 전) | 합리적 |
| Lickilicky | 0.15 | Lickitung (진화 전) | 합리적 |
| Togekiss | 0.20 | Togetic (진화 전) | 합리적 |
| Leafeon | 0.18 | Flareon (Eevee 진화) | 시각적 유사 |
| Glaceon | 0.21 | Umbreon | 오검색 |

이 결과는 SigLIP이 진화 계열을 시각적으로 유사하게 인식한다는 흥미로운 발견입니다.
단 n=6으로 통계적 결론을 내리기엔 부족합니다.
