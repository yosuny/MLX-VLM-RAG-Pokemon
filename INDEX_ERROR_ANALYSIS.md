# IndexError 심층 분석 보고서

## 🚨 문제 원인: Vocab Mismatch로 인한 학습 발산

### 1. 팩트 체크
- **Tokenizer Vocab Size**: `151,643` (실제 해독 가능 토큰 수)
- **Model Config Vocab Size**: `152,064` (모델 출력 레이어 크기)
- **차이**: 약 421개의 토큰이 모델에는 존재하지만, Tokenizer는 모르는 상태(Unknown/Special Tokens 영역)입니다.

### 2. 발생 메커니즘
1. **Model Structure**: Qwen2-VL 모델은 비전 처리를 위해 Vocab Size를 `152,064`로 확장해 두었습니다.
2. **Training Stability**: `1e-4`라는 높은 Learning Rate와 4bit 양자화 노이즈가 결합되어, 학습 중 **Gradient Explosion(기울기 폭발)**이 발생했습니다.
3. **Weight Corruption**: 이로 인해 LoRA 어댑터 가중치가 손상되어, 모델의 출력이 불안정해졌습니다.
4. **The Critical Failure**: 손상된 모델이 확률적으로 **151,657번 이상의 토큰(Tokenizer가 모르는 영역)**에 높은 점수를 부여하기 시작했습니다.
5. **Inference Crash**: 생성된 토큰 ID(예: 152,000)가 Detokenizer로 전달되는데, Tokenizer 리스트에는 151,657번까지만 있으므로 **`IndexError: list index out of range`**가 발생합니다.

### 3. 왜 Base Model은 괜찮은가?
Base Model은 수조 개의 데이터로 학습되어 이 영역의 확률이 0에 가깝게 잘 제어되어 있습니다. LoRA Fine-tuning이 이 균형을 깨트린 것입니다.

---

## ✅ 해결 방안: "학습 안정화"

이 에러는 단순히 인덱스 문제가 아니라, **모델이 망가졌다는 신호**입니다. 따라서 모델을 "덜 격렬하게" 학습시켜야 합니다.

1. **Learning Rate 감소**: `1e-4` → `1e-5` (필수)
   - 가중치 변화폭을 줄여 "모르는 영역"으로 튀는 것을 방지합니다.
2. **Steps 축소**: 200 → 100
   - 과적합을 방지합니다.

이 설정으로 재학습하면 모델이 정상적인 Vocab 범위 내에서 토큰을 생성하게 될 것입니다.
