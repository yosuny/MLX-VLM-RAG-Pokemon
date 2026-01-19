# LoRA 어댑터 로딩 개선 계획

## 현재 상태 분석

### ✅ 리포트 v2 업데이트 확인
- **바닐라(Vanilla)**: 정상 작동 - 실제 포켓몬 이름 생성 ("This Pokémon is named Umbreon", "Groudon" 등)
- **RAG**: 정상 작동 - 유사 이미지 기반 답변 생성
- **Tuned**: Base 모델로 동작 중 (어댑터 로딩 실패로 인해 Vanilla와 동일)

### ❌ 현재 문제
```python
# evaluate_models_v2.py의 현재 방식
adapters = list(mx.load(adapter_path).items())
model.update(tree_unflatten(adapters))
# 오류: Module does not have parameter named "B"
```

**원인**:
- 어댑터 파일 자체는 정상 (392개 키, 예: `'language_model.model.layers.9.mlp.down_proj.B'`)
- `model.update(tree_unflatten(...))` 방식이 모델 구조와 매칭 실패
- 모델 로드 시점과 어댑터 적용 시점의 파라미터 네이밍 불일치

---

## 개선 방안

### Option 1: `mlx_vlm.load()`의 `adapter_path` 직접 사용 ⭐ (권장)

**개요**:
- `mlx_vlm.load()` 함수가 모델과 어댑터를 동시에 로딩
- `server.py`에서 이미 검증된 방식

**구현**:
```python
# Phase 1: Vanilla + RAG (Base 모델)
model, processor = load(model_path)
# ... Vanilla + RAG 추론 ...

# Phase 2: Tuned (어댑터 적용 모델)
# 기존 모델 메모리 해제
del model
del processor
gc.collect()
mx.metal.clear_cache()

# 어댑터 포함하여 재로딩
model_tuned, processor_tuned = load(model_path, adapter_path="adapters")
# ... Tuned 추론 ...
```

**장점**:
- ✅ `mlx_vlm` 라이브러리가 내부 처리 (안전)
- ✅ `server.py`에서 검증됨
- ✅ 파라미터 매칭 이슈 없음

**단점**:
- 모델 재로딩으로 인한 시간 추가 (~30초)
- 메모리 사용량 일시 증가

---

### Option 2: LoRA 파라미터 수동 매핑

**개요**:
- 어댑터 키와 모델 파라미터 구조를 직접 매칭

**구현**:
```python
adapters = mx.load(adapter_path)
# 모델의 실제 파라미터 트리 구조 확인
from mlx.utils import tree_flatten
model_params = tree_flatten(model.parameters())

# 키 매핑 수동 조정
# 예: 'language_model.model.layers.X.mlp.down_proj.B' 
#  -> model.language_model.model.layers[X].mlp.down_proj.lora_B
```

**장점**:
- 모델 재로딩 불필요
- 메모리 효율적

**단점**:
- ❌ 복잡하고 오류 가능성 높음
- ❌ `mlx_vlm` 내부 구조 변경 시 깨질 수 있음

---

### Option 3: 학습 스크립트 수정 후 재학습

**개요**:
- `patched_lora.py`의 어댑터 저장 방식을 수정하여 재학습

**장점**:
- 근본적인 해결

**단점**:
- ❌ 200 스텝 재학습 필요 (~40분)
- ❌ 현재 어댑터가 이미 학습 완료

---

## 권장 솔루션: Option 1

`evaluate_models_v2.py`를 다음과 같이 수정:

1. **Phase 1**: Base 모델로 Vanilla + RAG 추론
2. **메모리 정리**: 모델 삭제 및 캐시 클리어  
3. **Phase 2**: 어댑터 포함 모델 재로딩 후 Tuned 추론

**예상 소요 시간**:
- 모델 재로딩: ~30초
- 전체 평가: 기존 5분 → 약 6분

---

## 구현 체크리스트

- [ ] `evaluate_models_v2.py` 수정: Tuned phase를 별도 함수로 분리
- [ ] 메모리 정리 로직 추가 (`gc.collect()`, `mx.metal.clear_cache()`)
- [ ] `load(model_path, adapter_path="adapters")` 호출
- [ ] Robust Processor Wrapper 재적용
- [ ] 테스트 실행 및 결과 검증
- [ ] 리포트 v2 재생성 및 Tuned 결과 비교

---

## 예상 결과

Tuned 모델이 정상 로딩되면:
- Gen 1-2 포켓몬 (훈련 데이터): **정확도 향상** 예상
- Gen 3+ 포켓몬 (검증 데이터): Base 모델과 유사하거나 약간 개선

이를 통해 Fine-tuning의 효과를 정량적으로 평가할 수 있습니다.
