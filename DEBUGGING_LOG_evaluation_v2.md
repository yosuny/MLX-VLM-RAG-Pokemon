# 평가 스크립트 디버깅 로그 (evaluate_models_v2.py)

## 문제 발견 및 해결 과정

### 1차 실행 결과
- **문제**: 모든 모델이 동일하게 "I'm sorry, but I can't identify..." 거부 응답 생성
- **증상**: 토큰 중복 (`II'mI'm sorry...`)

### 진단 과정

#### Phase 1: Prompt 형식 비교
**테스트**: `diagnose_prompt_issue.py` 실행
- MLX 방식 (`mlx_vlm.prompt_utils.apply_chat_template`) vs HF 방식 (`processor.apply_chat_template`) 비교
- **결과**: 두 방식 모두 동일한 prompt 생성 → 형식 문제 아님

#### Phase 2: Detokenizer 분석
**발견된 버그**:
```python
# 문제 코드 (기존 Custom Detokenizer)
def add_token(self, token):
    self.tokens.append(token)
    new_text = self.tokenizer.decode(self.tokens)  # 전체 재디코딩
    segment = new_text[len(self.text):]            # ❌ 토큰 중복 발생
    self.text = new_text
    return segment
```

**원인**: 
- 토큰이 누적되면서 매번 전체를 디코딩
- `last_text` 업데이트 로직 누락으로 세그먼트 계산 오류
- 결과적으로 `II'mI'm sorry...` 같은 중복 발생

### 해결 방안

#### 전면 리팩토링
`server.py`의 검증된 구현을 그대로 적용:

1. **Custom Detokenizer 제거** → MLX 네이티브 사용
2. **Prompt 생성 방식 변경**:
   ```python
   # 기존 (HF 방식)
   formatted_prompt = processor.processor.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
   )
   
   # 수정 (MLX 방식)
   from mlx_vlm.prompt_utils import apply_chat_template
   formatted_prompt = apply_chat_template(
       processor,
       config=model.config,
       prompt=prompt,
       num_images=1
   )
   ```

3. **RobustImageProcessorWrapper 추가**:
   - `ValueError: Only returning PyTorch tensors is currently supported` 에러 해결
   - PyTorch 텐서 → NumPy 자동 변환

### 최종 결과

#### 적용된 변경사항
- ✅ Custom Detokenizer 완전 제거
- ✅ `mlx_vlm.prompt_utils.apply_chat_template` 사용
- ✅ `RobustImageProcessorWrapper` 적용
- ✅ Monkey Patch (Qwen2-VL 패딩 버그) 유지

#### 성공 지표
- ✅ 30개 추론 완료 (Vanilla × 10, RAG × 10, Tuned × 10)
- ✅ 토큰 중복 현상 완전 해결
- ✅ 정상적인 텍스트 생성 확인
- ✅ Exit code: 0 (성공)

### 남은 이슈

#### LoRA 어댑터 로딩 실패
```
Error loading adapters: Module does not have parameter named "B".
Continuing with base model...
```

**원인 추정**:
- `adapters.safetensors` 파일의 키 구조가 현재 모델과 불일치
- 학습 시 저장된 LoRA 파라미터명(`B`)이 현재 모델 구조에 존재하지 않음

**영향**:
- Tuned 모드가 실제로는 Base 모델로 동작
- Vanilla와 Tuned 결과가 사실상 동일할 가능성

**차후 개선 방안**:
1. `patched_lora.py`의 어댑터 저장 로직 검증
2. `mlx_vlm`의 `load()` 함수에 `adapter_path` 직접 전달 방식 시도
3. 어댑터 파일 구조 직접 검사 및 수동 매핑

---

## 코드 변경 이력

### evaluate_models_v2.py (v1 → v2)

#### 제거된 코드
- `class Detokenizer` (전체)
- `class RobustImageProcessorWrapper` (이전 버전 - 부적절한 구현)
- `processor.processor.apply_chat_template()` 호출

#### 추가된 코드
- `from mlx_vlm.prompt_utils import apply_chat_template`
- `class RobustImageProcessorWrapper` (server.py 버전)
- `apply_chat_template(processor, config=model.config, prompt=prompt, num_images=1)`
- Robust Processor 적용 로직

#### 수정된 코드
- 모든 `generate()` 호출에 `temperature=0.1` 파라미터 추가 (일관성)
- RAG 컨텍스트 주입 방식 간소화

---

## 검증 완료
- 날짜: 2026-01-16
- 실행 시간: 약 5분
- 샘플 수: 10 (Train: 7, Valid: 3)
- 총 추론 수: 30회
- 성공률: 100%
