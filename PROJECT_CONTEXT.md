# PROJECT_CONTEXT.md
최종 갱신: 2026-05-15

## 프로젝트 개요
**MLX-VLM-RAG-Pokemon** — Apple MLX 프레임워크를 이용한 포켓몬 식별 VLM 파인튜닝 + RAG 실험 프로젝트

- **목적**: 800종+ 포켓몬을 한국어 이름과 함께 식별하는 AI 시스템 구축 및 평가
- **접근법**: RAG(검색 증강 생성) vs LoRA 파인튜닝 비교 실험
- **환경**: Apple Silicon MacBook, 48GB 메모리, MLX 프레임워크

---

## 기술 스택

| 구성요소 | 기술 |
|---|---|
| **베이스 모델** | Qwen2-VL-7B-Instruct (4-bit 양자화, mlx-community) |
| **파인튜닝** | LoRA (mlx-vlm), 16-bit 학습 후 4-bit 재양자화 |
| **비전 인코더 (RAG)** | SigLIP So400m (google/siglip-so400m-patch14-384) |
| **벡터 DB** | ChromaDB (로컬 PersistentClient) |
| **프레임워크** | MLX, PyTorch (SigLIP 임베딩용), FastAPI |
| **데이터** | diffusers/pokemon-gpt4-captions (883장) + 한국어 이름 보강 |

---

## 모델 현황

| 모델 | 경로 | 크기 | 상태 |
|---|---|---|---|
| Base (4-bit) | `mlx-community/Qwen2-VL-7B-Instruct-4bit` | HuggingFace | 다운로드 필요 |
| Fused 16-bit | `models/fused_qwen2_vl_4bit` | 15GB | 로컬 존재 |
| Fused 4-bit | `models/fused_qwen2_vl_4bit_quantized` | 4.4GB | 로컬 존재 (운영용) |

---

## Phase 마일스톤

### ✅ Phase 1 — VLM + RAG 구축 및 초기 평가 (완료)
- 데이터 준비, RAG 시스템 구축, LoRA 학습, 모델 퓨전, v3~v6 평가
- 스냅샷: `snapshots/phase_1/`
- 주요 결과: RAG > Tuned > Vanilla (단, 평가 설계 결함 존재)
- 비판적 분석: `docs/reports/CRITICAL_ANALYSIS.md`

### 🔄 Phase 2 — 공정한 평가 재설계 및 재실험 (진행 중)
- 평가 설계 결함 해소, 공정한 벤치마크 수립, 신뢰할 수 있는 결론 도출
- 스냅샷: `snapshots/phase_2/`
- 계획서: `docs/IMPROVEMENT_TEST_PLAN.md`

---

## 핵심 파일 위치

| 파일 | 설명 |
|---|---|
| `src/rag_engine.py` | SigLIP + ChromaDB 검색 엔진 |
| `src/server.py` | FastAPI 웹 서버 (추론) |
| `src/eval_utils.py` | 공통 평가 유틸리티 **(Phase 2 생성 예정)** |
| `scripts/train/lora_v3.py` | Phase 1 성공 LoRA 학습 스크립트 |
| `scripts/train/fuse_vlm.py` | 모델 퓨전 스크립트 |
| `scripts/eval/` | 평가 스크립트 모음 |
| `data/pokemon/` | 학습/검증 이미지 및 JSONL |
| `chroma_db/` | 현재 RAG 벡터 DB (Phase 1 기준, 재구축 필요) |

---

## 알려진 기술 이슈

1. **M-RoPE 상태 관리**: mlx_vlm의 generate()가 LoRA 어댑터 추론 시 출력 미생성 → 모델 퓨전으로 해결
2. **이미지 토큰 확장**: apply_chat_template이 이미지 토큰 1개만 삽입 → lora_v3.py에서 수동 확장
3. **평가 데이터 누수**: Phase 1 RAG 평가에서 테스트 이미지와 DB 이미지 동일 (Dist:0.00) → Phase 2에서 해소 예정
4. **macOS Python**: 시스템 Python 3.9 사용, 패키지는 `.venv` 관리

---

## 환경 설정

```bash
# 가상환경 활성화
source .venv/bin/activate

# 웹 서버 기동
uvicorn src.server:app --reload --port 8000

# Phase 2 초기화
bash snapshots/phase_2/init.sh
```
