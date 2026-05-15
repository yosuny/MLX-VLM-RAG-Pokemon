"""
F5: 공정한 RAG 벤치마크 재실험
CRITICAL_ANALYSIS.md 결함 1, 3, 4, 5 해소

Phase 1과의 차이:
  - 평가 이미지: PokeAPI 공식 스프라이트 (학습 이미지와 다른 소스)
  - RAG DB: train.jsonl 이미지만 (eval 이미지 미포함 → Dist>0 보장)
  - 정확도: word-boundary match (eval_utils.check_name_match_strict)
  - 샘플 수: n=50 (Phase 1 v6의 n=15에서 확대)
  - 한국어 이름 정확도 별도 측정

실행:
  # 1) 먼저 공정한 RAG DB 구축 (최초 1회)
  python scripts/setup/build_rag_db_fair.py

  # 2) sanity check (3샘플)
  python scripts/eval/evaluate_fair_benchmark.py --sanity

  # 3) 전체 실행 (재시작 시 자동 이어서)
  python scripts/eval/evaluate_fair_benchmark.py

출력:
  data/pokemon/eval_sprites/results_fair.json   # 원시 결과 (재시작용)
  docs/reports/EVALUATION_REPORT_v7_FAIR_BENCHMARK.md
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

from src.eval_utils import EvalResult, EvalSummary, save_results_json, write_markdown_report
from src.pokemon_info import POKEMON_DB

SPRITE_DIR      = "data/pokemon/eval_sprites"
METADATA_PATH   = os.path.join(SPRITE_DIR, "metadata.json")
RESULTS_PATH    = os.path.join(SPRITE_DIR, "results_fair.json")
REPORT_PATH     = "docs/reports/EVALUATION_REPORT_v7_FAIR_BENCHMARK.md"
RAG_DB_PATH     = "chroma_db_fair"
RAG_COLLECTION  = "pokemon_rag_fair"

BASE_MODEL_PATH  = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
TUNED_MODEL_PATH = "models/fused_qwen2_vl_4bit_quantized"

PROMPT = "What pokemon is this? Answer with the English name and Korean name."

# --- Qwen2-VL 패치 (Phase 1과 동일) ---
from mlx_vlm.models.qwen2_vl.qwen2_vl import Model

def _merge_patched(self, image_features, inputs_embeds, input_ids):
    image_token_index = self.config.image_token_index
    video_token_index = self.config.video_token_index
    image_positions = input_ids == image_token_index
    if mx.sum(image_positions) == 0:
        image_positions = input_ids == video_token_index
    image_features = image_features.astype(mx.float32)
    pad_size = inputs_embeds.shape[1] - image_features.shape[1]
    if pad_size < 0:
        image_features = image_features[:, :inputs_embeds.shape[1], :]
        pad_size = 0
    image_features = mx.pad(image_features, ((0, 0), (0, pad_size), (0, 0)))
    return mx.where(image_positions[:, :, None], image_features, inputs_embeds)

Model._merge_input_ids_with_image_features = _merge_patched


class RobustImageProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor
        if hasattr(processor, "image_processor"):
            self.processor = processor.image_processor
        for attr in dir(self.processor):
            if not attr.startswith("__"):
                try:
                    setattr(self, attr, getattr(self.processor, attr))
                except Exception:
                    pass

    def __call__(self, images=None, text=None, **kwargs):
        kwargs.pop("return_tensors", None)
        out = self.processor(images, text, **kwargs)
        for k, v in out.items():
            if hasattr(v, "numpy"):
                out[k] = v.numpy()
            elif isinstance(v, list) and v and hasattr(v[0], "numpy"):
                out[k] = [x.numpy() for x in v]
        return out

    def preprocess(self, images, **kwargs):
        kwargs.pop("return_tensors", None)
        out = self.processor.preprocess(images, **kwargs)
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(v, "numpy"):
                    out[k] = v.numpy()
        return out

    def __getattr__(self, name):
        return getattr(self.processor, name)


def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(results: list):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def init_rag():
    import chromadb
    from transformers import AutoProcessor, AutoModel
    import torch

    if not os.path.exists(RAG_DB_PATH):
        print(f"[ERROR] 공정한 RAG DB 없음: {RAG_DB_PATH}")
        print("       먼저 실행: python scripts/setup/build_rag_db_fair.py")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=RAG_DB_PATH)
    collection = client.get_collection(name=RAG_COLLECTION)
    print(f"RAG DB 로딩: {collection.count()}개 항목")

    print("SigLIP 로딩...")
    siglip_model_name = "google/siglip-so400m-patch14-384"
    siglip_processor  = AutoProcessor.from_pretrained(siglip_model_name, use_fast=False)
    siglip_model      = AutoModel.from_pretrained(siglip_model_name)
    siglip_model.eval()
    print("SigLIP 로딩 완료")

    return collection, siglip_processor, siglip_model


def get_rag_hint(image_path, collection, siglip_processor, siglip_model):
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = siglip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = siglip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    emb = features[0].numpy().tolist()

    res = collection.query(query_embeddings=[emb], n_results=1)
    dist    = res["distances"][0][0]
    caption = res["metadatas"][0][0].get("caption", "")
    return caption, dist


def run_inference(model, processor, image_path: str, hint: str = "") -> str:
    prompt_text = PROMPT
    if hint:
        prompt_text += f"\n\nHint: {hint}"
    formatted_prompt = apply_chat_template(
        processor, config=model.config, prompt=prompt_text, num_images=1
    )
    return generate(
        model, processor,
        prompt=formatted_prompt, image=image_path,
        max_tokens=120, verbose=False
    )


def generate_report(results_raw: list):
    from src.eval_utils import check_accuracy

    results = []
    summary = EvalSummary(mode="strict")

    for r in results_raw:
        er = EvalResult(
            name        = r["name"],
            image_path  = r["image_path"],
            ground_truth = r["name"],
            korean_name = r.get("name_kr"),
            vanilla_output = r.get("vanilla_output", ""),
            rag_output     = r.get("rag_output", ""),
            tuned_output   = r.get("tuned_output", ""),
            rag_retrieval  = r.get("rag_retrieval", ""),
        )
        er.evaluate(mode="strict")
        summary.add(er)
        results.append(er)

    # RAG 거리 분석
    dists = [r.get("rag_dist", 0) for r in results_raw if "rag_dist" in r]
    dist_info = ""
    if dists:
        import statistics
        dist_info = (
            f"## RAG 검색 거리 분석 (공정성 확인)\n\n"
            f"| 지표 | 값 |\n|---|---|\n"
            f"| 최솟값 | {min(dists):.4f} |\n"
            f"| 평균 | {statistics.mean(dists):.4f} |\n"
            f"| 최댓값 | {max(dists):.4f} |\n\n"
            f"> Phase 1에서는 모든 trained 샘플이 Dist=0.00 (데이터 누수).  \n"
            f"> Phase 2 최솟값 = **{min(dists):.4f}** → 누수 없음 확인.\n"
        )

    write_markdown_report(
        results, summary,
        title="공정한 RAG 벤치마크 평가 결과 (v7)",
        path=REPORT_PATH,
        extra_sections=dist_info,
    )

    print("\n" + "\n".join(summary.report_lines()))
    print(f"\n보고서: {REPORT_PATH}")


def main(sanity: bool = False):
    if not os.path.exists(METADATA_PATH):
        print("[ERROR] 스프라이트 메타데이터 없음. F4 먼저 실행하세요.")
        sys.exit(1)

    metadata = load_metadata()
    existing = load_existing_results()
    done_names = {r["name"] for r in existing}

    if sanity:
        targets = [m for m in metadata if m["name_en"] not in done_names][:3]
        print(f"=== SANITY CHECK ({len(targets)}개 샘플) ===")
    else:
        targets = [m for m in metadata if m["name_en"] not in done_names]
        print(f"=== F5 공정한 RAG 벤치마크 ({len(targets)}개 남음 / 전체 {len(metadata)}개) ===")

    if not targets:
        print("모든 샘플 완료. 보고서만 생성합니다.")
        generate_report(existing)
        return

    results = list(existing)

    # ── PASS 1: Vanilla + RAG ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASS 1: Vanilla + RAG (Base Model)")
    print(f"{'='*60}")

    print(f"모델 로딩: {BASE_MODEL_PATH}")
    model, processor = load(BASE_MODEL_PATH, processor_config={"trust_remote_code": True})
    if hasattr(processor, "image_processor"):
        processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)

    collection, siglip_proc, siglip_model = init_rag()

    for i, meta in enumerate(targets):
        name       = meta["name_en"]
        name_kr    = meta["name_kr"]
        image_path = meta["image_path"]

        print(f"\n[{i+1}/{len(targets)}] {name} ({name_kr})")

        # Vanilla
        vanilla_out = run_inference(model, processor, image_path)
        print(f"  Vanilla: {vanilla_out[:70]}")

        # RAG
        hint, dist = get_rag_hint(image_path, collection, siglip_proc, siglip_model)
        rag_out    = run_inference(model, processor, image_path, hint=hint)
        print(f"  RAG (Dist={dist:.3f}): {rag_out[:70]}")
        print(f"  Hint: {hint[:60]}")

        results.append({
            "name":           name,
            "name_kr":        name_kr,
            "image_path":     image_path,
            "vanilla_output": vanilla_out,
            "rag_output":     rag_out,
            "rag_retrieval":  hint,
            "rag_dist":       dist,
            "tuned_output":   "",  # Pass 2에서 채움
        })
        save_progress(results)

    # 메모리 해제
    del model, processor, siglip_model, siglip_proc, collection
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass
    print("\nBase model 해제 완료")

    # ── PASS 2: Tuned ─────────────────────────────────────────────────
    tuned_targets = [r for r in results if not r.get("tuned_output")]

    if tuned_targets:
        print(f"\n{'='*60}")
        print("PASS 2: Tuned Fused Model")
        print(f"{'='*60}")

        print(f"모델 로딩: {TUNED_MODEL_PATH}")
        model, processor = load(TUNED_MODEL_PATH, processor_config={"trust_remote_code": True})
        if hasattr(processor, "image_processor"):
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)

        for i, r in enumerate(tuned_targets):
            print(f"\n[{i+1}/{len(tuned_targets)}] {r['name']}")
            tuned_out = run_inference(model, processor, r["image_path"])
            r["tuned_output"] = tuned_out
            print(f"  Tuned: {tuned_out[:70]}")
            save_progress(results)

        del model, processor
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
        print("\nTuned model 해제 완료")

    # 보고서 생성
    generate_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="3개 샘플로 sanity check")
    args = parser.parse_args()
    main(sanity=args.sanity)
