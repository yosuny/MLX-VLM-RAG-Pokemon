"""
F6: Tuned 모델 과적합 vs 학습 미흡 진단
CRITICAL_ANALYSIS.md 결함 6 해소

가설:
  H1 (과적합): 학습 이미지 정확도 >> 다른 이미지 정확도
  H2 (학습 미흡): 학습 이미지 정확도 ≈ 다른 이미지 (둘 다 낮음)

방법:
  - 동일 포켓몬에 대해 (A) train.jsonl 원본 이미지, (B) PokeAPI 스프라이트 비교
  - Tuned 모델로만 두 소스 추론
  - 영어 이름 정확도 차이로 가설 판정

실행:
  python scripts/eval/evaluate_tuned_diagnosis.py
  python scripts/eval/evaluate_tuned_diagnosis.py --sanity  # 5개 샘플

출력:
  docs/reports/EVALUATION_REPORT_v8_TUNED_DIAGNOSIS.md
"""

import argparse
import gc
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

from src.eval_utils import check_name_match_strict
from src.pokemon_info import POKEMON_DB

SPRITE_META_PATH = "data/pokemon/eval_sprites/metadata.json"
TRAIN_JSONL      = "data/pokemon/train.jsonl"
TUNED_MODEL_PATH = "models/fused_qwen2_vl_4bit_quantized"
RESULTS_PATH     = "data/pokemon/eval_sprites/results_diagnosis.json"
REPORT_PATH      = "docs/reports/EVALUATION_REPORT_v8_TUNED_DIAGNOSIS.md"

PROMPT = "What pokemon is this? Answer with the English name and Korean name."

# --- Qwen2-VL 패치 ---
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


def find_train_image(pokemon_name: str) -> tuple:
    """
    train.jsonl에서 해당 포켓몬 이름이 포함된 이미지 경로와 캡션 반환.
    이름이 GEN 태그와 함께 명시적으로 있는 항목만 사용.
    """
    pattern = rf'\bThis is {re.escape(pokemon_name)}\b'
    with open(TRAIN_JSONL) as f:
        for line in f:
            entry = json.loads(line.strip())
            caption = entry["messages"][1]["content"]
            if re.search(pattern, caption, re.IGNORECASE):
                image_path = entry["images"][0]
                if os.path.exists(image_path):
                    return image_path, caption
    return None, None


def run_inference(model, processor, image_path: str) -> str:
    formatted = apply_chat_template(
        processor, config=model.config, prompt=PROMPT, num_images=1
    )
    return generate(
        model, processor,
        prompt=formatted, image=image_path,
        max_tokens=100, verbose=False
    )


def build_pairs(n_max: int = None):
    """학습 이미지와 스프라이트 쌍 구성"""
    with open(SPRITE_META_PATH, encoding="utf-8") as f:
        sprite_meta = json.load(f)

    pairs = []
    for meta in sprite_meta:
        name_en = meta["name_en"]
        sprite_path = meta["image_path"]

        train_path, train_caption = find_train_image(name_en)
        if train_path:
            pairs.append({
                "name":          name_en,
                "name_kr":       meta["name_kr"],
                "train_image":   train_path,
                "train_caption": train_caption,
                "sprite_image":  sprite_path,
            })

        if n_max and len(pairs) >= n_max:
            break

    return pairs


def main(sanity: bool = False):
    n_max = 5 if sanity else None
    label = "SANITY CHECK (5개)" if sanity else "전체"

    print(f"=== F6 Tuned 모델 진단 — {label} ===")

    # 기존 결과 로드 (재시작 지원)
    if os.path.exists(RESULTS_PATH) and not sanity:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        done = {r["name"] for r in results}
    else:
        results = []
        done = set()

    pairs = build_pairs(n_max)
    pairs = [p for p in pairs if p["name"] not in done]

    print(f"대상 쌍: {len(pairs)}개 (train 이미지 + 스프라이트)\n")

    if not pairs:
        print("모든 쌍 처리 완료. 보고서만 생성합니다.")
    else:
        print(f"모델 로딩: {TUNED_MODEL_PATH}")
        model, processor = load(TUNED_MODEL_PATH, processor_config={"trust_remote_code": True})
        if hasattr(processor, "image_processor"):
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)

        for i, pair in enumerate(pairs):
            name = pair["name"]
            print(f"\n[{i+1}/{len(pairs)}] {name} ({pair['name_kr']})")

            # A) train 이미지 (학습에서 본 이미지)
            out_train = run_inference(model, processor, pair["train_image"])
            en_train  = check_name_match_strict(out_train, name)
            print(f"  [Train 이미지] {out_train[:70]}")
            print(f"  → 정답: {'✅' if en_train else '❌'}")

            # B) PokeAPI 스프라이트 (학습에서 못 본 이미지)
            out_sprite = run_inference(model, processor, pair["sprite_image"])
            en_sprite  = check_name_match_strict(out_sprite, name)
            print(f"  [Sprite 이미지] {out_sprite[:70]}")
            print(f"  → 정답: {'✅' if en_sprite else '❌'}")

            results.append({
                "name":           name,
                "name_kr":        pair["name_kr"],
                "train_image":    pair["train_image"],
                "sprite_image":   pair["sprite_image"],
                "train_output":   out_train,
                "sprite_output":  out_sprite,
                "train_correct":  en_train,
                "sprite_correct": en_sprite,
            })

            if not sanity:
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        del model, processor
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

    # 집계
    train_correct  = sum(1 for r in results if r["train_correct"])
    sprite_correct = sum(1 for r in results if r["sprite_correct"])
    total = len(results)

    print(f"\n{'='*50}")
    print(f"진단 결과 (n={total})")
    print(f"  학습 이미지 정확도:  {train_correct}/{total} ({train_correct/total*100:.1f}%)")
    print(f"  스프라이트 정확도:   {sprite_correct}/{total} ({sprite_correct/total*100:.1f}%)")
    gap = train_correct/total - sprite_correct/total if total else 0
    print(f"  정확도 차이 (A-B):  {gap*100:+.1f}%p")

    if gap > 0.20:
        verdict = "H1 (과적합): 학습 이미지에서 현저히 잘 맞힘"
    elif abs(gap) <= 0.10:
        verdict = "H2 (학습 미흡): 두 소스 모두 유사하게 낮은 정확도"
    else:
        verdict = f"중간: {gap*100:+.1f}%p 차이 — 경미한 과적합 가능성"
    print(f"  판정: {verdict}")

    # 보고서 생성
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Tuned 모델 과적합 vs 학습 미흡 진단 (v8)\n\n")
        f.write("**생성일**: 2026-05-15  \n")
        f.write("**관련 결함**: CRITICAL_ANALYSIS.md 결함 6\n\n")
        f.write("---\n\n")
        f.write("## 실험 설계\n\n")
        f.write("| 구분 | 소스 | 설명 |\n|---|---|---|\n")
        f.write("| A (Train 이미지) | train.jsonl | Tuned 모델이 학습 시 본 이미지 |\n")
        f.write("| B (Sprite 이미지) | PokeAPI | 학습에서 보지 않은 공식 artwork |\n\n")
        f.write("가설: A >> B면 과적합(H1), A ≈ B면 학습 미흡(H2)\n\n")
        f.write("## 결과 요약\n\n")
        f.write(f"| | 정확도 |\n|---|---|\n")
        f.write(f"| A 학습 이미지 (n={total}) | {train_correct}/{total} ({train_correct/total*100:.1f}%) |\n")
        f.write(f"| B 스프라이트  (n={total}) | {sprite_correct}/{total} ({sprite_correct/total*100:.1f}%) |\n")
        f.write(f"| 차이 (A−B) | {gap*100:+.1f}%p |\n\n")
        f.write(f"## 판정\n\n**{verdict}**\n\n")
        f.write("## 상세 결과\n\n")
        f.write("| 포켓몬 | Train 출력 | 정답 | Sprite 출력 | 정답 |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            ta = "✅" if r["train_correct"]  else "❌"
            sa = "✅" if r["sprite_correct"] else "❌"
            f.write(
                f"| **{r['name']}** "
                f"| {r['train_output'][:50]}... | {ta} "
                f"| {r['sprite_output'][:50]}... | {sa} |\n"
            )

    print(f"\n보고서: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    main(sanity=args.sanity)
