"""
F7: OOD 강화 테스트 (Gen3-9, n=30)
CRITICAL_ANALYSIS.md 결함 4 해소

Phase 1 v5와의 차이:
  - 샘플 수: 6 → 30종
  - 포켓몬 분류: 진화계열(10) / 시각적 유사(10) / 완전이질(10)
  - RAG DB: chroma_db_fair (학습 이미지 기반, eval 이미지 미포함)
  - 정확도: eval_utils word-boundary match
  - 이미지: PokeAPI 공식 스프라이트

목적:
  1. RAG의 진화계열 매칭 효과 정량화 (Phase 1에서 흥미로운 패턴 발견)
  2. Vanilla vs RAG OOD 성능 비교
  3. RAG DB가 Gen1-2만 있을 때 Gen3-9을 얼마나 잘 처리하는가

실행:
  python scripts/eval/evaluate_ood_enhanced.py
  python scripts/eval/evaluate_ood_enhanced.py --download-only  # 이미지만 다운로드

출력:
  data/pokemon/eval_ood/          # OOD 스프라이트 이미지
  docs/reports/EVALUATION_REPORT_v9_OOD_ENHANCED.md
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

from src.eval_utils import check_name_match_strict

OOD_DIR      = "data/pokemon/eval_ood"
RESULTS_PATH = os.path.join(OOD_DIR, "results_ood.json")
REPORT_PATH  = "docs/reports/EVALUATION_REPORT_v9_OOD_ENHANCED.md"
RAG_DB_PATH  = "chroma_db_fair"
RAG_COLLECTION = "pokemon_rag_fair"
BASE_MODEL   = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon"
PROMPT       = "What pokemon is this? Answer with the English name and Korean name."

# Gen3+ 포켓몬 30종 (도감 이름, 한국어, 분류, 관련 Gen1-2 포켓몬)
OOD_POKEMON = [
    # 진화계열 (Gen1-2 포켓몬의 진화형 또는 전진화)
    # RAG가 시각적으로 유사한 Gen1-2를 찾아낼 것으로 기대
    ("electivire",  "에레키블",   "evolution", "electabuzz (에레브)"),
    ("magmortar",   "마그모라타", "evolution", "magmar (마그마)"),
    ("lickilicky",  "내룸벨트",   "evolution", "lickitung (내루미)"),
    ("togekiss",    "토게키스",   "evolution", "togetic/togepi (토게틱/토게피)"),
    ("leafeon",     "리피아",     "evolution", "eevee (이브이)"),
    ("glaceon",     "글레이시아", "evolution", "eevee (이브이)"),
    ("sylveon",     "님피아",     "evolution", "eevee (이브이)"),
    ("munchlax",    "먹고자",     "evolution", "snorlax (잠만보)"),
    ("mime-jr",     "흉내내기",   "evolution", "mr-mime (마임맨)"),
    ("bonsly",      "꼬지모",     "evolution", "sudowoodo (나무킹)"),

    # 시각적 유사 (Gen1-2 포켓몬과 외형 유사)
    ("ralts",       "랄토스",     "visual",    "gardevoir 계열"),
    ("gardevoir",   "가디안",     "visual",    "humanoid 실루엣"),
    ("metagross",   "메타그로스", "visual",    "psychic 계열"),
    ("absol",       "앱솔",       "visual",    "흰색 포켓몬"),
    ("lucario",     "루카리오",   "visual",    "riolu 계열"),
    ("garchomp",    "한카리아스", "visual",    "dragon 계열"),
    ("infernape",   "infernape",  "visual",    "chimchar 계열"),
    ("empoleon",    "엠페르트",   "visual",    "piplup 계열"),
    ("torterra",    "토대부기",   "visual",    "turtwig 계열"),
    ("roserade",    "로즈레이드", "visual",    "budew/roselia 계열"),

    # 완전 이질적 (Gen1-2와 외형 매우 다름 — RAG 매칭 어려울 것)
    ("zekrom",      "제크롬",     "distinct",  "없음"),
    ("reshiram",    "레시라무",   "distinct",  "없음"),
    ("kyurem",      "큐레무",     "distinct",  "없음"),
    ("xerneas",     "제르네아스", "distinct",  "없음"),
    ("yveltal",     "이벨타르",   "distinct",  "없음"),
    ("solgaleo",    "솔가레오",   "distinct",  "없음"),
    ("lunala",      "루나아라",   "distinct",  "없음"),
    ("zacian",      "자시안",     "distinct",  "없음"),
    ("zamazenta",   "자마젠타",   "distinct",  "없음"),
    ("eternatus",   "무한다이노", "distinct",  "없음"),
]


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


def download_sprite(name: str, save_dir: str) -> tuple:
    url = f"{POKEAPI_BASE}/{name.lower()}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        poke_id = data["id"]
        artwork_url = (
            data.get("sprites", {})
                .get("other", {})
                .get("official-artwork", {})
                .get("front_default")
        )
        if not artwork_url:
            artwork_url = data.get("sprites", {}).get("front_default")
        if not artwork_url:
            return None, None

        img_resp = requests.get(artwork_url, timeout=15)
        img = Image.open(BytesIO(img_resp.content)).convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        save_path = os.path.join(save_dir, f"{name}.png")
        bg.convert("RGB").save(save_path, "PNG")
        return save_path, poke_id
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None, None


def download_all():
    os.makedirs(OOD_DIR, exist_ok=True)
    existing = {os.path.splitext(f)[0] for f in os.listdir(OOD_DIR) if f.endswith(".png")}

    print(f"=== OOD 이미지 다운로드 ({len(OOD_POKEMON)}종) ===")
    ok, skip, fail = 0, 0, 0
    for name_en, name_kr, category, related in OOD_POKEMON:
        if name_en in existing:
            skip += 1
            continue
        print(f"  [{category}] {name_en} ({name_kr})...")
        path, poke_id = download_sprite(name_en, OOD_DIR)
        if path:
            print(f"    → OK (#{poke_id})")
            ok += 1
        else:
            print(f"    → 실패")
            fail += 1
        time.sleep(0.3)

    print(f"완료: 성공={ok}, 스킵={skip}, 실패={fail}")
    return fail == 0


def init_rag():
    import chromadb
    from transformers import AutoProcessor, AutoModel
    import torch

    client     = chromadb.PersistentClient(path=RAG_DB_PATH)
    collection = client.get_collection(name=RAG_COLLECTION)

    siglip_proc  = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", use_fast=False)
    siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model.eval()
    return collection, siglip_proc, siglip_model


def get_rag_hint(image_path, collection, siglip_proc, siglip_model):
    from PIL import Image as PILImage
    import torch

    img    = PILImage.open(image_path).convert("RGB")
    inputs = siglip_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feat = siglip_model.get_image_features(**inputs)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    emb  = feat[0].numpy().tolist()

    res  = collection.query(query_embeddings=[emb], n_results=1)
    dist = res["distances"][0][0]
    cap  = res["metadatas"][0][0].get("caption", "")
    return cap, dist


def run_inference(model, processor, image_path, hint=""):
    prompt_text = PROMPT + (f"\n\nHint: {hint}" if hint else "")
    formatted   = apply_chat_template(processor, config=model.config, prompt=prompt_text, num_images=1)
    return generate(model, processor, prompt=formatted, image=image_path, max_tokens=100, verbose=False)


def main(download_only=False):
    os.makedirs(OOD_DIR, exist_ok=True)

    # 이미지 다운로드
    ok = download_all()
    if download_only:
        return

    # 기존 결과 로드
    existing = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            for r in json.load(f):
                existing[r["name"]] = r

    targets = [(n, k, c, rel) for n, k, c, rel in OOD_POKEMON if n not in existing]
    print(f"\n=== F7 OOD 강화 테스트 ({len(targets)}개 남음 / 30개 전체) ===")

    results = list(existing.values())

    if targets:
        print(f"\nBase Model + RAG 로딩...")
        model, processor = load(BASE_MODEL, processor_config={"trust_remote_code": True})
        if hasattr(processor, "image_processor"):
            processor.image_processor = RobustImageProcessorWrapper(processor.image_processor)
        collection, siglip_proc, siglip_model = init_rag()

        for i, (name_en, name_kr, category, related) in enumerate(targets):
            image_path = os.path.join(OOD_DIR, f"{name_en}.png")
            if not os.path.exists(image_path):
                print(f"  [SKIP] {name_en}: 이미지 없음")
                continue

            print(f"\n[{i+1}/{len(targets)}] {name_en} ({name_kr}) [{category}]")

            vanilla_out = run_inference(model, processor, image_path)
            hint, dist  = get_rag_hint(image_path, collection, siglip_proc, siglip_model)
            rag_out     = run_inference(model, processor, image_path, hint=hint)

            v_ok = check_name_match_strict(vanilla_out, name_en)
            r_ok = check_name_match_strict(rag_out, name_en)

            print(f"  Vanilla: {vanilla_out[:70]} → {'✅' if v_ok else '❌'}")
            print(f"  RAG (Dist={dist:.3f}): {rag_out[:70]} → {'✅' if r_ok else '❌'}")
            print(f"  Hint: {hint[:60]}")

            results.append({
                "name": name_en, "name_kr": name_kr,
                "category": category, "related": related,
                "image_path": image_path,
                "vanilla_output": vanilla_out, "rag_output": rag_out,
                "rag_hint": hint, "rag_dist": dist,
                "vanilla_correct": v_ok, "rag_correct": r_ok,
            })
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        del model, processor, siglip_model, siglip_proc, collection
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

    # 집계
    cats = ["evolution", "visual", "distinct"]
    print(f"\n{'='*55}")
    print("OOD 강화 테스트 결과 요약")
    print(f"{'='*55}")

    for cat in cats:
        sub = [r for r in results if r.get("category") == cat]
        if not sub:
            continue
        v_ok = sum(1 for r in sub if r["vanilla_correct"])
        r_ok = sum(1 for r in sub if r["rag_correct"])
        print(f"  [{cat:9s}] n={len(sub):2d} | Vanilla {v_ok}/{len(sub)} | RAG {r_ok}/{len(sub)}")

    all_v = sum(1 for r in results if r["vanilla_correct"])
    all_r = sum(1 for r in results if r["rag_correct"])
    n = len(results)
    print(f"  [전체      ] n={n:2d} | Vanilla {all_v}/{n} ({all_v/n*100:.1f}%) | RAG {all_r}/{n} ({all_r/n*100:.1f}%)")

    # 보고서
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# OOD 강화 테스트 결과 (v9)\n\n")
        f.write("**생성일**: 2026-05-15  \n")
        f.write("**관련 결함**: CRITICAL_ANALYSIS.md 결함 4\n\n---\n\n")
        f.write("## 실험 설계\n\n")
        f.write("Gen3-9 미학습 포켓몬 30종을 3가지 카테고리로 분류하여 RAG의 한계를 정량화.\n\n")
        f.write("| 카테고리 | 설명 | 종 수 |\n|---|---|---|\n")
        f.write("| evolution | Gen1-2 포켓몬의 진화형/전진화 | 10 |\n")
        f.write("| visual | 시각적으로 Gen1-2와 유사 | 10 |\n")
        f.write("| distinct | 외형이 완전히 이질적 | 10 |\n\n")
        f.write("## 카테고리별 정확도\n\n")
        f.write("| 카테고리 | Vanilla | RAG |\n|---|---|---|\n")
        for cat in cats:
            sub = [r for r in results if r.get("category") == cat]
            if not sub:
                continue
            v_ok = sum(1 for r in sub if r["vanilla_correct"])
            r_ok = sum(1 for r in sub if r["rag_correct"])
            f.write(f"| {cat} | {v_ok}/{len(sub)} ({v_ok/len(sub)*100:.1f}%) | {r_ok}/{len(sub)} ({r_ok/len(sub)*100:.1f}%) |\n")
        f.write(f"| **전체** | **{all_v}/{n} ({all_v/n*100:.1f}%)** | **{all_r}/{n} ({all_r/n*100:.1f}%)** |\n\n")
        f.write("## 상세 결과\n\n")
        f.write("| 포켓몬 | 분류 | 관련 Gen1-2 | Vanilla | RAG | Dist |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            v_icon = "✅" if r["vanilla_correct"] else "❌"
            r_icon = "✅" if r["rag_correct"]     else "❌"
            f.write(
                f"| **{r['name']}** ({r['name_kr']}) | {r.get('category','?')} | {r.get('related','?')} "
                f"| {v_icon} {r['vanilla_output'][:40]}... "
                f"| {r_icon} {r['rag_output'][:40]}... "
                f"| {r.get('rag_dist',0):.3f} |\n"
            )

    print(f"\n보고서: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    main(download_only=args.download_only)
