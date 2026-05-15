"""
F3: 공정한 RAG DB 구축
CRITICAL_ANALYSIS.md 결함 1 해소

Phase 1 문제:
  - RAG DB = 학습 이미지 전체 인덱싱
  - 평가 시 동일 이미지 쿼리 → Dist:0.00 (데이터 누수)

Phase 2 해결:
  - 평가용 이미지(PokeAPI 스프라이트)는 DB에서 완전 제외
  - DB 구성: train.jsonl 이미지만 (세대 확인된 197개 + 세대불명 323개)
  - 구축 후 검증: 평가 이미지와 DB 최소 거리 확인

실행:
  python scripts/setup/build_rag_db_fair.py
  python scripts/setup/build_rag_db_fair.py --verify  # 검증만
  python scripts/setup/build_rag_db_fair.py --reset   # DB 재생성

출력 DB 경로: chroma_db_fair/
검증 로그: docs/logs/rag_db_fair_verification.txt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chromadb
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np

DB_PATH        = "chroma_db_fair"
COLLECTION_NAME = "pokemon_rag_fair"
TRAIN_JSONL    = "data/pokemon/train.jsonl"
EVAL_SPRITE_DIR = "data/pokemon/eval_sprites"  # F4에서 생성
LOG_PATH       = "docs/logs/rag_db_fair_verification.txt"

SIGLIP_MODEL   = "google/siglip-so400m-patch14-384"


def load_siglip():
    print(f"SigLIP 로딩: {SIGLIP_MODEL}")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, use_fast=False)
    model     = AutoModel.from_pretrained(SIGLIP_MODEL)
    model.eval()
    print("SigLIP 로딩 완료")
    return processor, model


def get_image_embedding(image_path: str, processor, model) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].numpy().tolist()


def load_train_entries():
    entries = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def build_db(reset: bool = False):
    """train.jsonl의 이미지를 chromadb_fair에 인덱싱"""
    if reset and os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        print(f"기존 DB 삭제: {DB_PATH}")

    client     = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_ids = set(collection.get()["ids"])
    print(f"기존 DB 항목: {len(existing_ids)}개")

    entries = load_train_entries()
    processor, model = load_siglip()

    indexed, skipped, failed = 0, 0, 0
    for i, entry in enumerate(entries):
        image_path = entry["images"][0]
        caption    = entry["messages"][1]["content"]
        doc_id     = os.path.basename(image_path)

        if doc_id in existing_ids:
            skipped += 1
            continue

        if not os.path.exists(image_path):
            print(f"  [SKIP] 파일 없음: {image_path}")
            failed += 1
            continue

        try:
            emb = get_image_embedding(image_path, processor, model)
            collection.add(
                ids=[doc_id],
                embeddings=[emb],
                metadatas=[{"path": image_path, "caption": caption}],
            )
            indexed += 1
            if (indexed % 50) == 0:
                print(f"  인덱싱 중... {indexed}/{len(entries) - skipped}")
        except Exception as e:
            print(f"  [ERROR] {image_path}: {e}")
            failed += 1

    total = collection.count()
    print(f"\nDB 구축 완료: {total}개 항목 (신규:{indexed}, 스킵:{skipped}, 실패:{failed})")
    return total


def verify_no_leakage(processor, model):
    """
    평가 스프라이트 이미지와 DB 이미지 간 최소 거리 확인.
    Dist=0.00 이면 동일 이미지가 DB에 존재 → 누수.
    """
    if not os.path.exists(EVAL_SPRITE_DIR):
        print(f"[SKIP] 평가 스프라이트 디렉토리 없음: {EVAL_SPRITE_DIR}")
        print("       F4(download_eval_sprites.py) 먼저 실행하세요")
        return None

    client     = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    sprite_files = sorted([
        os.path.join(EVAL_SPRITE_DIR, f)
        for f in os.listdir(EVAL_SPRITE_DIR)
        if f.endswith(".png")
    ])

    if not sprite_files:
        print("[SKIP] 스프라이트 이미지 없음")
        return None

    print(f"\n누수 검증: 스프라이트 {len(sprite_files)}개 vs DB {collection.count()}개")

    results = []
    min_dist_overall = 1.0
    leakage_count    = 0

    for sprite_path in sprite_files:
        name = os.path.splitext(os.path.basename(sprite_path))[0]
        try:
            emb = get_image_embedding(sprite_path, processor, model)
            res = collection.query(query_embeddings=[emb], n_results=1)
            dist = res["distances"][0][0]
            retrieved_caption = res["metadatas"][0][0].get("caption", "")[:50]

            results.append({
                "name":    name,
                "dist":    dist,
                "caption": retrieved_caption,
                "leakage": dist < 0.01,
            })

            if dist < 0.01:
                leakage_count += 1
                print(f"  [누수!] {name}: Dist={dist:.4f} → {retrieved_caption}")
            else:
                print(f"  [OK]   {name}: Dist={dist:.4f} → {retrieved_caption}")

            min_dist_overall = min(min_dist_overall, dist)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # 로그 저장
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("RAG DB 공정성 검증 로그\n")
        f.write("=" * 50 + "\n")
        f.write(f"검증일: 2026-05-15\n")
        f.write(f"DB 경로: {DB_PATH}\n")
        f.write(f"DB 항목 수: {collection.count()}\n")
        f.write(f"스프라이트 수: {len(sprite_files)}\n")
        f.write(f"최소 거리: {min_dist_overall:.4f}\n")
        f.write(f"누수 의심 (Dist<0.01): {leakage_count}개\n\n")
        f.write("상세 결과:\n")
        for r in results:
            leak_tag = " [누수!]" if r["leakage"] else ""
            f.write(f"  {r['name']}: Dist={r['dist']:.4f}{leak_tag} → {r['caption']}\n")

    print(f"\n검증 결과: 누수={leakage_count}, 최소거리={min_dist_overall:.4f}")
    print(f"로그 저장: {LOG_PATH}")

    if leakage_count == 0:
        print("✅ 데이터 누수 없음 — 공정한 RAG 평가 가능")
    else:
        print(f"❌ 누수 {leakage_count}개 — 해당 스프라이트를 평가셋에서 제외하거나 DB에서 제거 필요")

    return results


def main():
    parser = argparse.ArgumentParser(description="공정한 RAG DB 구축")
    parser.add_argument("--reset",  action="store_true", help="DB 초기화 후 재생성")
    parser.add_argument("--verify", action="store_true", help="검증만 실행 (DB 구축 스킵)")
    args = parser.parse_args()

    if not args.verify:
        build_db(reset=args.reset)

    # 평가 스프라이트가 있으면 누수 검증
    if os.path.exists(EVAL_SPRITE_DIR):
        processor, model = load_siglip()
        verify_no_leakage(processor, model)
    else:
        print(f"\n[INFO] 누수 검증은 F4(download_eval_sprites.py) 실행 후 가능합니다")
        print(f"       이후: python scripts/setup/build_rag_db_fair.py --verify")


if __name__ == "__main__":
    main()
