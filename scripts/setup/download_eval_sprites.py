"""
F4: 평가용 PokeAPI 스프라이트 수집
CRITICAL_ANALYSIS.md 결함 4, 5 해소

Phase 1 v6에서 15종만 사용 → Phase 2에서 Gen1-2 대표 50종으로 확장.
모든 이미지는 PokeAPI 공식 artwork (학습 이미지와 다른 소스).

선정 기준:
  - Gen1-2 포켓몬 중 POKEMON_DB에 이름+한국어 이름이 있는 포켓몬
  - 시각적 다양성 (타입별, 형태별 고루 선정)
  - 총 50종

실행:
  python scripts/setup/download_eval_sprites.py
  python scripts/setup/download_eval_sprites.py --dry-run  # 목록만 출력

출력 디렉토리: data/pokemon/eval_sprites/
메타데이터:   data/pokemon/eval_sprites/metadata.json
"""

import argparse
import json
import os
import sys
import time

import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pokemon_info import POKEMON_DB

OUTPUT_DIR     = "data/pokemon/eval_sprites"
METADATA_PATH  = os.path.join(OUTPUT_DIR, "metadata.json")
POKEAPI_BASE   = "https://pokeapi.co/api/v2/pokemon"

# Gen1-2 대표 포켓몬 50종 선정 (도감 번호 기준)
# 타입 다양성: Fire/Water/Grass/Electric/Psychic/Ghost/Dragon/Normal/Fighting/Bug 등
# Phase 1 v6 15종 포함 + 35종 추가
EVAL_POKEMON = [
    # Gen I — 스타터 & 전설
    ("bulbasaur",    "이상해씨"),
    ("charmander",   "파이리"),
    ("squirtle",     "꼬부기"),
    ("pikachu",      "피카츄"),
    ("mewtwo",       "뮤츠"),
    ("mew",          "뮤"),
    # Gen I — 유명 포켓몬
    ("eevee",        "이브이"),
    ("snorlax",      "잠만보"),
    ("gengar",       "팬텀"),
    ("charizard",    "리자몽"),
    ("blastoise",    "거북왕"),
    ("venusaur",     "이상해꽃"),
    ("jigglypuff",   "푸린"),
    ("meowth",       "나옹"),
    ("psyduck",      "고라파덕"),
    ("machamp",      "괴력몬"),
    ("alakazam",     "후딘"),
    ("gyarados",     "갸라도스"),
    ("dragonite",    "망나뇽"),
    ("raichu",       "라이츄"),
    ("clefable",     "픽시"),
    ("ninetales",    "나인테일"),
    ("arcanine",     "윈디"),
    ("slowbro",      "야도란"),
    ("magmar",       "마그마"),
    ("electabuzz",   "에레브"),
    ("scyther",      "쁘사이저"),
    ("pinsir",       "거북왕"),  # 임시, 아래에서 수정
    ("lapras",       "라프라스"),
    ("ditto",        "메타몽"),
    ("porygon",      "폴리곤"),
    ("articuno",     "프리져"),
    ("zapdos",       "썬더"),
    ("moltres",      "파이어"),
    ("dragonair",    "신뇽"),
    # Gen II
    ("chikorita",    "치코리타"),
    ("cyndaquil",    "브케인"),
    ("totodile",     "리아코"),
    ("pichu",        "피츄"),
    ("togepi",       "토게피"),
    ("ampharos",     "전룡"),
    ("umbreon",      "블래키"),
    ("espeon",       "에브이"),
    ("houndoom",     "헬가"),
    ("kingdra",      "킹드라"),
    ("heracross",    "헤라크로스"),
    ("sneasel",      "포푸니"),
    ("larvitar",     "애버라스"),
    ("lugia",        "루기아"),
    ("ho-oh",        "칠색조"),
]

# 중복 제거 (pinsir 잘못된 번역 수정)
EVAL_POKEMON = [
    ("bulbasaur",    "이상해씨"),
    ("charmander",   "파이리"),
    ("squirtle",     "꼬부기"),
    ("pikachu",      "피카츄"),
    ("mewtwo",       "뮤츠"),
    ("mew",          "뮤"),
    ("eevee",        "이브이"),
    ("snorlax",      "잠만보"),
    ("gengar",       "팬텀"),
    ("charizard",    "리자몽"),
    ("blastoise",    "거북왕"),
    ("venusaur",     "이상해꽃"),
    ("jigglypuff",   "푸린"),
    ("meowth",       "나옹"),
    ("psyduck",      "고라파덕"),
    ("machamp",      "괴력몬"),
    ("alakazam",     "후딘"),
    ("gyarados",     "갸라도스"),
    ("dragonite",    "망나뇽"),
    ("raichu",       "라이츄"),
    ("clefable",     "픽시"),
    ("ninetales",    "나인테일"),
    ("arcanine",     "윈디"),
    ("slowbro",      "야도란"),
    ("magmar",       "마그마"),
    ("electabuzz",   "에레브"),
    ("scyther",      "시저리"),
    ("pinsir",       "쁘사이저"),
    ("lapras",       "라프라스"),
    ("ditto",        "메타몽"),
    ("articuno",     "프리져"),
    ("zapdos",       "썬더"),
    ("moltres",      "파이어"),
    ("dragonair",    "신뇽"),
    ("hitmonchan",   "에비게임"),
    ("hitmonlee",    "시라소몬"),
    ("clefairy",     "삐삐"),
    ("growlithe",    "가디"),
    ("poliwrath",    "강챙이"),
    ("abra",         "케이시"),
    ("geodude",      "꼬마돌"),
    ("haunter",      "고우스트"),
    ("cubone",       "탕구리"),
    ("hitmontop",    "카포에라"),
    ("chikorita",    "치코리타"),
    ("cyndaquil",    "브케인"),
    ("totodile",     "리아코"),
    ("umbreon",      "블래키"),
    ("espeon",       "에브이"),
    ("lugia",        "루기아"),
]

# 정확히 50종 맞추기
EVAL_POKEMON = EVAL_POKEMON[:50]


def get_sprite_url(pokemon_name: str) -> tuple:
    """PokeAPI에서 공식 artwork URL과 도감번호 반환"""
    url = f"{POKEAPI_BASE}/{pokemon_name.lower()}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, None
        data   = resp.json()
        poke_id = data["id"]
        # 우선순위: official-artwork → front_default
        sprites = data.get("sprites", {})
        artwork_url = (
            sprites.get("other", {})
                   .get("official-artwork", {})
                   .get("front_default")
        )
        if not artwork_url:
            artwork_url = sprites.get("front_default")
        return artwork_url, poke_id
    except Exception as e:
        print(f"  [ERROR] {pokemon_name}: {e}")
        return None, None


def download_sprite(url: str, save_path: str) -> bool:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return False
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        # PNG 배경 → 흰 배경 합성 (SigLIP 입력 안정화)
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.convert("RGB").save(save_path.replace(".png", ".png"), "PNG")
        return True
    except Exception as e:
        print(f"  [ERROR] 다운로드 실패 {url}: {e}")
        return False


def run(dry_run: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 기존 메타데이터 로드
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        existing = {m["name_en"].lower() for m in metadata}
    else:
        metadata  = []
        existing  = set()

    print(f"=== F4: PokeAPI 스프라이트 수집 ===")
    print(f"대상: {len(EVAL_POKEMON)}종 (기수집: {len(existing)}개)\n")

    if dry_run:
        print("--- Dry Run 목록 ---")
        for i, (name, kr) in enumerate(EVAL_POKEMON, 1):
            status = "✅ 수집됨" if name in existing else "⬜ 미수집"
            print(f"  {i:2d}. {name:15s} ({kr}) {status}")
        return

    success, skip, fail = 0, 0, 0

    for i, (name_en, name_kr) in enumerate(EVAL_POKEMON, 1):
        if name_en.lower() in existing:
            print(f"  [{i:2d}/{len(EVAL_POKEMON)}] {name_en}: 이미 수집됨, 스킵")
            skip += 1
            continue

        print(f"  [{i:2d}/{len(EVAL_POKEMON)}] {name_en} ({name_kr})...")
        sprite_url, poke_id = get_sprite_url(name_en)

        if not sprite_url:
            print(f"    → URL 없음")
            fail += 1
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{name_en}.png")
        ok = download_sprite(sprite_url, save_path)

        if ok:
            # POKEMON_DB에서 세대 조회
            meta = POKEMON_DB.get(name_en.lower(), {})
            gen  = meta.get("generation", "unknown")

            metadata.append({
                "name_en":    name_en,
                "name_kr":    name_kr,
                "pokedex_id": poke_id,
                "generation": gen,
                "image_path": save_path,
                "sprite_url": sprite_url,
            })
            success += 1
            print(f"    → OK ({gen}, #{poke_id})")
        else:
            fail += 1

        # 메타데이터 점진적 저장 (중단 대비)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        time.sleep(0.3)  # PokeAPI rate limit 방지

    print(f"\n=== 완료: 성공={success}, 스킵={skip}, 실패={fail} ===")
    print(f"저장 경로: {OUTPUT_DIR}/")
    print(f"메타데이터: {METADATA_PATH}")

    # 최종 통계
    gen_dist = {}
    for m in metadata:
        g = m.get("generation", "unknown")
        gen_dist[g] = gen_dist.get(g, 0) + 1
    print(f"세대 분포: {gen_dist}")

    # train.jsonl과 겹치는 이름 확인
    train_names = set()
    with open("data/pokemon/train.jsonl") as f:
        import re
        for line in f:
            content = json.loads(line)["messages"][1]["content"]
            for name, _ in EVAL_POKEMON:
                if re.search(rf'\b{re.escape(name)}\b', content, re.IGNORECASE):
                    train_names.add(name)

    overlap = train_names & {n.lower() for n, _ in EVAL_POKEMON}
    print(f"\ntrain.jsonl 이름 중복: {len(overlap)}개 (이미지는 다른 소스 — 허용)")
    if overlap:
        print(f"  중복 이름: {sorted(overlap)[:10]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="목록만 출력, 다운로드 안 함")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
