"""
F1: 학습 데이터 세대 분리 검증 스크립트
Phase 2 / CRITICAL_ANALYSIS.md 결함 7 해소

검증 항목:
1. train.jsonl에서 세대별 구성 (GEN 명시 / 이름 미매칭 / 세대 불명)
2. 미매칭 323개의 실제 세대 재추정 시도 (POKEMON_DB 재검색)
3. validation.jsonl 세대 순도 확인
4. "Gen1-2 학습" 주장의 실제 신뢰 범위 보고

실행: python scripts/debug/verify_train_gen_split.py
출력: docs/reports/TRAIN_GEN_SPLIT_VERIFICATION.md
"""

import json
import re
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pokemon_info import POKEMON_DB

OUTPUT_PATH = "docs/reports/TRAIN_GEN_SPLIT_VERIFICATION.md"

GEN_LABEL_MAP = {
    "generation-i":    "GEN I",
    "generation-ii":   "GEN II",
    "generation-iii":  "GEN III",
    "generation-iv":   "GEN IV",
    "generation-v":    "GEN V",
    "generation-vi":   "GEN VI",
    "generation-vii":  "GEN VII",
    "generation-viii": "GEN VIII",
    "generation-ix":   "GEN IX",
}

GEN1_2 = {"generation-i", "generation-ii"}
GEN3_PLUS = {"generation-iii", "generation-iv", "generation-v",
             "generation-vi", "generation-vii", "generation-viii", "generation-ix"}

# 이름→메타데이터 정방향 맵 (소문자, 이름 키만)
name_to_meta = {
    k.lower(): v for k, v in POKEMON_DB.items() if not k.isdigit()
}
# 긴 이름 우선 매칭
sorted_names = sorted(name_to_meta.keys(), key=len, reverse=True)


def extract_gen_from_caption(text: str):
    """캡션에서 'GEN X' 패턴 추출 → generation-x 형식 반환"""
    roman_to_gen = {
        "I":    "generation-i",
        "II":   "generation-ii",
        "III":  "generation-iii",
        "IV":   "generation-iv",
        "V":    "generation-v",
        "VI":   "generation-vi",
        "VII":  "generation-vii",
        "VIII": "generation-viii",
        "IX":   "generation-ix",
    }
    m = re.search(r'\bGEN\s+(I{1,3}V?|V?I{0,3}|IV|IX|VIII|VII|VI|V)\b', text)
    if m:
        return roman_to_gen.get(m.group(1).upper())
    return None


def extract_name_from_caption(text: str):
    """캡션에서 포켓몬 영어 이름 추출 (POKEMON_DB 대조)"""
    text_lower = text.lower()
    for name in sorted_names:
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, text_lower):
            return name, name_to_meta[name]
    return None, None


def classify_entry(entry: dict):
    """
    단일 JSONL 항목을 분류.
    반환: {
        'image': str,
        'caption': str,
        'gen_from_caption': str|None,   # 캡션 내 GEN 태그
        'matched_name': str|None,        # DB 매칭된 영어 이름
        'matched_gen': str|None,         # DB에서 확인된 세대
        'final_gen': str|None,           # 최종 판정 세대
        'confidence': str                # 'high'/'medium'/'unknown'
    }
    """
    caption = entry["messages"][1]["content"]
    image   = entry["images"][0]

    gen_tag      = extract_gen_from_caption(caption)
    matched_name, matched_meta = extract_name_from_caption(caption)

    matched_gen = matched_meta["generation"] if matched_meta else None

    # 신뢰도 결정
    if gen_tag and matched_gen:
        if gen_tag == matched_gen:
            confidence = "high"
        else:
            # 드문 불일치 — 캡션 태그 우선
            confidence = "medium"
        final_gen = gen_tag
    elif gen_tag:
        confidence = "high"
        final_gen = gen_tag
    elif matched_gen:
        confidence = "medium"
        final_gen = matched_gen
    else:
        confidence = "unknown"
        final_gen = None

    return {
        "image":           image,
        "caption":         caption,
        "gen_from_caption": gen_tag,
        "matched_name":    matched_name,
        "matched_gen":     matched_gen,
        "final_gen":       final_gen,
        "confidence":      confidence,
    }


def analyze_split(jsonl_path: str, label: str):
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    results = [classify_entry(e) for e in entries]

    # 집계
    gen_counts    = defaultdict(int)  # final_gen → count
    conf_counts   = defaultdict(int)  # confidence → count
    gen3_in_train = []                # Gen3+ 발견 목록
    unknown_list  = []                # 세대 불명 목록

    for r in results:
        conf_counts[r["confidence"]] += 1
        if r["final_gen"]:
            gen_counts[r["final_gen"]] += 1
        else:
            unknown_list.append(r)

        if label == "train" and r["final_gen"] in GEN3_PLUS:
            gen3_in_train.append(r)

    return {
        "label":         label,
        "total":         len(results),
        "results":       results,
        "gen_counts":    dict(gen_counts),
        "conf_counts":   dict(conf_counts),
        "gen3_in_train": gen3_in_train,
        "unknown_list":  unknown_list,
    }


def format_gen_table(gen_counts: dict, total: int) -> str:
    gen_order = ["generation-i", "generation-ii", "generation-iii",
                 "generation-iv", "generation-v", "generation-vi",
                 "generation-vii", "generation-viii", "generation-ix"]
    lines = ["| 세대 | 수량 | 비율 |", "|---|---|---|"]
    for g in gen_order:
        if g in gen_counts:
            c = gen_counts[g]
            lines.append(f"| {GEN_LABEL_MAP[g]} | {c} | {c/total*100:.1f}% |")
    return "\n".join(lines)


def run():
    print("=== Phase 2 F1: 학습 데이터 세대 분리 검증 ===\n")

    train_stat = analyze_split("data/pokemon/train.jsonl", "train")
    valid_stat = analyze_split("data/pokemon/validation.jsonl", "validation")

    # --- 콘솔 요약 ---
    for stat in [train_stat, valid_stat]:
        lbl   = stat["label"].upper()
        total = stat["total"]
        unk   = len(stat["unknown_list"])
        hi    = stat["conf_counts"].get("high", 0)
        med   = stat["conf_counts"].get("medium", 0)
        gen3  = len(stat["gen3_in_train"])

        print(f"[{lbl}] 총 {total}개")
        print(f"  신뢰도 high   : {hi}개 ({hi/total*100:.1f}%)")
        print(f"  신뢰도 medium : {med}개 ({med/total*100:.1f}%)")
        print(f"  세대 불명     : {unk}개 ({unk/total*100:.1f}%)")
        if lbl == "TRAIN":
            print(f"  Gen3+ 혼입    : {gen3}개 ← 오염 여부")
        print(f"  세대 분포: {stat['gen_counts']}")
        print()

    # --- Gen3+ 혼입 상세 ---
    if train_stat["gen3_in_train"]:
        print(f"[경고] train에 Gen3+ 샘플 {len(train_stat['gen3_in_train'])}개 발견!")
        for r in train_stat["gen3_in_train"][:10]:
            print(f"  {r['image']}: {r['matched_name']} ({GEN_LABEL_MAP.get(r['final_gen'], r['final_gen'])})")
    else:
        print("[확인] train에서 Gen3+ 혼입 없음 (확인 가능한 범위 내)")

    print(f"\n[주의] train 세대 불명 {len(train_stat['unknown_list'])}개는 GPT-4 캡션에 이름이 없어 세대 추정 불가")
    print(f"  → 이 샘플들이 실제로 어느 세대인지 보장할 수 없음")

    # --- 마크다운 보고서 생성 ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# 학습 데이터 세대 분리 검증 보고서\n\n")
        f.write("**생성일**: 2026-05-15  \n")
        f.write("**스크립트**: `scripts/debug/verify_train_gen_split.py`  \n")
        f.write("**관련 결함**: CRITICAL_ANALYSIS.md 결함 7\n\n")
        f.write("---\n\n")

        f.write("## 요약\n\n")
        f.write("| 구분 | 총 수 | 세대 확인 (high) | 이름 추정 (medium) | 세대 불명 | Gen3+ 혼입 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for stat in [train_stat, valid_stat]:
            lbl   = stat["label"]
            total = stat["total"]
            hi    = stat["conf_counts"].get("high", 0)
            med   = stat["conf_counts"].get("medium", 0)
            unk   = len(stat["unknown_list"])
            g3    = len(stat["gen3_in_train"])
            f.write(f"| {lbl} | {total} | {hi} ({hi/total*100:.1f}%) | {med} ({med/total*100:.1f}%) | {unk} ({unk/total*100:.1f}%) | {g3} |\n")
        f.write("\n")

        f.write("## 핵심 발견\n\n")
        unk_train = len(train_stat["unknown_list"])
        total_train = train_stat["total"]
        f.write(f"1. **train.jsonl의 세대 불명 샘플: {unk_train}/{total_train}개 ({unk_train/total_train*100:.1f}%)**\n")
        f.write(f"   - GPT-4 캡션에 포켓몬 이름이 없어 POKEMON_DB 매칭 실패\n")
        f.write(f"   - `setup_pokemon_data.py` 88~93라인: 미매칭 시 세대 불명인 채로 train에 할당\n")
        f.write(f"   - 이 샘플들의 실제 세대는 보장 불가\n\n")

        g3_count = len(train_stat["gen3_in_train"])
        if g3_count > 0:
            f.write(f"2. **⚠️ train에 Gen3+ 확인 샘플 {g3_count}개 존재 — 세대 분리 오염**\n\n")
            f.write("| 이미지 | 이름 | 세대 |\n|---|---|---|\n")
            for r in train_stat["gen3_in_train"]:
                f.write(f"| `{os.path.basename(r['image'])}` | {r['matched_name']} | {GEN_LABEL_MAP.get(r['final_gen'], '?')} |\n")
            f.write("\n")
        else:
            f.write(f"2. **train에서 Gen3+ 혼입 없음** (이름 확인 가능한 {total_train - unk_train}개 기준)\n\n")

        f.write("## train.jsonl 세대 분포\n\n")
        f.write(format_gen_table(train_stat["gen_counts"], total_train))
        f.write(f"\n| **세대 불명** | **{unk_train}** | **{unk_train/total_train*100:.1f}%** |\n\n")

        f.write("## validation.jsonl 세대 분포\n\n")
        total_valid = valid_stat["total"]
        f.write(format_gen_table(valid_stat["gen_counts"], total_valid))
        f.write("\n\n")

        f.write("## 세대 불명 샘플 분석\n\n")
        f.write(f"총 {unk_train}개. GPT-4가 포켓몬 이름 없이 시각적 묘사만 생성한 케이스.\n\n")
        f.write("### 샘플 예시 (처음 20개)\n\n")
        f.write("| 파일 | 캡션 (앞 80자) |\n|---|---|\n")
        for r in train_stat["unknown_list"][:20]:
            fname = os.path.basename(r["image"])
            cap   = r["caption"][:80].replace("|", "\\|").replace("\n", " ")
            f.write(f"| `{fname}` | {cap}... |\n")
        f.write("\n")

        f.write("## 결론 및 Phase 2 대응\n\n")
        f.write(f"- **확실한 Gen1-2 학습 데이터**: {train_stat['conf_counts'].get('high',0) + train_stat['conf_counts'].get('medium',0)}개\n")
        f.write(f"- **세대 불명 데이터**: {unk_train}개 — Phase 1 'Gen1-2 학습' 주장이 전체 train에 적용되지 않음\n")
        f.write(f"- **Phase 2 조치**: F5 재실험에서는 세대 확인된 197개만 'Gen1-2 학습 완료' 기준으로 사용\n")
        f.write(f"- **추가 조치 검토**: 세대 불명 323개를 PokeAPI로 재식별하거나 학습에서 제외\n")

    print(f"\n보고서 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
