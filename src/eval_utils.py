"""
Phase 2 F2: 공통 평가 유틸리티
CRITICAL_ANALYSIS.md 결함 2, 3 해소

변경 사항 (Phase 1 대비):
- substring match → word-boundary 정규식 매칭
- 한국어 이름 별도 측정
- 정확도 산출이 재현 가능하도록 함수 모듈화
- 판정 근거 로그 출력 지원
"""

import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 정확도 판정 함수
# ---------------------------------------------------------------------------

def check_name_match(output: str, target_name: str) -> bool:
    """
    Word-boundary 기반 영어 이름 매칭.

    Phase 1 문제: 'pikachu' in output.lower() → 'pikachu' in 'not pikachu' = True
    Phase 2 수정: \b 단어 경계로 부분 포함 false positive 제거.

    Args:
        output:      모델 출력 전체 텍스트
        target_name: 정답 포켓몬 영어 이름 (예: "Pikachu")
    Returns:
        bool: 이름이 단어 단위로 포함되면 True
    """
    pattern = rf'\b{re.escape(target_name)}\b'
    return bool(re.search(pattern, output, re.IGNORECASE))


def check_name_match_strict(output: str, target_name: str) -> bool:
    """
    첫 번째 문장에서만 이름 매칭 (응답 후반부 우연 포함 제거).

    Args:
        output:      모델 출력 전체 텍스트
        target_name: 정답 포켓몬 영어 이름
    Returns:
        bool
    """
    first_sentence = re.split(r'[.!?\n]', output)[0]
    pattern = rf'\b{re.escape(target_name)}\b'
    return bool(re.search(pattern, first_sentence, re.IGNORECASE))


def check_korean_match(output: str, target_korean: str) -> bool:
    """
    한국어 이름 매칭 (word boundary 없이, 한국어는 공백/구두점 인접 확인).

    Args:
        output:         모델 출력
        target_korean:  정답 한국어 이름 (예: "피카츄")
    Returns:
        bool
    """
    if not target_korean:
        return False
    return target_korean in output


def check_accuracy(output: str, target_name: str, target_korean: Optional[str] = None,
                   mode: str = "strict") -> dict:
    """
    통합 정확도 판정 함수. 판정 근거도 함께 반환.

    Args:
        output:         모델 출력 텍스트
        target_name:    정답 영어 이름
        target_korean:  정답 한국어 이름 (None이면 측정 안 함)
        mode:           "strict" (첫 문장) | "loose" (전체 텍스트 word-boundary)
    Returns:
        dict: {
            "en_correct":    bool,
            "kr_correct":    bool | None,
            "correct":       bool,   # en_correct 기준
            "mode":          str,
            "matched_text":  str,    # 매칭된 텍스트 (디버그용)
        }
    """
    if mode == "strict":
        en_correct = check_name_match_strict(output, target_name)
        matched_text = re.split(r'[.!?\n]', output)[0][:80]
    else:
        en_correct = check_name_match(output, target_name)
        matched_text = output[:80]

    kr_correct = check_korean_match(output, target_korean) if target_korean else None

    return {
        "en_correct":   en_correct,
        "kr_correct":   kr_correct,
        "correct":      en_correct,
        "mode":         mode,
        "matched_text": matched_text,
    }


# ---------------------------------------------------------------------------
# 결과 집계 클래스
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """단일 샘플 평가 결과"""
    name:           str
    image_path:     str
    ground_truth:   str
    korean_name:    Optional[str]

    vanilla_output: str = ""
    rag_output:     str = ""
    tuned_output:   str = ""
    rag_retrieval:  str = ""

    vanilla_acc:    dict = field(default_factory=dict)
    rag_acc:        dict = field(default_factory=dict)
    tuned_acc:      dict = field(default_factory=dict)

    def evaluate(self, mode: str = "strict"):
        korean = self.korean_name
        self.vanilla_acc = check_accuracy(self.vanilla_output, self.ground_truth, korean, mode)
        self.rag_acc     = check_accuracy(self.rag_output,     self.ground_truth, korean, mode)
        self.tuned_acc   = check_accuracy(self.tuned_output,   self.ground_truth, korean, mode)

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "image_path":     self.image_path,
            "ground_truth":   self.ground_truth,
            "korean_name":    self.korean_name,
            "vanilla_output": self.vanilla_output,
            "rag_output":     self.rag_output,
            "tuned_output":   self.tuned_output,
            "rag_retrieval":  self.rag_retrieval,
            "vanilla_acc":    self.vanilla_acc,
            "rag_acc":        self.rag_acc,
            "tuned_acc":      self.tuned_acc,
        }


@dataclass
class EvalSummary:
    """평가 결과 집계"""
    total:           int = 0
    vanilla_en:      int = 0
    vanilla_kr:      int = 0
    rag_en:          int = 0
    rag_kr:          int = 0
    tuned_en:        int = 0
    tuned_kr:        int = 0
    mode:            str = "strict"

    def add(self, result: EvalResult):
        self.total += 1
        if result.vanilla_acc.get("en_correct"): self.vanilla_en += 1
        if result.vanilla_acc.get("kr_correct"): self.vanilla_kr += 1
        if result.rag_acc.get("en_correct"):     self.rag_en     += 1
        if result.rag_acc.get("kr_correct"):     self.rag_kr     += 1
        if result.tuned_acc.get("en_correct"):   self.tuned_en   += 1
        if result.tuned_acc.get("kr_correct"):   self.tuned_kr   += 1

    def pct(self, n: int) -> str:
        if self.total == 0:
            return "N/A"
        return f"{n}/{self.total} ({n/self.total*100:.1f}%)"

    def report_lines(self) -> list[str]:
        lines = [
            f"## 정확도 요약 (n={self.total}, mode={self.mode})\n",
            "| 모델 | 영어 이름 정확도 | 한국어 이름 정확도 |",
            "|---|---|---|",
            f"| Vanilla | {self.pct(self.vanilla_en)} | {self.pct(self.vanilla_kr)} |",
            f"| RAG     | {self.pct(self.rag_en)}     | {self.pct(self.rag_kr)}     |",
            f"| Tuned   | {self.pct(self.tuned_en)}   | {self.pct(self.tuned_kr)}   |",
        ]
        return lines


# ---------------------------------------------------------------------------
# 보고서 생성 유틸
# ---------------------------------------------------------------------------

def save_results_json(results: list[EvalResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


def write_markdown_report(results: list[EvalResult], summary: EvalSummary,
                           title: str, path: str, extra_sections: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**생성일**: 2026-05-15  \n")
        f.write(f"**판정 기준**: word-boundary match, mode={summary.mode}  \n\n")
        f.write("---\n\n")

        # 요약 테이블
        f.write("\n".join(summary.report_lines()))
        f.write("\n\n")

        if extra_sections:
            f.write(extra_sections)
            f.write("\n\n")

        # 상세 결과
        f.write("## 상세 결과\n\n")
        f.write("| 포켓몬 | Vanilla | RAG | Tuned |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            v_icon = "✅" if r.vanilla_acc.get("en_correct") else "❌"
            r_icon = "✅" if r.rag_acc.get("en_correct")     else "❌"
            t_icon = "✅" if r.tuned_acc.get("en_correct")   else "❌"
            v_kr = "🇰🇷" if r.vanilla_acc.get("kr_correct") else ""
            r_kr = "🇰🇷" if r.rag_acc.get("kr_correct")     else ""
            t_kr = "🇰🇷" if r.tuned_acc.get("kr_correct")   else ""
            f.write(
                f"| **{r.name}** "
                f"| {v_icon}{v_kr} {r.vanilla_output[:50]}... "
                f"| {r_icon}{r_kr} {r.rag_output[:50]}... "
                f"| {t_icon}{t_kr} {r.tuned_output[:50]}... |\n"
            )


# ---------------------------------------------------------------------------
# 셀프 테스트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== eval_utils.py 셀프 테스트 ===\n")

    # 1. word-boundary 테스트
    cases = [
        ("This is Pikachu. 피카츄.",         "Pikachu",   True,  "정상 매칭"),
        ("This is not Pikachu, it's Raichu", "Pikachu",   True,  "문장 내 포함 (loose에서 true)"),
        ("PikachuExtra is here",             "Pikachu",   False, "단어 경계 미매칭"),
        ("This is Raichu.",                  "Pikachu",   False, "오답"),
        ("This is Charizard (리자몽).",       "Charizard", True,  "괄호 앞 매칭"),
    ]

    all_pass = True
    for text, name, expected, desc in cases:
        result = check_name_match(text, name)
        icon = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {icon} [{desc}] check_name_match('{text[:40]}...', '{name}') = {result}")

    # 2. strict mode (첫 문장)
    strict_cases = [
        ("This is Pikachu. But it could also be Raichu.", "Pikachu",   True,  "첫 문장 정답"),
        ("This is Raichu. Pikachu is different.",         "Pikachu",   False, "두 번째 문장에만 있음"),
    ]
    for text, name, expected, desc in strict_cases:
        result = check_name_match_strict(text, name)
        icon = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {icon} [strict/{desc}] check_name_match_strict('{text[:50]}', '{name}') = {result}")

    # 3. 한국어 테스트
    kr_cases = [
        ("This is Pikachu (피카츄).", "피카츄", True),
        ("이것은 피카츄입니다.",        "피카츄", True),
        ("이것은 라이츄입니다.",        "피카츄", False),
    ]
    for text, kr, expected in kr_cases:
        result = check_korean_match(text, kr)
        icon = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {icon} [한국어] check_korean_match('{text}', '{kr}') = {result}")

    print(f"\n{'모든 테스트 통과 ✅' if all_pass else '일부 테스트 실패 ❌'}")
