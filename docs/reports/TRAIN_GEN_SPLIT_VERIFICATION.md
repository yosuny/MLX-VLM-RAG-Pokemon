# 학습 데이터 세대 분리 검증 보고서

**생성일**: 2026-05-15  
**스크립트**: `scripts/debug/verify_train_gen_split.py`  
**관련 결함**: CRITICAL_ANALYSIS.md 결함 7

---

## 요약

| 구분 | 총 수 | 세대 확인 (high) | 이름 추정 (medium) | 세대 불명 | Gen3+ 혼입 |
|---|---|---|---|---|---|
| train | 520 | 197 (37.9%) | 0 (0.0%) | 323 (62.1%) | 0 |
| validation | 313 | 313 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 |

## 핵심 발견

1. **train.jsonl의 세대 불명 샘플: 323/520개 (62.1%)**
   - GPT-4 캡션에 포켓몬 이름이 없어 POKEMON_DB 매칭 실패
   - `setup_pokemon_data.py` 88~93라인: 미매칭 시 세대 불명인 채로 train에 할당
   - 이 샘플들의 실제 세대는 보장 불가

2. **train에서 Gen3+ 혼입 없음** (이름 확인 가능한 197개 기준)

## train.jsonl 세대 분포

| 세대 | 수량 | 비율 |
|---|---|---|
| GEN I | 141 | 27.1% |
| GEN II | 56 | 10.8% |
| **세대 불명** | **323** | **62.1%** |

## validation.jsonl 세대 분포

| 세대 | 수량 | 비율 |
|---|---|---|
| GEN III | 104 | 33.2% |
| GEN IV | 79 | 25.2% |
| GEN V | 75 | 24.0% |
| GEN VI | 51 | 16.3% |
| GEN VII | 2 | 0.6% |
| GEN VIII | 2 | 0.6% |

## 세대 불명 샘플 분석

총 323개. GPT-4가 포켓몬 이름 없이 시각적 묘사만 생성한 케이스.

### 샘플 예시 (처음 20개)

| 파일 | 캡션 (앞 80자) |
|---|---|
| `pokemon_003.jpg` | A friendly, smiling white orb with a rosy blush and closed eyes expresses conten... |
| `pokemon_012.jpg` | An illustrated image of a green, cocoon-like Pokemon with a single visible eye, ... |
| `pokemon_021.jpg` | A serene illustration of a light blue sea serpent-like dragon with a cream under... |
| `pokemon_026.jpg` | A stylized illustration of a purple and gray star-shaped object with a yellow ce... |
| `pokemon_027.jpg` | An animated character resembling a jester with a smiling face, blue appendages, ... |
| `pokemon_032.jpg` | A fierce animated bee-like creature with piercing scythes, large wings, and a bo... |
| `pokemon_035.jpg` | An illustrated orange fish-like character with noticeable whiskers and iconic fi... |
| `pokemon_037.jpg` | A ferocious blue and yellow aquatic Pokémon with sharp fangs and red spots prepa... |
| `pokemon_040.jpg` | A cheerful, amorphous purple blob with a simple, contented smile.... |
| `pokemon_045.jpg` | A colorful, geometric illustration of an origami-style bird with shades of pink ... |
| `pokemon_054.jpg` | A majestic blue bird Pokémon with expansive wings and a long flowing tail, appea... |
| `pokemon_055.jpg` | An illustrated electric-type Pokémon with sharp talons and a fierce expression, ... |
| `pokemon_056.jpg` | A majestic, fiery-winged bird with a streamlined body, resembling a phoenix in f... |
| `pokemon_060.jpg` | A fearsome, stylized yellow and black bee-like creature with large wings and sha... |
| `pokemon_061.jpg` | A stylized illustration of a yellow and black striped, bee-like creature with la... |
| `pokemon_065.jpg` | A whimsical pink creature with large blue eyes and long, slender tail floats aga... |
| `pokemon_067.jpg` | An illustration of a friendly-looking yellow dinosaur-like creature with leaf-sh... |
| `pokemon_070.jpg` | A stern-looking bipedal Pokémon with fiery orange spikes on its back and a cool ... |
| `pokemon_076.jpg` | A curious brown and cream-colored rabbit-like creature with a distinctive long, ... |
| `pokemon_079.jpg` | A stern-faced, brown-feathered owl Pokémon with a leaf-shaped crown and piercing... |

## 결론 및 Phase 2 대응

- **확실한 Gen1-2 학습 데이터**: 197개
- **세대 불명 데이터**: 323개 — Phase 1 'Gen1-2 학습' 주장이 전체 train에 적용되지 않음
- **Phase 2 조치**: F5 재실험에서는 세대 확인된 197개만 'Gen1-2 학습 완료' 기준으로 사용
- **추가 조치 검토**: 세대 불명 323개를 PokeAPI로 재식별하거나 학습에서 제외
