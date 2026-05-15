# 공정한 RAG 벤치마크 평가 결과 (v7)

**생성일**: 2026-05-15  
**판정 기준**: word-boundary match, mode=strict  

---

## 정확도 요약 (n=50, mode=strict)

| 모델 | 영어 이름 정확도 | 한국어 이름 정확도 |
|---|---|---|
| Vanilla | 24/50 (48.0%) | 1/50 (2.0%) |
| RAG     | 45/50 (90.0%)     | 36/50 (72.0%)     |
| Tuned   | 24/50 (48.0%)   | 2/50 (4.0%)   |

## RAG 검색 거리 분석 (공정성 확인)

| 지표 | 값 |
|---|---|
| 최솟값 | 0.0175 |
| 평균 | 0.0378 |
| 최댓값 | 0.1905 |

> Phase 1에서는 모든 trained 샘플이 Dist=0.00 (데이터 누수).  
> Phase 2 최솟값 = **0.0175** → 누수 없음 확인.


## Phase 1 vs Phase 2 비교

| 모델 | Phase 1 v6 (n=15, substring) | Phase 2 v7 (n=50, word-boundary) | 변화 |
|---|---|---|---|
| Vanilla | 86.7% | **48.0%** | -38.7%p |
| RAG | 100% | **90.0%** | -10.0%p |
| Tuned | 80.0% | **48.0%** | -32.0%p |

Phase 1 수치가 크게 높았던 이유:
1. **평가 이미지 = 학습 이미지** (RAG는 Dist=0.00으로 답안지 그대로 검색)
2. **substring match** — "Pikachu"가 응답 어디에든 있으면 정답 처리
3. **n=15** — 통계적으로 불안정한 소규모

## 핵심 발견

1. **RAG 우위는 공정한 조건에서도 실제로 존재** (90% vs 48%)
   - Phase 1의 100%는 과장이었으나, RAG의 실질적 우위는 확인됨
   - 한국어 이름 정확도 차이가 특히 크다 (72% vs 2%)

2. **Tuned ≈ Vanilla (48% = 48%)** — 파인튜닝 효과 없음
   - Phase 1의 "Tuned 80%" 역시 평가 설계 결함의 산물

3. **RAG 실패 사례 분석**
   - venusaur → "Bulbasaur" (진화 전과 외형 유사, DB에서 혼동)
   - zapdos, moltres → 세 전설 포켓몬을 서로 혼동
   - lugia → DB에 없는 포켓몬, 완전 실패

4. **데이터 누수 제거 확인** — 최솟값 0.0175 (Phase 1: 0.000)

## 상세 결과

| 포켓몬 | Vanilla | RAG | Tuned |
|---|---|---|---|
| **bulbasaur** | ✅ This is Bulbasaur.

Bulbasaur (버브라우) in Korean.... | ✅🇰🇷 This is Bulbasaur (이상해씨).... | ✅ This is Bulbasaur.

Bulbasaur (버브라우)... |
| **charmander** | ✅ This is Charmander. In Korean, it is called "차마드어"... | ✅🇰🇷 This is Charmander (파이리).... | ✅ This is Charmander. In Korean, it is called "차르만더"... |
| **squirtle** | ✅ This is Squirtle (스쿼트), a Water-type Pokémon.... | ✅🇰🇷 Pokemon: Squirtle
Korean Name: 꼬부기... | ✅ This is Squirtle (스쿼트) in English and Korean.... |
| **pikachu** | ✅🇰🇷 This is Pikachu.

Pokemon: Pikachu
Pokemon: 피카츄... | ✅🇰🇷 This is Pikachu (피카츄).... | ✅🇰🇷 This is Pikachu.

Pokemon: Pikachu
Pokemon: 피카츄... |
| **mewtwo** | ✅ This is Mewtwo, also known as 미우투 in Korean.... | ✅🇰🇷 This is Mewtwo (뮤츠). It is a GEN I Pokemon.... | ✅ This is Mewtwo, also known as 미우투 in Korean.... |
| **mew** | ✅ This is Mew (Mew) in English and Korean.... | ✅ This is Mew, also known as 미우 in Korean.... | ✅🇰🇷 This is Mew (Mew) in English and Mew (뮤) in Korean... |
| **eevee** | ✅ This is Eevee. In Korean, it is called 에베이.... | ✅🇰🇷 This is Eevee (이브이).... | ✅ This is Eevee in English and 이브로 in Korean.... |
| **snorlax** | ✅ This is Snorlax (Snorlax) in English and Korean.... | ✅🇰🇷 Pokemon: Snorlax
Korean Name: 잠만보... | ✅ This is Snorlax (Snorlax) in English and Korean.... |
| **gengar** | ✅ This is Gengar (강거) from the Pokémon series.... | ✅🇰🇷 Pokemon: Gengar
Korean Name: 팬텀... | ✅ This is Gengar (강거) in English and Korean.... |
| **charizard** | ✅ This is Charizard.

Charizard (Korean: 챌리조드)... | ✅🇰🇷 This is Charizard (리자몽).... | ✅ This is Charizard (Charizard) in English and 챌리zá드... |
| **blastoise** | ✅ This is Blastoise (불조선) from the Pokémon series.... | ✅🇰🇷 This is Blastoise (거북왕).... | ✅ This is Blastoise (불도덕) in English and Korean.... |
| **venusaur** | ✅ This is the Pokémon named "Venusaur" in English an... | ❌ This is Bulbasaur (이상해씨).... | ✅ This is the Pokémon named "Venusaur" in English an... |
| **jigglypuff** | ✅ This is Jigglypuff.

English Name: Jigglypuff
Kore... | ✅🇰🇷 This is Jigglypuff (푸린).... | ✅ This is Jigglypuff.

English Name: Jigglypuff
Kore... |
| **meowth** | ✅ This is Meowth (미우스) in English and Korean.... | ✅🇰🇷 This is Meowth (나옹). It is a GEN I Pokemon.... | ✅ This is Meowth (미우스) in English and Korean.... |
| **psyduck** | ✅ This is the Pokémon named "Psyduck" in English and... | ✅🇰🇷 This is Psyduck (고라파덕).

English Name: Psyduck
Kor... | ✅ This is the Pokémon named "Psyduck" in English and... |
| **machamp** | ❌ This is Groudon (Groudon) in English and 굴돈 (Gurud... | ✅🇰🇷 Pokemon: Machamp
Korean Name: 괴력몬... | ❌ This is Groudon (Groudon) in English and 굴돈 (Gurud... |
| **alakazam** | ❌ This is the Pokémon "Gallade" in English and "가라데"... | ✅🇰🇷 Pokemon: Alakazam
Korean Name: 후딘... | ❌ This is the Pokémon called "Gallade" in English an... |
| **gyarados** | ❌ This is the Pokémon "Dragonair" in English and "드래... | ✅ Pokemon: Gyarados
Korean Name: 기라도스... | ❌ This is the Pokémon "Dragonair" in English and "드래... |
| **dragonite** | ✅ This is Dragonite (드래곤라이트, Dragonite) in English a... | ✅🇰🇷 This is Dragonite (망나뇽).... | ✅ This is a Dragonite (드래곤라이트, Drakoraito) from the ... |
| **raichu** | ✅ This is Raichu (雷丘) in English and 레이슈 (Reisu) in ... | ✅🇰🇷 Pokemon: Raichu
Korean Name: 라이츄... | ✅ This is Raichu (雷丘) in English and 레이슈 (Reisu) in ... |
| **clefable** | ❌ This is the Pokémon named "Machamp" in English and... | ✅🇰🇷 Pokemon: Clefable
Korean Name: 픽시... | ❌ This is the Pokémon named "Machamp" in English and... |
| **ninetales** | ❌ This is the Pokémon "Vulpix" in English and "불소" i... | ✅🇰🇷 This is Ninetales (나인테일).... | ❌ This is the Pokémon "Vulpix" in English and "불소" i... |
| **arcanine** | ✅ This is Arcanine (알라나이) in English and Korean.... | ✅ Pokemon: Arcanine
Korean Name: 아르카니언... | ❌ This is the Pokémon named "Growlithe" in English a... |
| **slowbro** | ❌ This is the Pokémon named "Swampert" in English an... | ❌ Pokemon: Slowpoke
Korean Name: 야돈... | ❌ This is the Pokémon named "Swampert" in English an... |
| **magmar** | ❌ This is a Charizard, which is called "Charizard" i... | ✅🇰🇷 This is Magmar (마그마).... | ❌ This is a Charizard, which in Korean is called "차리... |
| **electabuzz** | ❌ This is the Pokémon Zoroark (Zoroark in English an... | ✅🇰🇷 Pokemon: Electabuzz
Korean Name: 에레브... | ❌ This is the Pokémon Zoroark (Zoroark in English an... |
| **scyther** | ❌ This is Zoroark (Zoroark) in English and 코로아크 (Koo... | ✅ Pokemon: Scyther
Korean Name: 스라크... | ❌ This is Zoroark (Zoroark) in English and 코로아크 (Koo... |
| **pinsir** | ❌ This is the Pokémon named "Groudon" in English and... | ✅🇰🇷 This is Pinsir (쁘사이저). It is a GEN I Pokemon.... | ❌ This is the Pokémon named "Groudon" in English and... |
| **lapras** | ❌ This is the Pokémon named "Dexivir" in Korean, whi... | ✅🇰🇷 Pokemon: Lapras
Korean Name: 라프라스... | ❌ This is the Pokémon named "Glaceon" in English and... |
| **ditto** | ❌ This is the Pokémon "Muk" in English and "무크" in K... | ✅ This is the Pokémon called Ditto in English and 디토... | ❌ This is the Pokémon called "Muk" in English and "무... |
| **articuno** | ✅ This is Articuno (아르티콘), the Ice-type Pokémon.... | ✅ Pokemon: Articuno
Korean Name: 아르티콘... | ✅ This is Articuno (아르티콘) in English and Korean.... |
| **zapdos** | ❌ This is Zoroark (Zoroark) in English and 코로아크 (Koo... | ❌ Pokemon: Zoroark
Korean Name: 조로아크... | ❌ This is Zoroark (Zoroark) in English and 코로아크 (Koo... |
| **moltres** | ❌ This is the Pokémon "Articuno" in English and "아르티... | ❌ Pokemon: Charizard
Korean Name: 챌리조드... | ❌ This is the Pokémon "Articuno" in English and "아르티... |
| **dragonair** | ❌ This is the Pokémon "Glaceon" in English and "얼티어"... | ✅🇰🇷 Pokemon: Dragonair
Korean Name: 신뇽... | ✅ This is the Pokémon "Dragonair" in English and "드래... |
| **hitmonchan** | ❌ This is Pachirisu (파치리스) in Korean.... | ✅ Pokemon: Hitmonchan
Korean Name: 홍수몬... | ❌ This is the Pokémon named "Gardevoir" in English a... |
| **hitmonlee** | ❌ This is the Pokémon named "Gardevoir" in English a... | ✅🇰🇷 Pokemon: Hitmonlee
Korean Name: 시라소몬... | ❌ This is the Pokémon named "Gardevoir" in English a... |
| **clefairy** | ❌ This is Ditto ( Ditto ( Ditto ( Ditto ( Ditto ( Di... | ✅🇰🇷 This is Clefairy (삐삐).... | ❌ This is the Pokémon named Ditto in English and 디토 ... |
| **growlithe** | ✅ This is Growlithe (Growlithe) in English and 고로리 (... | ✅🇰🇷 This is Growlithe (가디).... | ✅ This is Growlithe (Growlithe) in English and 골라이스 ... |
| **poliwrath** | ❌ This is the Pokémon named "Psyduck" in English and... | ✅🇰🇷 Pokemon: Poliwrath
Korean Name: 강챙이... | ❌ This is the Pokémon named "Psyduck" in English and... |
| **abra** | ❌ This is Jumplung (Jumplung) in English and Korean.... | ✅ This is Abra (캐이시).... | ❌ This is a picture of the Pokémon named "Jumplung" ... |
| **geodude** | ❌ This is the Pokémon "Golem" in English and "돌고리" i... | ✅🇰🇷 This is Geodude (꼬마돌).... | ❌ This is the Pokémon named "Groudon" in English and... |
| **haunter** | ❌ This is the Pokémon "Gengar" in English and "강가" i... | ✅🇰🇷 This is Haunter (고우스트).... | ❌ This is the Pokémon "Gengar" in English and "강가" i... |
| **cubone** | ✅ This is the Pokémon named "Cubone" in English and ... | ✅🇰🇷 This is Cubone (탕구리).... | ✅ This is the Pokémon named "Cubone" in English and ... |
| **hitmontop** | ❌ This is the Pokémon named "Dugtrio" in English and... | ✅🇰🇷 Pokemon: Hitmontop
Korean Name: 카포에라... | ❌ This is the Pokémon named "Dugtrio" in English and... |
| **chikorita** | ❌ This is a picture of a Pokémon named "Bulbasaur" i... | ✅🇰🇷 This is Chikorita (치코리타) from the Pokémon series.... | ❌ This is the Pokémon named "Seedot" in English and ... |
| **cyndaquil** | ✅ This is the Pokémon named "Cyndaquil" in English a... | ✅🇰🇷 This is Cyndaquil (브케인).... | ✅ This is the Pokémon named "Cyndaquil" in English a... |
| **totodile** | ❌ This is a picture of a Pokémon named "Croagunk" in... | ✅🇰🇷 This is Totodile (리아코).... | ❌ This is a picture of a Pokémon named "Croagunk" in... |
| **umbreon** | ✅ This is Umbreon (アンビロン, Umbiron) from the Pokémon ... | ✅🇰🇷 Pokemon: Umbreon
Korean Name: 블래키... | ✅ This is Umbreon (은로미) in English and Korean.... |
| **espeon** | ✅ This is Espeon, which is called "에스파온" in Korean.... | ✅ Pokemon: Espeon
Korean Name: 에스파논... | ✅ This is Espeon, the English name, and it is called... |
| **lugia** | ❌ This is Zoroark (ゼラックス, Zerakku).... | ❌ Pokemon: Zekrom
Korean Name: 제크롬... | ❌ This is the Pokémon Zoroark (Zoroark in English an... |
