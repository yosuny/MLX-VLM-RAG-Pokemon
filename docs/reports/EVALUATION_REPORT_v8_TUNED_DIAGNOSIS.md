# Tuned 모델 과적합 vs 학습 미흡 진단 (v8)

**생성일**: 2026-05-15  
**관련 결함**: CRITICAL_ANALYSIS.md 결함 6

---

## 실험 설계

| 구분 | 소스 | 설명 |
|---|---|---|
| A (Train 이미지) | train.jsonl | Tuned 모델이 학습 시 본 이미지 |
| B (Sprite 이미지) | PokeAPI | 학습에서 보지 않은 공식 artwork |

가설: A >> B면 과적합(H1), A ≈ B면 학습 미흡(H2)

## 결과 요약

| | 정확도 |
|---|---|
| A 학습 이미지 (n=41) | 21/41 (51.2%) |
| B 스프라이트  (n=41) | 21/41 (51.2%) |
| 차이 (A−B) | +0.0%p |

## 판정

**H2 (학습 미흡): 두 소스 모두 유사하게 낮은 정확도**

## 분석

### H2 확정의 의미

학습 이미지와 새 이미지에서 동일한 51.2% 정확도는 파인튜닝이 이미지-이름 매핑을 전혀 학습하지 못했음을 의미한다.

**오답 패턴 분류:**

| 패턴 | 예시 | 빈도 |
|---|---|---|
| 두 이미지에서 **동일 오답** | machamp→"Groudon", gyarados→"Dragonair" | 다수 |
| 두 이미지에서 **다른 오답** | alakazam→"Garchomp"(train), "Gallade"(sprite) | 일부 |
| **학습 이미지만 정답** | venusaur: train ❌, sprite ✅ | 1건 (역방향!) |
| **새 이미지만 정답** | chikorita: train ✅, sprite ❌ | 1건 |

### 왜 학습 미흡이 발생했는가?

Phase 2 F1에서 발견한 사실: train.jsonl 520개 중 323개(62.1%)의 정답 캡션에 **포켓몬 이름 자체가 없다** (GPT-4가 시각적 묘사만 생성).

예시 캡션: `"A blue, crab-like Pokémon with a hard shell and prominent claws stares intently."`

이 경우 모델이 학습하는 것은 `"시각적 묘사 → 해당 묘사 텍스트"` 패턴일 뿐,
`"이 이미지 → 이름"` 매핑이 아니다.

**결론**: 파인튜닝이 한국어 이름을 주입하지 못한 것은 아키텍처 문제가 아니라
**학습 데이터의 62.1%에 이름 레이블이 없는** 데이터 품질 문제다.

### Phase 3에서 개선하려면

1. 세대 불명 323개를 PokeAPI로 재식별하여 이름 레이블 추가
2. LoRA rank 증가 (현재 8 → 32 이상)
3. 학습 step 증가 및 더 낮은 LR

## 상세 결과

| 포켓몬 | Train 출력 | 정답 | Sprite 출력 | 정답 |
|---|---|---|---|---|
| **bulbasaur** | This is Bulbasaur (Bulbasaur in English and 불바사루 i... | ✅ | This is Bulbasaur.

Bulbasaur (버브라우)... | ✅ |
| **charmander** | This is the Pokémon named "Charmander" in English ... | ✅ | This is Charmander. In Korean, it is called "차르만더"... | ✅ |
| **squirtle** | This is Squirtle. In Korean, it is called "스쿼트" (S... | ✅ | This is Squirtle (스쿼트) in English and Korean.... | ✅ |
| **pikachu** | This is the Pokémon Pikachu dressed as a princess.... | ✅ | This is Pikachu.

Pokemon: Pikachu
Pokemon: 피카츄... | ✅ |
| **mewtwo** | This is the Pokémon Mewtwo. In Korean, it is calle... | ✅ | This is Mewtwo, also known as 미우투 in Korean.... | ✅ |
| **eevee** | This is the Pokémon called "Eevee" in English and ... | ✅ | This is Eevee in English and 이브로 in Korean.... | ✅ |
| **snorlax** | This is Snorlax (Snorlax) in English and 콩알 (Konga... | ✅ | This is Snorlax (Snorlax) in English and Korean.... | ✅ |
| **gengar** | This is Gengar (강거) from the Pokémon series.... | ✅ | This is Gengar (강거) in English and Korean.... | ✅ |
| **charizard** | This is the Pokémon Charizard. In Korean, it is ca... | ✅ | This is Charizard (Charizard) in English and 챌리zá드... | ✅ |
| **blastoise** | This is the Pokémon Blastoise (블라스토이즈, Blastoise).... | ✅ | This is Blastoise (불도덕) in English and Korean.... | ✅ |
| **venusaur** | This is the Pokémon named "Bulbasaur" in English a... | ❌ | This is the Pokémon named "Venusaur" in English an... | ✅ |
| **jigglypuff** | This is Jigglypuff. In Korean, it is called "제일피플"... | ✅ | This is Jigglypuff.

English Name: Jigglypuff
Kore... | ✅ |
| **meowth** | This is Meowth (미우스) in English and Korean.... | ✅ | This is Meowth (미우스) in English and Korean.... | ✅ |
| **psyduck** | This is the Pokémon named "Psyduck" in English and... | ✅ | This is the Pokémon named "Psyduck" in English and... | ✅ |
| **machamp** | This is Groudon (Groudon) in English and 굴돈 (Gurud... | ❌ | This is Groudon (Groudon) in English and 굴돈 (Gurud... | ❌ |
| **alakazam** | This is the Pokémon named "Garchomp" in English an... | ❌ | This is the Pokémon called "Gallade" in English an... | ❌ |
| **gyarados** | This is the Pokémon named "Dragonair" in English a... | ❌ | This is the Pokémon "Dragonair" in English and "드래... | ❌ |
| **dragonite** | This is the Pokémon named "Dragonite" in English a... | ✅ | This is a Dragonite (드래곤라이트, Drakoraito) from the ... | ✅ |
| **raichu** | This is Raichu (雷丘) in English and 레이슈 (Reisu) in ... | ✅ | This is Raichu (雷丘) in English and 레이슈 (Reisu) in ... | ✅ |
| **clefable** | This is the Pokémon named "Gardevoir" in English a... | ❌ | This is the Pokémon named "Machamp" in English and... | ❌ |
| **ninetales** | This is the Pokémon Vulpix (불소) in English and 불소 ... | ❌ | This is the Pokémon "Vulpix" in English and "불소" i... | ❌ |
| **magmar** | This is the Pokémon named "Growlithe" in English a... | ❌ | This is a Charizard, which in Korean is called "차리... | ❌ |
| **electabuzz** | This is the Pokémon named "Electivire" in English ... | ❌ | This is the Pokémon Zoroark (Zoroark in English an... | ❌ |
| **scyther** | This is the Pokémon Zoroark (Zoroark in English an... | ❌ | This is Zoroark (Zoroark) in English and 코로아크 (Koo... | ❌ |
| **pinsir** | This is the Pokémon named "Groudon" in English and... | ❌ | This is the Pokémon named "Groudon" in English and... | ❌ |
| **lapras** | This Pokémon is named "Crobat" in English and "크로브... | ❌ | This is the Pokémon named "Glaceon" in English and... | ❌ |
| **dragonair** | This is the Pokémon called "Dragonair" in English ... | ✅ | This is the Pokémon "Dragonair" in English and "드래... | ✅ |
| **hitmonchan** | This is the Pokémon named "Bellossom" in English a... | ❌ | This is the Pokémon named "Gardevoir" in English a... | ❌ |
| **hitmonlee** | This is the Pokémon named "Gastly" in English and ... | ❌ | This is the Pokémon named "Gardevoir" in English a... | ❌ |
| **clefairy** | This is the Pokémon named "Cubone" in English and ... | ❌ | This is the Pokémon named Ditto in English and 디토 ... | ❌ |
| **growlithe** | This is the Pokémon named "Growlithe" in English a... | ✅ | This is Growlithe (Growlithe) in English and 골라이스 ... | ✅ |
| **poliwrath** | This is the Pokémon named "Gengar" in English and ... | ❌ | This is the Pokémon named "Psyduck" in English and... | ❌ |
| **abra** | This is the Pokémon named "Meowth" in English and ... | ❌ | This is a picture of the Pokémon named "Jumplung" ... | ❌ |
| **geodude** | This is the Pokémon named "Groudon" in English and... | ❌ | This is the Pokémon named "Groudon" in English and... | ❌ |
| **haunter** | This is the Pokémon named "Gengar" in English and ... | ❌ | This is the Pokémon "Gengar" in English and "강가" i... | ❌ |
| **cubone** | This is the Pokémon named "Cubone" in English and ... | ✅ | This is the Pokémon named "Cubone" in English and ... | ✅ |
| **hitmontop** | This Pokémon is named "Dugtrio" in English and "도그... | ❌ | This is the Pokémon named "Dugtrio" in English and... | ❌ |
| **chikorita** | This is the Pokémon named "Chikorita" in English a... | ✅ | This is the Pokémon named "Seedot" in English and ... | ❌ |
| **cyndaquil** | This is the Pokémon named "Cyndaquil" in English a... | ✅ | This is the Pokémon named "Cyndaquil" in English a... | ✅ |
| **totodile** | This is the Pokémon named "Growlithe" in English a... | ❌ | This is a picture of a Pokémon named "Croagunk" in... | ❌ |
| **umbreon** | This is Umbreon (은브론) in Korean.... | ✅ | This is Umbreon (은로미) in English and Korean.... | ✅ |
