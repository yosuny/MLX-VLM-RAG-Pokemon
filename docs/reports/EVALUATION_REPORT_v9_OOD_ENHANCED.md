# OOD 강화 테스트 결과 (v9)

**생성일**: 2026-05-15  
**관련 결함**: CRITICAL_ANALYSIS.md 결함 4

---

## 실험 설계

Gen3-9 미학습 포켓몬 30종을 3가지 카테고리로 분류하여 RAG의 한계를 정량화.

| 카테고리 | 설명 | 종 수 |
|---|---|---|
| evolution | Gen1-2 포켓몬의 진화형/전진화 | 10 |
| visual | 시각적으로 Gen1-2와 유사 | 10 |
| distinct | 외형이 완전히 이질적 | 10 |

## 카테고리별 정확도

| 카테고리 | Vanilla | RAG |
|---|---|---|
| evolution | 3/10 (30.0%) | 1/10 (10.0%) |
| visual | 1/10 (10.0%) | 2/10 (20.0%) |
| distinct | 1/10 (10.0%) | 1/10 (10.0%) |
| **전체** | **5/30 (16.7%)** | **4/30 (13.3%)** |

## 핵심 발견

### 1. evolution 카테고리에서 RAG < Vanilla
가장 놀라운 결과. RAG(10%)가 Vanilla(30%)보다 낮다.

**원인**: RAG가 진화 전 포켓몬을 힌트로 제공 → 모델이 힌트를 그대로 답으로 출력
```
Electivire 입력 → RAG 힌트: "This is Electabuzz..." → 모델 출력: "Electabuzz"
Lickilicky 입력 → RAG 힌트: "This is Lickitung..."  → 모델 출력: "Lickitung"
```
Vanilla는 힌트 없이 유추하므로 오히려 맞출 가능성이 있다 (Leafeon, Glaceon, Sylveon은 base model이 이미 알고 있음).

### 2. RAG 거리와 정확도 무관계
낮은 거리(=좋은 매칭)가 OOD 정답을 보장하지 않는다.

| 포켓몬 | Dist | RAG 결과 |
|---|---|---|
| mime-jr | 0.055 | ❌ Jigglypuff |
| gardevoir | 0.032 | ✅ (base model이 이미 앎) |
| metagross | 0.033 | ❌ Groudon |

### 3. Vanilla의 편향 패턴
OOD에서 Vanilla는 "Zoroark"와 "Groudon"으로 극도로 편향된다.
- Zoroark로 오답: electivire, scyther(v7), absol, lucario, garchomp, zekrom, solgaleo, lunala, zacian, zamazenta (10건)
- Groudon으로 오답: munchlax, kyurem, yveltal, zekrom 등

이는 base model(Qwen2-VL-7B)의 특정 시각적 패턴에 대한 과잉 반응으로 보인다.

### 4. Phase 1 v5 OOD 결과와 비교
- Phase 1 v5 OOD(n=6): 모든 모델 16.7%
- Phase 2 v9 OOD(n=30): Vanilla 16.7%, RAG 13.3%
- Phase 1 수치가 우연히 실제와 가까웠음 (OOD는 설계 결함의 영향이 작은 영역이었음)

### 결론
OOD 문제의 근본 해결책은 RAG DB를 Gen1-9 전 세대로 확장하는 것이다.
현재 Gen1-2 DB만으로는 RAG가 OOD에서 도움이 안 되며 오히려 역효과 가능.

## 상세 결과

| 포켓몬 | 분류 | 관련 Gen1-2 | Vanilla | RAG | Dist |
|---|---|---|---|---|---|
| **electivire** (에레키블) | evolution | electabuzz (에레브) | ❌ This is the Pokémon Zoroark (Zoroark in ... | ❌ Pokemon: Electabuzz
Korean Name: 에레브... | 0.198 |
| **magmortar** (마그모라타) | evolution | magmar (마그마) | ❌ This is Groudon (구로돈) in English and Kor... | ❌ This is Magmar (마그마).... | 0.057 |
| **lickilicky** (내룸벨트) | evolution | lickitung (내루미) | ❌ This is the Pokémon named "Dugtrio" in E... | ❌ This is Lickitung (내루미).... | 0.160 |
| **togekiss** (토게키스) | evolution | togetic/togepi (토게틱/토게피) | ❌ This is the Pokémon "Swellow" in English... | ❌ Pokemon: Togetic
Korean Name: 토게틱... | 0.200 |
| **leafeon** (리피아) | evolution | eevee (이브이) | ✅ This is a Leafeon (leaf-ee-ahn) in Engli... | ❌ This is Flareon (부스터).... | 0.204 |
| **glaceon** (글레이시아) | evolution | eevee (이브이) | ✅ This is Glaceon (Glaceon) in English and... | ❌ Pokemon: Umbreon
Korean Name: 블래키... | 0.241 |
| **sylveon** (님피아) | evolution | eevee (이브이) | ✅ This is Sylveon (Sylveon) in English and... | ✅ Pokemon: Sylveon
Korean Name: 실베온... | 0.213 |
| **munchlax** (먹고자) | evolution | snorlax (잠만보) | ❌ This is the Pokémon named "Dugtrio" in E... | ❌ Pokemon: Snorlax
Korean Name: 잠만보... | 0.254 |
| **mime-jr** (흉내내기) | evolution | mr-mime (마임맨) | ❌ This is the Pokémon "Mr. Mime" in Englis... | ❌ Pokemon: Jigglypuff
Korean Name: 짭짤두 (Ja... | 0.055 |
| **bonsly** (꼬지모) | evolution | sudowoodo (나무킹) | ❌ This is a Chikorita, which is called "치코... | ❌ Pokemon: Sudowoodo
Korean Name: 꼬지모... | 0.231 |
| **ralts** (랄토스) | visual | gardevoir 계열 | ❌ This is the Pokémon "Bunnelby" in Englis... | ❌ The Pokémon in the picture is **Bouffleu... | 0.045 |
| **gardevoir** (가디안) | visual | humanoid 실루엣 | ✅ This is the Pokémon called "Gardevoir" i... | ✅ This is the Pokémon called "Gardevoir" i... | 0.032 |
| **metagross** (메타그로스) | visual | psychic 계열 | ❌ This is the Pokémon "Groudon" in English... | ❌ This is the Pokémon "Groudon" in English... | 0.033 |
| **absol** (앱솔) | visual | 흰색 포켓몬 | ❌ This is the Pokémon Zoroark (ゼロアーツ, Zero... | ❌ Pokemon: Zoroark
Korean Name: 코로아크... | 0.270 |
| **lucario** (루카리오) | visual | riolu 계열 | ❌ This is Zoroark (ゼラックス, Zerakku).... | ❌ This is Mewtwo (뮤츠).... | 0.268 |
| **garchomp** (한카리아스) | visual | dragon 계열 | ❌ This is the Pokémon Zoroark (Zoroark in ... | ❌ This Pokémon is Groudon (구돈, Gudon) in E... | 0.057 |
| **infernape** (infernape) | visual | chimchar 계열 | ❌ This is a picture of the Pokémon "Arcani... | ✅ This is Infernape, a Fire and Fighting-t... | 0.020 |
| **empoleon** (엠페르트) | visual | piplup 계열 | ❌ This is the Pokémon "Glaceon" in English... | ❌ This is the Pokémon named "Glaceon" in E... | 0.038 |
| **torterra** (토대부기) | visual | turtwig 계열 | ❌ This is the Pokémon named "Treecko" in E... | ❌ This is Rhydon (코뿌리).... | 0.232 |
| **roserade** (로즈레이드) | visual | budew/roselia 계열 | ❌ This is the Pokémon called "Gardevoir" i... | ❌ Pokemon: Deoxys (Deoxys)
Korean Name: 도크... | 0.210 |
| **zekrom** (제크롬) | distinct | 없음 | ❌ This is Zoroark (ゼラックス, Zerakku) from th... | ❌ This is Groudon (그로돈) in English and Kor... | 0.044 |
| **reshiram** (레시라무) | distinct | 없음 | ❌ This is the Pokémon Articuno. In Korean,... | ❌ Pokemon: Arceus
Korean Name: 아르세우스... | 0.187 |
| **kyurem** (큐레무) | distinct | 없음 | ❌ This is Groudon (Groudon) in English and... | ❌ This is the Pokémon Zoroark (Zoroark in ... | 0.026 |
| **xerneas** (제르네아스) | distinct | 없음 | ✅ This is the Pokémon Xerneas.

Pokemon: X... | ✅ Pokemon: Xerneas
Korean Name: 셀렌니스... | 0.042 |
| **yveltal** (이벨타르) | distinct | 없음 | ❌ This is the Pokémon "Groudon" in English... | ❌ This is the Pokémon Zoroark (Zoroark in ... | 0.221 |
| **solgaleo** (솔가레오) | distinct | 없음 | ❌ This is Zoroark (ゼラックス, Zerakku) from th... | ❌ Pokemon: Giratina
Korean Name: 기라티나... | 0.299 |
| **lunala** (루나아라) | distinct | 없음 | ❌ This is Zoroark (Zoroark) in English and... | ❌ Pokemon: Giratina
Korean Name: 기라티나... | 0.337 |
| **zacian** (자시안) | distinct | 없음 | ❌ This is Zoroark (ゼラックス, Zerakku) from th... | ❌ Pokemon: Zoroark
Korean Name: 코로아크... | 0.393 |
| **zamazenta** (자마젠타) | distinct | 없음 | ❌ This is Zoroark (ゼラックス, Zerakku) from th... | ❌ This is Zoroark (조로아크).... | 0.328 |
| **eternatus** (무한다이노) | distinct | 없음 | ❌ This is a Zoroark (조로아크) from the Pokémo... | ❌ Pokemon: Zoroark
Korean Name: 코로아크... | 0.441 |
