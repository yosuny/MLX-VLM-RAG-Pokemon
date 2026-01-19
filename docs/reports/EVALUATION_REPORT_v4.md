# Evaluation Report v4: Hinted Prompt Comparison

**Prompt**: `What pokemon is this? Answer in English and Korean.`
**Configurations**:
1. **Vanilla**: Base Qwen2-VL-7B-Instruct (4-bit)
2. **RAG**: Base + Vector Retrieval Hint
3. **Tuned**: Custom Fine-tuned + Fused + Quantized Model (Ours)

| Type | Image | Vanilla | RAG | Tuned (Ours) |
| :---: | :---: | --- | --- | --- |
| **TRAIN** | ![](../../data/pokemon/images/pokemon_117.jpg)<br><sub>train</sub> | This is Umbreon. | This is Umbreon (블래키). | **This is Umbreon.** |
| **TRAIN** | ![](../../data/pokemon/images/pokemon_025.jpg)<br><sub>train</sub> | This is the Pokémon called "Staraptor" in English and "별사다리" in Korean. | This is the Pokémon called "Staraptor" in English and "별오리" in Korean. | **This is the Pokémon called "Staraptor" in English and "별사다리" in Korean.** |
| **VALID** | ![](../../data/pokemon/images/pokemon_440.jpg)<br><sub>valid</sub> | This is the Pokémon Umbreon. | This is the Pokémon Riolu. | **This is the Pokémon named "Glaceon" in English and "얼음조" in Korean.** |
| **VALID** | ![](../../data/pokemon/images/pokemon_411.jpg)<br><sub>valid</sub> | This is the Pokémon named "Gallade" in English and "가라데" in Korean. | This is Gastrodon, a sea slug-like Pokémon. It is a Water/Poison type Pokémon.  Pokemon: Gastrodon Pokemon: 가스토돈 | **This is the Pokémon named "Gallade" in English and "가라데" in Korean.** |
