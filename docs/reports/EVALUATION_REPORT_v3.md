# Evaluation Report v3: Final Model Comparison

**Prompt**: `What is this? Answer in English and Korean.`
**Configurations**:
1. **Vanilla**: Base Qwen2-VL-7B-Instruct (4-bit)
2. **RAG**: Base + Vector Retrieval Hint
3. **Tuned**: Custom Fine-tuned + Fused + Quantized Model (Ours)

| Type | Image | Vanilla | RAG | Tuned (Ours) |
| :---: | :---: | --- | --- | --- |
| **TRAIN** | ![](../../data/pokemon/images/pokemon_117.jpg)<br><sub>train</sub> | This is a picture of Umbreon, a Pokémon from the Pokémon series. | This is Umbreon (블래키). It is a sleek, midnight blue-colored Pokémon known for its bright, crimson eyes and yellow ring-like patterns on its ears and body. | **This is a picture of a Pokémon called Umbreon.** |
| **TRAIN** | ![](../../data/pokemon/images/pokemon_025.jpg)<br><sub>train</sub> | This is a picture of a star-shaped object with a gem in the center. It appears to be a stylized representation rather than a real object.  In English: This is a picture of a star-shaped object with a gem in the center.  In Korean: 이는 설탕형의 물체입니다. 설탕형의 물체의 중앙에 보석이 있습니다. | This is a picture of a Pokémon called Staryu. It is a five-pointed, star-shaped Pokémon with a reddish center surrounded by a yellow and beige body. | **This is a picture of a star-shaped object with a gem in the center.** |
| **VALID** | ![](../../data/pokemon/images/pokemon_440.jpg)<br><sub>valid</sub> | This is a picture of a Pokémon called Glaceon. It is a type of Pokémon that is part of the Ice type. | This is a picture of a Pokémon called Riolu. It appears to be preparing for a battle with a determined expression. | **This is a picture of a Pokémon called Glaceon. It is a Ice-type Pokémon.** |
| **VALID** | ![](../../data/pokemon/images/pokemon_411.jpg)<br><sub>valid</sub> | This is a picture of a Pokémon called "Gallade" (가라데). | This is a colorful cartoon illustration of Gastrodon, a sea slug-like Pokémon with blue, green, and yellow tones. | **This is a picture of a Pokémon called "Gallade" from the Pokémon series.** |
