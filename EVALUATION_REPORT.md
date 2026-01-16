# 🕵️ VLM Evaluation Report: Pokemon Identification
**Report Version**: Final (Ground Truth Comparison)

Comparing Vanilla Qwen2-VL, RAG Retrieval, and LoRA Tuned Model.

## Test Case: pokemon_000.jpg
| Query Image | Model Results |
| :---: | --- |
| ![Query](data_pokemon/images/pokemon_000.jpg) | **Ground Truth**: **Bulbasaur (이상해씨)** <br><br> **Vanilla Model**: ✅ Correct <br> *"This is a picture of a Pokémon. The name of this Pokémon is "Bulbasaur" in English and "버브라우" in Korean."* <br><br> **Tuned Model**: ❌ Incorrect <br> *"and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and"* |

### RAG Context
Retrieved Similar Image (Dist: 0.0000):
**Info**: *This is Bulbasaur (이상해씨). A cheerful Bulbasaur ready for its next Pokémon adventure.*

![RAG](data_pokemon/images/pokemon_000.jpg)

---
## Test Case: pokemon_024.jpg
| Query Image | Model Results |
| :---: | --- |
| ![Query](data_pokemon/images/pokemon_024.jpg) | **Ground Truth**: **Butterfree (버터플)** <br><br> **Vanilla Model**: ❌ Incorrect <br> *"This Pokemon's name is "Pikachu" in English and "피카츄" in Korean."* <br><br> **Tuned Model**: ❌ Incorrect <br> *"and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and"* |

### RAG Context
Retrieved Similar Image (Dist: 0.0000):
**Info**: *This is Butterfree (버터플). A cartoon illustration of Butterfree, a butterfly-like Pokémon character w...*

![RAG](data_pokemon/images/pokemon_024.jpg)

---
## Test Case: pokemon_043.jpg
| Query Image | Model Results |
| :---: | --- |
| ![Query](data_pokemon/images/pokemon_043.jpg) | **Ground Truth**: An illustration of Jolteon, the Electric-type evol... <br><br> **Vanilla Model**: ❌ Incorrect <br> *"This Pokemon's name is "Pikachu" in English and "피카츄" in Korean."* <br><br> **Tuned Model**: ❌ Incorrect <br> *"and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and"* |

### RAG Context
Retrieved Similar Image (Dist: 0.0000):
**Info**: *An illustration of Jolteon, the Electric-type evolution of Eevee from the Pokémon series, with its d...*

![RAG](data_pokemon/images/pokemon_043.jpg)

---
