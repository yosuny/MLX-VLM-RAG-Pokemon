from datasets import load_dataset
import os
import json
from PIL import Image

def setup_pokemon_data(num_samples=50, output_dir="data_pokemon"):
    print(f"Downloading {num_samples} samples from diffusers/pokemon-gpt4-captions...")
    
    # Load dataset in streaming mode to avoid downloading everything
    ds = load_dataset("diffusers/pokemon-gpt4-captions", split="train", streaming=True)
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    data_entries = []
    
    # Mapping for first 50 pokemon
    pokemon_map = {
        "Bulbasaur": "이상해씨", "Ivysaur": "이상해풀", "Venusaur": "이상해꽃",
        "Charmander": "파이리", "Charmeleon": "리자드", "Charizard": "리자몽",
        "Squirtle": "꼬부기", "Wartortle": "어니부기", "Blastoise": "거북왕",
        "Caterpie": "캐터피", "Metapod": "단데기", "Butterfree": "버터플",
        "Weedle": "뿔충이", "Kakuna": "딱충이", "Beedrill": "독침붕",
        "Pidgey": "구구", "Pidgeotto": "피죤", "Pidgeot": "피죤투",
        "Rattata": "꼬렛", "Raticate": "레트라", "Spearow": "깨비참",
        "Fearow": "깨비드릴조", "Ekans": "아보", "Arbok": "아보크",
        "Pikachu": "피카츄", "Raichu": "라이츄", "Sandshrew": "모래두지",
        "Sandslash": "고지", "Nidoran♀": "니드런♀", "Nidorina": "니드리나",
        "Nidoqueen": "니드퀸", "Nidoran♂": "니드런♂", "Nidorino": "니드리노",
        "Nidoking": "니드킹", "Clefairy": "삐삐", "Clefable": "픽시",
        "Vulpix": "식스테일", "Ninetales": "나인테일", "Jigglypuff": "푸린",
        "Wigglytuff": "푸크린", "Zubat": "주뱃", "Golbat": "골뱃",
        "Oddish": "뚜벅쵸", "Gloom": "냄새꼬", "Vileplume": "라플레시아",
        "Paras": "파라스", "Parasect": "파라섹트", "Venonat": "콘팡",
        "Venomoth": "도나리", "Diglett": "디그다"
    }
    
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
            
        image = item["image"]
        caption = item["text"]
        
        # Simple heuristic to find pokemon name in caption (usually starts with "A [Pokemon]")
        # This is not perfect but good for demo
        found_name_en = "Pokemon"
        found_name_kr = ""
        
        for name_en, name_kr in pokemon_map.items():
            if name_en.lower() in caption.lower():
                found_name_en = name_en
                found_name_kr = name_kr
                break
        
        # Save image
        image_name = f"pokemon_{i:03d}.jpg"
        image_path = os.path.join(images_dir, image_name)
        image.save(image_path)
        
        # Create VLM chat format entry with Korean augmentation
        if found_name_kr:
            answer_text = f"This is {found_name_en} ({found_name_kr}). {caption}"
        else:
            answer_text = caption # Fallback if name not found in map
        
        entry = {
            "images": [image_path],
            "messages": [
                {
                    "role": "user", 
                    "content": "What pokemon is this?"
                },
                {
                    "role": "assistant", 
                    "content": answer_text
                }
            ]
        }
        
        data_entries.append(entry)
        print(f"[{i+1}/{num_samples}] Saved {image_name}: {caption[:30]}...")

    # Save JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in data_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nData preparation complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Training data saved to: {jsonl_path}")

if __name__ == "__main__":
    setup_pokemon_data()
