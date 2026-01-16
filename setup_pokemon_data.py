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
        "Venomoth": "도나리", "Diglett": "디그다", "Dugtrio": "닥트리오",
        "Meowth": "나옹", "Persian": "페르시온", "Psyduck": "고라파덕",
        "Golduck": "골덕", "Mankey": "망키", "Primeape": "성원숭",
        "Growlithe": "가디", "Arcanine": "윈디", "Poliwag": "발챙이",
        "Poliwhirl": "슈륙챙이", "Poliwrath": "강챙이", "Abra": "캐이시",
        "Kadabra": "윤겔라", "Alakazam": "후딘", "Machop": "알통몬",
        "Machoke": "근육몬", "Machamp": "괴력몬", "Bellsprout": "모다피",
        "Weepinbell": "우츠동", "Victreebel": "우츠보트", "Tentacool": "왕눈해",
        "Tentacruel": "독파리", "Geodude": "꼬마돌", "Graveler": "데구리",
        "Golem": "딱구리", "Ponyta": "포니타", "Rapidash": "날쌩마",
        "Slowpoke": "야돈", "Slowbro": "야도란", "Magnemite": "코일",
        "Magneton": "레어코일", "Farfetch'd": "파오리", "Doduo": "두두",
        "Dodrio": "두트리오", "Seel": "쥬쥬", "Dewgong": "쥬레곤",
        "Grimer": "질퍽이", "Muk": "질뻐기", "Shellder": "셀러",
        "Cloyster": "파르셀", "Gastly": "고오스", "Haunter": "고우스트",
        "Gengar": "팬텀", "Onix": "롱스톤", "Drowzee": "슬리프",
        "Hypno": "슬리퍼", "Krabby": "크랩", "Kingler": "킹크랩",
        "Voltorb": "찌리리공", "Electrode": "붐볼", "Exeggcute": "아라리",
        "Exeggutor": "나시", "Cubone": "탕구리", "Marowak": "텅구리",
        "Hitmonlee": "시라소몬", "Hitmonchan": "홍수몬", "Lickitung": "내루미",
        "Koffing": "또가스", "Weezing": "또도가스", "Rhyhorn": "뿔카노",
        "Rhydon": "코뿌리", "Chansey": "럭키", "Tangela": "덩쿠리",
        "Kangaskhan": "캥카", "Horsea": "쏘드라", "Seadra": "시드라",
        "Goldeen": "콘치", "Seaking": "왕콘치", "Staryu": "별가사리",
        "Starmie": "아쿠스타", "Mr. Mime": "마임맨", "Scyther": "스라크",
        "Jynx": "루주라", "Electabuzz": "에레브", "Magmar": "마그마",
        "Pinsir": "쁘사이저", "Tauros": "켄타로스", "Magikarp": "잉어킹",
        "Gyarados": "갸라도스", "Lapras": "라프라스", "Ditto": "메타몽",
        "Eevee": "이브이", "Vaporeon": "샤미드", "Jolteon": "쥬피썬더",
        "Flareon": "부스터", "Porygon": "폴리곤", "Omanyte": "암나이트",
        "Omastar": "암스타", "Kabuto": "투구", "Kabutops": "투구푸스",
        "Aerodactyl": "프테라", "Snorlax": "잠만보", "Articuno": "프리져",
        "Zapdos": "썬더", "Moltres": "파이어", "Dratini": "미뇽",
        "Dragonair": "신뇽", "Dragonite": "망나뇽", "Mewtwo": "뮤츠",
        "Mew": "뮤", "Electivire": "에레키블", "Bouffalant": "버프론"
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
