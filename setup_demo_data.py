import os
import json
from PIL import Image, ImageDraw

def create_dummy_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. Create dummy images
    colors = ["red", "blue", "green"]
    shapes = ["circle", "rectangle", "triangle"]
    
    dataset = []
    
    for color in colors:
        for shape in shapes:
            filename = f"{color}_{shape}.png"
            filepath = os.path.join(images_dir, filename)
            
            # Create image
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            if shape == "circle":
                draw.ellipse((50, 50, 174, 174), fill=color)
            elif shape == "rectangle":
                draw.rectangle((50, 50, 174, 174), fill=color)
            elif shape == "triangle":
                draw.polygon([(112, 50), (50, 174), (174, 174)], fill=color)
            
            img.save(filepath)
            
            # Create VLM chat format entry
            # Qwen-VL often uses a specific format, but for fine-tuning we'll use a standard list of messages
            # which our trainer will convert.
            entry = {
                "image": filepath,
                "messages": [
                    {"role": "user", "content": "What shape and color is this?"},
                    {"role": "assistant", "content": f"This is a {color} {shape}."}
                ]
            }
            dataset.append(entry)
            
    # 2. Save to JSONL
    train_path = os.path.join(data_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Created {len(dataset)} items in {train_path}")

if __name__ == "__main__":
    create_dummy_data()
