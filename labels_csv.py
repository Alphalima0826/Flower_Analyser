import os
import pandas as pd
import random

root_dir = "data"
flower_types = os.listdir(root_dir)

data = []

dominant_colors_map = {
    "daisy": "white",
    "dandelion": "yellow",
    "rose": "red",
    "sunflower":"yellow",
    "tulip" : "pink"
}

for flower_type in flower_types:
    folder_path = os.path.join(root_dir, flower_type)
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg','.jpeg','.png')):
            img_path = f"{flower_type}/{filename}"
            data.append({
                "image_filename":img_path,
                "flower_type" : flower_type,
                "dominant_colors" : dominant_colors_map.get(flower_type,"unknown"),
                "Linalool": round(random.uniform(0.1,.05),2),
                "Geraniol":round(random.uniform(0.1,0.5),2),
                "Citronellol":round(random.uniform(0.1,0.5),2),
            })

df = pd.DataFrame(data)
df.to_csv("data/labels.csv")