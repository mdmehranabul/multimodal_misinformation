# utils/preprocessing.py

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

def load_and_clean_data(path, sample_size=None, seed=42):
    df = pd.read_csv(path, sep="\t")
    df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['hasImage'] = df['hasImage'].astype(str).str.upper() == 'TRUE'
    df = df[df['hasImage']]
    df = df.drop_duplicates(subset='id')

    df['2_way_label_name'] = df['2_way_label'].map({0: 'Misleading', 1: 'Real'})
    df['6_way_label_name'] = df['6_way_label'].map({
        0: 'Real', 1: 'Satire', 2: 'Misleading', 4: 'Manipulated',
        5: 'False Connection', 6: 'Imposter Content'
    })

    if sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df


def download_image(row, image_folder='images'):
    try:
        os.makedirs(image_folder, exist_ok=True)
        path = os.path.join(image_folder, f"{row['id']}.jpg")
        if os.path.exists(path):
            return path  # Skip download if already exists

        response = requests.get(row['image_url'], timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(path)
        return path
    except:
        return None