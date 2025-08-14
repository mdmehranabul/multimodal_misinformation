import joblib
import os

MODEL_SAVE_DIR = "saved_models"
splits = ['train', 'valid', 'test']

for split in splits:
    file_path = os.path.join(MODEL_SAVE_DIR, f"{split}_processed.pkl")
    print(f"Checking {split} split checkpoint...")

    try:
        df = joblib.load(file_path)
        print(f"{split.capitalize()} file loaded successfully! No corruption detected.")

        total_rows = len(df)
        valid_rows = df.dropna(subset=['text_embedding', 'image_embedding']).shape[0]

        print(f"Total rows loaded: {total_rows}")
        print(f"Rows with both embeddings present: {valid_rows}\n")
    except Exception as e:
        print(f"Error loading {split} file: {e}\n")
