import os
import pandas as pd

MODEL_SAVE_DIR = "saved_models"  # Adjust if your folder is different

def check_checkpoint(file_path):
    print(f"Checking checkpoint: {file_path}")
    if not os.path.exists(file_path):
        print("  File not found!")
        return

    df = pd.read_pickle(file_path)
    total_rows = len(df)
    print(f"  Total rows in checkpoint: {total_rows}")

    has_text_emb = 'text_embedding' in df.columns
    has_img_emb = 'image_embedding' in df.columns
    print(f"  Has 'text_embedding' column? {has_text_emb}")
    print(f"  Has 'image_embedding' column? {has_img_emb}")

    if has_text_emb:
        missing_text = df['text_embedding'].isnull().sum()
        print(f"  Missing 'text_embedding' values: {missing_text} ({missing_text/total_rows*100:.2f}%)")
    else:
        print("  No 'text_embedding' column found")

    if has_img_emb:
        missing_img = df['image_embedding'].isnull().sum()
        print(f"  Missing 'image_embedding' values: {missing_img} ({missing_img/total_rows*100:.2f}%)")
    else:
        print("  No 'image_embedding' column found")

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"{split}_processed.pkl")
        check_checkpoint(checkpoint_path)
        print("-" * 40)
