import os
import pickle
import pandas as pd

MODEL_SAVE_DIR = "saved_models"

def load_df_checkpoint(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load checkpoint {filename}: {e}")
            return None
    print(f"Checkpoint {filename} not found.")
    return None

def main():
    for split_name in ['train', 'valid', 'test']:
        checkpoint_file = os.path.join(MODEL_SAVE_DIR, f"{split_name}_processed.pkl")
        df = load_df_checkpoint(checkpoint_file)
        if df is not None:
            print(f"\n=== {split_name.upper()} DATA ===")
            # Show rows where embeddings are missing to focus on problem cases
            missing_text_emb = df[df['text_embedding'].isnull()]
            missing_img_emb = df[df['image_embedding'].isnull()]

            print(f"Total rows: {len(df)}")
            print(f"Rows with missing text_embedding: {len(missing_text_emb)}")
            print(f"Rows with missing image_embedding: {len(missing_img_emb)}")

            print("\nSample rows with missing text_embedding:")
            print(missing_text_emb[['title', 'image_path', 'text_embedding', 'image_embedding']].head())

            print("\nSample rows with missing image_embedding:")
            print(missing_img_emb[['title', 'image_path', 'text_embedding', 'image_embedding']].head())

            print("\nSample rows with embeddings present:")
            has_both = df.dropna(subset=['text_embedding', 'image_embedding'])
            print(has_both[['title', 'image_path', 'text_embedding', 'image_embedding']].head())

if __name__ == "__main__":
    main()