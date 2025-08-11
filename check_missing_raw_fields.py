import pandas as pd

df = pd.read_csv("data/multimodal_train.tsv", sep='\t')
print(df.columns)

def check_missing_fields():
    for split in ['train', 'validate', 'test_public']:
        path = f"data/multimodal_{split}.tsv"
        df = pd.read_csv(path, sep='\t')

        total_rows = len(df)
        missing_title = df['title'].isnull().sum() + (df['title'] == '').sum()
        missing_image_url = df['image_url'].isnull().sum() + (df['image_url'] == '').sum()

        print(f"--- {split.upper()} ---")
        print(f"Total rows: {total_rows}")
        print(f"Missing titles: {missing_title}")
        print(f"Missing image_url: {missing_image_url}")
        print()

if __name__ == "__main__":
    check_missing_fields()
