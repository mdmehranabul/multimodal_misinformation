import sys
sys.path.append(".")
from tqdm import tqdm
tqdm.pandas()
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import pickle

from utils.preprocessing import load_and_clean_data, download_image
from utils.embedding_utils import get_text_embedding, get_image_embedding
from utils.analysis import plot_train_label_distributions

from transformers import BertTokenizer, BertModel, ViTImageProcessor, ViTModel
from torchvision import transforms

import torch.nn as nn

# Train model with checkpointing on best validation accuracy
def train_model(model, loader, device, epochs=10, optimizer=None, val_loader=None, checkpoint_path=None):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    best_val_acc = 0.0
    best_model_wts = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

        if val_loader is not None:
            val_acc, val_f1 = evaluate_model(model, val_loader, device)
            print(f"Validation Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Save best model
            if checkpoint_path is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

    # Load best weights before returning if available
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model

# Evaluate model function remains unchanged
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1


# Seed everything for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# Paths and device config
TRAIN_PATH = "data/multimodal_train.tsv"
VALID_PATH = "data/multimodal_validate.tsv"
TEST_PATH  = "data/multimodal_test_public.tsv"
IMAGE_DIR = "images/"
MODEL_SAVE_DIR = "saved_models/unimodal"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained BERT and ViT models for embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
vit_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=vit_extractor.image_mean, std=vit_extractor.image_std),
])

# Function to filter rows with missing text or images
def filter_missing_text_or_image(df):
    return df[(df['title'].notnull()) & (df['title'] != '') & (df['image_url'].notnull()) & (df['image_url'] != '')].reset_index(drop=True)

# Checkpoint save/load helpers
def save_df_checkpoint(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)

def load_df_checkpoint(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {filename}: {e}")
            return None
    return None

def process_split_with_checkpoint_skip_done(df, split_name, checkpoint_file, batch_size=1000):
    import math
    from tqdm import tqdm

    if 'text_embedding' not in df.columns:
        df['text_embedding'] = None
    if 'image_embedding' not in df.columns:
        df['image_embedding'] = None
    if 'image_path' not in df.columns:
        df['image_path'] = None

    total_rows = len(df)
    num_batches = math.ceil(total_rows / batch_size)

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {split_name} batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx].copy()

        all_text_emb = batch_df['text_embedding'].apply(lambda x: x is not None).all()
        all_img_emb = batch_df['image_embedding'].apply(lambda x: x is not None).all()
        if all_text_emb and all_img_emb:
            print(f"Skipping {split_name} rows {start_idx} to {end_idx} - embeddings already present.")
            continue

        print(f"Processing {split_name} rows {start_idx} to {end_idx}...")

        missing_img_path_mask = batch_df['image_path'].isnull() | (batch_df['image_path'] == '')
        if missing_img_path_mask.any():
            batch_df.loc[missing_img_path_mask, 'image_path'] = batch_df.loc[missing_img_path_mask].progress_apply(
                lambda row: download_image(row, IMAGE_DIR), axis=1)

        missing_text_emb_mask = batch_df['text_embedding'].apply(lambda x: x is None)
        if missing_text_emb_mask.any():
            batch_df.loc[missing_text_emb_mask, 'text_embedding'] = batch_df.loc[missing_text_emb_mask, 'title'].progress_apply(
                lambda x: get_text_embedding(x, bert_tokenizer, bert_model, device))

        missing_img_emb_mask = batch_df['image_embedding'].apply(lambda x: x is None)
        if missing_img_emb_mask.any():
            batch_df.loc[missing_img_emb_mask, 'image_embedding'] = batch_df.loc[missing_img_emb_mask, 'image_path'].progress_apply(
                lambda x: get_image_embedding(x, image_transform, vit_model, device))

        batch_df.dropna(subset=['text_embedding', 'image_embedding'], inplace=True)

        df.loc[batch_df.index, ['text_embedding', 'image_embedding', 'image_path']] = batch_df.loc[:, ['text_embedding', 'image_embedding', 'image_path']]

        print(f"Saving checkpoint for {split_name} rows {start_idx}-{end_idx}...")
        save_df_checkpoint(df, checkpoint_file)
        print(f"Checkpoint saved.")

    return df


# Load and preprocess data splits
print("Loading and cleaning data...")
train_df = load_and_clean_data(TRAIN_PATH, sample_size=75000, seed=42)
valid_df = load_and_clean_data(VALID_PATH, sample_size=4000, seed=42)
test_df  = load_and_clean_data(TEST_PATH,  sample_size=4000, seed=42)

train_df = filter_missing_text_or_image(train_df)
valid_df = filter_missing_text_or_image(valid_df)
test_df = filter_missing_text_or_image(test_df)

for df, split_name in zip([train_df, valid_df, test_df], ['train', 'valid', 'test']):
    checkpoint_file = os.path.join(MODEL_SAVE_DIR, f"{split_name}_processed.pkl")
    cached_df = load_df_checkpoint(checkpoint_file)
    if cached_df is not None:
        print(f"Loaded {split_name} split from checkpoint.")
        if split_name == 'train':
            train_df = cached_df
        elif split_name == 'valid':
            valid_df = cached_df
        else:
            test_df = cached_df
        continue

    print(f"\nProcessing {split_name} split with checkpointing and skipping done rows...")
    processed_df = process_split_with_checkpoint_skip_done(df, split_name, checkpoint_file, batch_size=1000)
    if split_name == 'train':
        train_df = processed_df
    elif split_name == 'valid':
        valid_df = processed_df
    else:
        test_df = processed_df

train_df = train_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)
valid_df = valid_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)
test_df = test_df.dropna(subset=['text_embedding', 'image_embedding']).reset_index(drop=True)

plot_train_label_distributions(train_df)


# Define unimodal dataset classes
class TextDataset(Dataset):
    def __init__(self, df):
        self.text_embeddings = list(df['text_embedding'])
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.text_embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

class ImageDataset(Dataset):
    def __init__(self, df):
        self.image_embeddings = list(df['image_embedding'])
        self.labels = df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.image_embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

# Define unimodal classifiers
class TextClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6):
        super(TextClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class ImageClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=6):
        super(ImageClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)


# Train unimodal models for each label type
for label_type in ['2_way_label_name', '6_way_label_name']:
    print(f"\n========== {label_type.upper()} UNIMODAL TRAINING ==========")

    # Label encoding
    le = LabelEncoder()
    combined_labels = pd.concat([train_df[label_type], valid_df[label_type], test_df[label_type]])
    le.fit(combined_labels)

    train_df['label'] = le.transform(train_df[label_type])
    valid_df['label'] = le.transform(valid_df[label_type])
    test_df['label']  = le.transform(test_df[label_type])

    output_dim = len(le.classes_)

    # Prepare datasets and loaders
    train_text_dataset = TextDataset(train_df)
    valid_text_dataset = TextDataset(valid_df)
    test_text_dataset  = TextDataset(test_df)

    train_image_dataset = ImageDataset(train_df)
    valid_image_dataset = ImageDataset(valid_df)
    test_image_dataset  = ImageDataset(test_df)

    train_text_loader = DataLoader(train_text_dataset, batch_size=32, shuffle=True)
    valid_text_loader = DataLoader(valid_text_dataset, batch_size=32)
    test_text_loader  = DataLoader(test_text_dataset, batch_size=32)

    train_image_loader = DataLoader(train_image_dataset, batch_size=32, shuffle=True)
    valid_image_loader = DataLoader(valid_image_dataset, batch_size=32)
    test_image_loader  = DataLoader(test_image_dataset, batch_size=32)

    # Text-only model training
    print("Training Text-Only Model...")
    text_model = TextClassifier(output_dim=output_dim).to(device)
    text_optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-4)
    text_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"text_model_{label_type}_best.pt")
    text_model = train_model(text_model, train_text_loader, device=device, epochs=10, optimizer=text_optimizer, val_loader=valid_text_loader, checkpoint_path=text_checkpoint_path)
    text_val_acc, text_val_f1 = evaluate_model(text_model, valid_text_loader, device=device)
    text_test_acc, text_test_f1 = evaluate_model(text_model, test_text_loader, device=device)
    print(f"Text Model - Val Acc: {text_val_acc:.4f}, Val F1: {text_val_f1:.4f}")
    print(f"Text Model - Test Acc: {text_test_acc:.4f}, Test F1: {text_test_f1:.4f}")
    torch.save(text_model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"text_model_{label_type}.pt"))

    # Image-only model training
    print("Training Image-Only Model...")
    image_model = ImageClassifier(output_dim=output_dim).to(device)
    image_optimizer = torch.optim.Adam(image_model.parameters(), lr=1e-4)
    image_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"image_model_{label_type}_best.pt")
    image_model = train_model(image_model, train_image_loader, device=device, epochs=10, optimizer=image_optimizer, val_loader=valid_image_loader, checkpoint_path=image_checkpoint_path)
    image_val_acc, image_val_f1 = evaluate_model(image_model, valid_image_loader, device=device)
    image_test_acc, image_test_f1 = evaluate_model(image_model, test_image_loader, device=device)
    print(f"Image Model - Val Acc: {image_val_acc:.4f}, Val F1: {image_val_f1:.4f}")
    print(f"Image Model - Test Acc: {image_test_acc:.4f}, Test F1: {image_test_f1:.4f}")
    torch.save(image_model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"image_model_{label_type}.pt"))